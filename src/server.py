from utils import serialize, deserialize
from typing import Any, Dict, List
import tenseal as ts

class Server:
    def __init__(self):
        self.storage: Dict[str, List[bytes]] = {}
        self.running_sums: Dict[str, ts.CKKSVector] = {}
        self.data_counts: Dict[str, int] = {}

    def handle_request(self, request: str) -> str:
        request_dict = deserialize(request)
        context = ts.context_from(request_dict['context'])
        
        if request_dict['action'] == 'store':
            return serialize(self.store_data(context, request_dict['key'], request_dict['data'], request_dict['size']))
        elif request_dict['action'] == 'compute_average':
            return serialize(self.compute_average(context, request_dict['key']))
        elif request_dict['action'] == 'compute_overall_average':
            return serialize(self.compute_overall_average(context, request_dict['keys']))
        else:
            return serialize({'status': 'error', 'message': 'Invalid action'})

    def store_data(self, context: ts.Context, key: str, data: bytes, size: int) -> Dict[str, str]:
        if key not in self.storage:
            self.storage[key] = []
            self.running_sums[key] = ts.ckks_vector_from(context, data).sum()
            self.data_counts[key] = size
        else:
            self.storage[key].append(data)
            self.running_sums[key] += ts.ckks_vector_from(context, data).sum()
            self.data_counts[key] += size
        
        return {'status': 'success', 'message': 'Data stored'}

    def compute_average(self, context: ts.Context, key: str) -> Dict[str, Any]:
        if key not in self.storage:
            return {'status': 'error', 'message': 'Key not found'}
        
        encrypted_sum = self.running_sums[key]
        data_count = self.data_counts[key]
        encrypted_average = encrypted_sum * (1 / data_count)
        
        return {'status': 'success', 'result': encrypted_average.serialize()}

    def compute_overall_average(self, context: ts.Context, keys: List[str]) -> Dict[str, Any]:
        overall_sum = None
        overall_count = 0
        
        for key in keys:
            if key not in self.storage:
                return {'status': 'error', 'message': f'Key {key} not found'}
            
            if overall_sum is None:
                overall_sum = self.running_sums[key]
            else:
                overall_sum += self.running_sums[key]
            
            overall_count += self.data_counts[key]
        
        encrypted_average = overall_sum * (1 / overall_count)
        
        return {'status': 'success', 'result': encrypted_average.serialize()}
    