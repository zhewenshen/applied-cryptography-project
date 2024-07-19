from utils import serialize, deserialize
from typing import Any, Dict, List
import tenseal as ts

class Server:
    def __init__(self):
        self.storage: Dict[str, List[bytes]] = {}
        self.running_sums: Dict[str, ts.CKKSVector] = {}
        self.squared_sums: Dict[str, ts.CKKSVector] = {}
        self.data_counts: Dict[str, int] = {}

    def handle_request(self, request: str) -> str:
        request_dict = deserialize(request)
        context = ts.context_from(request_dict['context'])
        
        if request_dict['action'] == 'store':
            return serialize(self.store_data(context, request_dict['key'], request_dict['data'], request_dict['size']))
        elif request_dict['action'] == 'compute_average':
            return serialize(self.compute_average(context, request_dict['key']))
        elif request_dict['action'] == 'compute_variance':
            return serialize(self.compute_variance(context, request_dict['key']))
        elif request_dict['action'] == 'sd':
            return serialize(self.compute_standard_deviation(context, request_dict['key']))
        elif request_dict['action'] == 'compute_overall_average':
            return serialize(self.compute_overall_average(context, request_dict['keys']))
        else:
            return serialize({'status': 'error', 'message': 'Invalid action'})

    def store_data(self, context: ts.Context, key: str, data: bytes, size: int) -> Dict[str, str]:
        print(f"size: {size}")
        if key in self.storage:
            return {'status': 'error', 'message': 'Key already exists'}
        
        self.storage[key] = data
        vector = ts.ckks_vector_from(context, data)
        self.running_sums[key] = vector.sum()
        self.squared_sums[key] = (vector * vector).sum()
        self.data_counts[key] = size
        
        return {'status': 'success', 'message': 'Data stored'}

    def compute_average(self, context: ts.Context, key: str) -> Dict[str, Any]:
        if key not in self.storage:
            return {'status': 'error', 'message': 'Key not found'}
        
        print(f"key: {key}")
        
        encrypted_sum = self.running_sums[key]
        data_count = self.data_counts[key]
        encrypted_average = encrypted_sum * (1 / data_count)
        
        print(data_count)
        
        return {'status': 'success', 'result': encrypted_average.serialize()}

    def compute_variance(self, context: ts.Context, key: str) -> Dict[str, Any]:
        if key not in self.storage:
            return {'status': 'error', 'message': 'Key not found'}

        encrypted_sum = self.running_sums[key]
        encrypted_squared_sum = self.squared_sums[key]
        data_count = self.data_counts[key]

        # Mean of the elements
        encrypted_mean = encrypted_sum * (1 / data_count)

        # Mean of the squares
        encrypted_mean_of_squares = encrypted_squared_sum * (1 / data_count)

        # Variance: E(X^2) - (E(X))^2
        encrypted_variance = encrypted_mean_of_squares - (encrypted_mean * encrypted_mean)

        return {'status': 'success', 'result': encrypted_variance.serialize()}

    def compute_standard_deviation(self, context: ts.Context, key: str) -> Dict[str, Any]:
        variance_result = self.compute_variance(context, key)
        if variance_result['status'] == 'error':
            return variance_result
        
        encrypted_variance = ts.ckks_vector_from(context, deserialize(variance_result['result']))
        
        # Compute the square root of the variance
        encrypted_std_dev = encrypted_variance.polyval([0.5])
        
        return {'status': 'success', 'result': encrypted_std_dev.serialize()}

    def compute_overall_average(self, context: ts.Context, keys: List[str]) -> Dict[str, Any]:
        overall_sum = ts.CKKSVector(context, [0])
        overall_count = 0
        
        for key in keys:
            if key not in self.storage:
                return {'status': 'error', 'message': f'Key {key} not found'}
            
            overall_sum += self.running_sums[key]
            overall_count += self.data_counts[key]
        
        encrypted_average = overall_sum * (1 / overall_count)
        
        return {'status': 'success', 'result': encrypted_average.serialize()}
