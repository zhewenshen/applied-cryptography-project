import tenseal as ts
import base64
import numpy as np
from typing import List, Dict, Any
import random

def serialize(obj):
    if isinstance(obj, bytes):
        return base64.b64encode(obj).decode('utf-8')
    elif isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize(v) for v in obj]
    else:
        return obj

def deserialize(obj):
    if isinstance(obj, str):
        try:
            return base64.b64decode(obj.encode('utf-8'))
        except:
            return obj
    elif isinstance(obj, dict):
        return {k: deserialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [deserialize(v) for v in obj]
    else:
        return obj

class TenSEALContext:
    @staticmethod
    def create_context():
        context = ts.context(ts.SCHEME_TYPE.CKKS, 8192, -1, [60, 40, 40, 60])
        context.global_scale = 2**40
        context.generate_galois_keys()
        return context
    
    def create_machine_learning_context():
        context = ts.context(ts.SCHEME_TYPE.CKKS, 8192, -1, [40, 21, 21, 21, 21, 21, 21, 40])
        context.global_scale = 2**21
        context.generate_galois_keys()
        return context

class Client:
    def __init__(self):
        self.context = TenSEALContext.create_context()
        self.model_training_context = TenSEALContext.create_machine_learning_context()
        self.model_inference_context = TenSEALContext.create_machine_learning_context()

    def encrypt_data(self, data: List[float]) -> bytes:
        encrypted_data = ts.ckks_vector(self.context, data)
        return encrypted_data.serialize()

    def decrypt_data(self, serialized_data: bytes) -> List[float]:
        encrypted_data = ts.ckks_vector_from(self.context, serialized_data)
        return encrypted_data.decrypt()

    def send_request(self, server: 'Server', request: Dict[str, Any]) -> Dict[str, Any]:
        request['context'] = self.context.serialize()
        if 'data' in request:
            request['data'] = self.encrypt_data(request['data'])
        
        serialized_request = serialize(request)
        response = server.handle_request(serialized_request)
        response_dict = deserialize(response)
        
        if 'result' in response_dict and isinstance(response_dict['result'], bytes):
            response_dict['result'] = self.decrypt_data(response_dict['result'])
        
        return response_dict

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

if __name__ == "__main__":
    client = Client()
    server = Server()

    # Test with different datasets
    datasets = [
        [random.uniform(0, 100) for _ in range(347)],
        [random.uniform(0, 100) for _ in range(443)],
        [random.uniform(0, 100) for _ in range(42)]
    ]

    all_data = []

    for i, data in enumerate(datasets):
        print(f"\nDataset {i + 1}:")
        # print(f"Original data: {data}")

        # Store data
        response = client.send_request(server, {
            'action': 'store',
            'key': f'dataset_{i}',
            'data': data,
            'size': len(data)
        })
        print(f"Store response: {response}")

        # Compute average
        response = client.send_request(server, {
            'action': 'compute_average',
            'key': f'dataset_{i}'
        })
        print(f"Compute average response: {response}")
        print(f"Actual average: {np.mean(data)}")

        all_data.extend(data)

    # Compute overall average across all datasets
    keys = [f'dataset_{i}' for i in range(len(datasets))]
    response = client.send_request(server, {
        'action': 'compute_overall_average',
        'keys': keys
    })
    print(f"\nCompute overall average response: {response}")
    print(f"Actual overall average: {np.mean(all_data)}")
