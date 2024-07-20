from typing import Any, Dict, List
from utils import serialize, deserialize
from model import EncryptedLR
from cryptography.fernet import Fernet
import tenseal as ts


class Server:
    def __init__(self):
        self.storage: Dict[str, List[bytes]] = {}
        self.running_sums: Dict[str, ts.CKKSVector] = {}
        self.squared_sums: Dict[str, ts.CKKSVector] = {}
        self.data_counts: Dict[str, int] = {}
        self.training_data: Dict[str, Dict[str, List[ts.CKKSVector]]] = {}
        self.models: Dict[str, EncryptedLR] = {}
        self.key = b""

    def set_key(self, key: bytes):
        self.key = key

    def handle_request(self, request: str) -> str:
        cur_key = Fernet(self.key)
        request = cur_key.decrypt(request)
        request_dict = deserialize(request)
        context = ts.context_from(bytes.fromhex(request_dict['context']))

        if request_dict['action'] == 'store':
            return cur_key.encrypt(serialize(self.store_data(context, request_dict['key'], request_dict['data'], request_dict['size'])).encode('utf-8'))
        elif request_dict['action'] == 'compute_average':
            return cur_key.encrypt(serialize(self.compute_average(context, request_dict['key'])).encode('utf-8'))
        elif request_dict['action'] == 'compute_variance':
            return cur_key.encrypt(serialize(self.compute_variance(context, request_dict['key'])).encode('utf-8'))
        elif request_dict['action'] == 'sd':
            return cur_key.encrypt(serialize(self.compute_standard_deviation(context, request_dict['key'])).encode('utf-8'))
        elif request_dict['action'] == 'compute_overall_average':
            return cur_key.encrypt(serialize(self.compute_overall_average(context, request_dict['keys'])).encode('utf-8'))
        elif request_dict['action'] == 'store_training_data':
            return cur_key.encrypt(serialize(self.store_training_data(context, request_dict['key'], request_dict['training_data'])).encode('utf-8'))
        elif request_dict['action'] == 'initialize_model':
            return cur_key.encrypt(serialize(self.initialize_model(context, request_dict['key'], request_dict['n_features'])).encode('utf-8'))
        elif request_dict['action'] == 'train_epoch':
            return cur_key.encrypt(serialize(self.train_epoch(context, request_dict['key'])).encode('utf-8'))
        elif request_dict['action'] == 'get_model_params':
            return cur_key.encrypt(serialize(self.get_model_params(request_dict['key'])).encode('utf-8'))
        elif request_dict['action'] == 'set_model_params':
            return cur_key.encrypt(serialize(self.set_model_params(context, request_dict['key'], request_dict['params'])).encode('utf-8'))
        elif request_dict['action'] == 'predict':
            return cur_key.encrypt(serialize(self.predict(context, request_dict['key'], request_dict['inference_data']['x'])).encode('utf-8'))
        elif request_dict['action'] == 'predict_all':
            return cur_key.encrypt(serialize(self.predict_all(context, request_dict['key'], request_dict['inference_data']['x'])).encode('utf-8'))
        else:
            return cur_key.encrypt(serialize({'status': 'error', 'message': 'Invalid action'}).encode('utf-8'))

    def store_data(self, context: ts.Context, key: str, data: str, size: int) -> Dict[str, str]:
        if key in self.storage:
            return {'status': 'error', 'message': 'Key already exists'}

        bs = bytes.fromhex(data)
        self.storage[key] = bs
        vector = ts.ckks_vector_from(context, bs)
        self.running_sums[key] = vector.sum()
        self.squared_sums[key] = (vector * vector).sum()
        self.data_counts[key] = size

        return {'status': 'success', 'message': 'Data stored'}

    def compute_average(self, context: ts.Context, key: str) -> Dict[str, Any]:
        if key not in self.storage:
            return {'status': 'error', 'message': 'Key not found'}

        encrypted_sum = self.running_sums[key]
        data_count = self.data_counts[key]
        encrypted_average = encrypted_sum * (1 / data_count)

        return {'status': 'success', 'result': encrypted_average.serialize().hex()}

    def compute_variance(self, context: ts.Context, key: str) -> Dict[str, Any]:
        if key not in self.storage:
            return {'status': 'error', 'message': 'Key not found'}

        encrypted_sum = self.running_sums[key]
        encrypted_squared_sum = self.squared_sums[key]
        data_count = self.data_counts[key]

        encrypted_mean = encrypted_sum * (1 / data_count)
        encrypted_mean_of_squares = encrypted_squared_sum * (1 / data_count)
        encrypted_variance = encrypted_mean_of_squares - \
            (encrypted_mean * encrypted_mean)

        return {'status': 'success', 'result': encrypted_variance.serialize().hex()}

    def compute_standard_deviation(self, context: ts.Context, key: str) -> Dict[str, Any]:
        return {'status': 'error', 'message': 'Not implemented'}

    def compute_overall_average(self, context: ts.Context, keys: List[str]) -> Dict[str, Any]:
        overall_sum = ts.CKKSVector(context, [0])
        overall_count = 0

        for key in keys:
            if key not in self.storage:
                return {'status': 'error', 'message': f'Key {key} not found'}

            overall_sum += self.running_sums[key]
            overall_count += self.data_counts[key]

        encrypted_average = overall_sum * (1 / overall_count)

        return {'status': 'success', 'result': encrypted_average.serialize().hex()}

    def store_training_data(self, context: ts.Context, key: str, training_data: Dict[str, List[bytes]]) -> Dict[str, str]:
        if key in self.training_data:
            return {'status': 'error', 'message': 'Key already exists'}

        self.training_data[key] = {
            'x': [ts.ckks_vector_from(context, bytes.fromhex(data))
                  for data in training_data['x']],
            'y': [ts.ckks_vector_from(context, bytes.fromhex(data))
                  for data in training_data['y']]
        }

        return {'status': 'success', 'message': 'Training data stored'}

    def initialize_model(self, context: ts.Context, key: str, n_features: int) -> Dict[str, str]:
        if key not in self.training_data:
            return {'status': 'error', 'message': 'Training data not found'}

        self.models[key] = EncryptedLR(n_features)

        return {'status': 'success', 'message': 'Model initialized'}

    def train_epoch(self, context: ts.Context, key: str) -> Dict[str, str]:
        if key not in self.models:
            return {'status': 'error', 'message': 'Model not found'}

        model = self.models[key]
        x_train = self.training_data[key]['x']
        y_train = self.training_data[key]['y']

        for enc_x, enc_y in zip(x_train, y_train):
            enc_out = model.forward(enc_x)
            model.backward(enc_x, enc_out, enc_y)
        model.update_parameters()

        return {'status': 'success', 'message': 'Epoch completed'}

    def get_model_params(self, key: str) -> Dict[str, Any]:
        if key not in self.models:
            return {'status': 'error', 'message': 'Model not found'}

        model = self.models[key]
        return {'status': 'success', 'params': model.get_parameters()}

    def set_model_params(self, context: ts.Context, key: str, params: Dict[str, bytes]) -> Dict[str, str]:
        if key not in self.models:
            return {'status': 'error', 'message': 'Model not found'}

        model = self.models[key]
        params['context'] = context
        model.set_parameters(params)

        return {'status': 'success', 'message': 'Model parameters updated'}

    def predict(self, context: ts.Context, key: str, x: List[str]) -> Dict[str, Any]:
        if key not in self.models:
            return {'status': 'error', 'message': 'Model not found'}

        model = self.models[key]
        enc_x = ts.ckks_vector_from(context, bytes.fromhex(x[0]))
        prediction = model.forward(enc_x)

        return {'status': 'success', 'result': prediction.serialize().hex()}

    def predict_all(self, context: ts.Context, key: str, x: List[bytes]) -> Dict[str, Any]:
        if key not in self.models:
            return {'status': 'error', 'message': 'Model not found'}

        model = self.models[key]
        predictions = []
        for enc_x in x:
            enc_x_vector = ts.ckks_vector_from(context, bytes.fromhex(enc_x))
            prediction = model.forward(enc_x_vector)
            predictions.append(prediction.serialize().hex())

        return {'status': 'success', 'result': predictions}
