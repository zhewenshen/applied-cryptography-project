from typing import Any, Dict, List
from utils import serialize, deserialize, \
    validate_and_extract_header_from_client_hello, validate_tag
from model import EncryptedLR
import pysodium as nacl
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

    def set_server_key_pair(self, server_pk, server_sk):
        self.server_pk = server_pk
        self.server_sk = server_sk

    def set_client_pk(self, client_pk):
        self.client_pk = client_pk

    def nacl_encrypt(self, message):
        return nacl.crypto_secretstream_xchacha20poly1305_push(state=self.state,
                                                               message=message,
                                                               ad=None,
                                                               tag=nacl.crypto_secretstream_xchacha20poly1305_TAG_REKEY)

    def nacl_decrypt(self, ciphertext):
        message, tag = \
            nacl.crypto_secretstream_xchacha20poly1305_pull(state=self.state,
                                                            ciphertext=ciphertext,
                                                            ad=None)
        validate_tag(tag)
        return message

    def hello(self, request):
        (recv_key, sent_key) = \
            nacl.crypto_kx_server_session_keys(server_pk=self.server_pk,
                                               server_sk=self.server_sk,
                                               client_pk=self.client_pk)
        print(f'received Client Hello: {request}')
        header = validate_and_extract_header_from_client_hello(request)
        self.state = \
            nacl.crypto_secretstream_xchacha20poly1305_init_pull(header=header,
                                                                 key=recv_key)
        response = serialize({'status': 'success', 'message': 'Client Hello ACKed'})
        return self.nacl_encrypt(response)

    def handle_request(self, encrypted_request: bytes) -> bytes:
        request = self.nacl_decrypt(encrypted_request)
        request_dict = deserialize(request) # FIXME: validate request
        context = ts.context_from(bytes.fromhex(request_dict['context']))

        if request_dict['action'] == 'store':
            return self.nacl_encrypt(serialize(self.store_data(context, request_dict['key'], request_dict['data'], request_dict['size'])))
        elif request_dict['action'] == 'compute_average':
            return self.nacl_encrypt(serialize(self.compute_average(context, request_dict['key'])))
        elif request_dict['action'] == 'compute_variance':
            return self.nacl_encrypt(serialize(self.compute_variance(context, request_dict['key'])))
        elif request_dict['action'] == 'sd':
            return self.nacl_encrypt(serialize(self.compute_standard_deviation(context, request_dict['key'])))
        elif request_dict['action'] == 'compute_overall_average':
            return self.nacl_encrypt(serialize(self.compute_overall_average(context, request_dict['keys'])))
        elif request_dict['action'] == 'store_training_data':
            return self.nacl_encrypt(serialize(self.store_training_data(context, request_dict['key'], request_dict['training_data'])))
        elif request_dict['action'] == 'initialize_model':
            return self.nacl_encrypt(serialize(self.initialize_model(context, request_dict['key'], request_dict['n_features'])))
        elif request_dict['action'] == 'train_epoch':
            return self.nacl_encrypt(serialize(self.train_epoch(context, request_dict['key'])))
        elif request_dict['action'] == 'get_model_params':
            return self.nacl_encrypt(serialize(self.get_model_params(request_dict['key'])))
        elif request_dict['action'] == 'set_model_params':
            return self.nacl_encrypt(serialize(self.set_model_params(context, request_dict['key'], request_dict['params'])))
        elif request_dict['action'] == 'predict':
            return self.nacl_encrypt(serialize(self.predict(context, request_dict['key'], request_dict['inference_data']['x'])))
        elif request_dict['action'] == 'predict_all':
            return self.nacl_encrypt(serialize(self.predict_all(context, request_dict['key'], request_dict['inference_data']['x'])))
        else:
            return self.nacl_encrypt(serialize({'status': 'error', 'message': 'Invalid action'}))

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
