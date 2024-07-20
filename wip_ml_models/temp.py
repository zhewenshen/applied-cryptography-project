import tenseal as ts
from typing import Any, Dict, List
from utils import serialize, deserialize
from context_gen import TenSEALContext
import random
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm import tqdm
import sys
from rich.console import Console
from rich.progress import track


class Client:
    def __init__(self):
        self.context = TenSEALContext.create_context()
        self.model_training_context = TenSEALContext.create_machine_learning_context()
        self.model_inference_context = TenSEALContext.create_machine_learning_context()
        self.test_data = {}
        self.console = Console()

    def encrypt_data(self, data: List[float], context: ts.Context) -> bytes:
        encrypted_data = ts.ckks_vector(context, data)
        return encrypted_data.serialize()

    def decrypt_data(self, serialized_data: bytes, context: ts.Context) -> List[float]:
        encrypted_data = ts.ckks_vector_from(context, serialized_data)
        return encrypted_data.decrypt()

    def send_request(self, server: 'Server', request: Dict[str, Any]) -> Dict[str, Any]:
        if 'request_type' not in request:
            request['request_type'] = 'normal'

        if request['request_type'] == 'ml':
            context = self.model_training_context
        elif request['request_type'] == 'inference':
            context = self.model_inference_context
        else:
            context = self.context

        request['context'] = context.serialize()

        if 'data' in request:
            request['data'] = self.encrypt_data(request['data'], context)
        elif 'training_data' in request:
            request['training_data'] = {
                # 'x': [self.encrypt_data(x, context) for x in track(request['training_data']['x'], description="Encrypting X", console=self.console)],
                # 'y': [self.encrypt_data([y], context) for y in track(request['training_data']['y'], description="Encrypting y", console=self.console)]
                'x': [self.encrypt_data(x, context) for x in request['training_data']['x']],
                'y': [self.encrypt_data([y], context) for y in request['training_data']['y']]
            }
        elif 'inference_data' in request:
            request['inference_data'] = {
                # 'x': [self.encrypt_data(x, context) for x in track(request['inference_data']['x'], description="Encrypting inference data", console=self.console)]
                'x': [self.encrypt_data(x, context) for x in request['inference_data']['x']]
            }

        serialized_request = serialize(request)
        response = server.handle_request(serialized_request)
        response_dict = deserialize(response)

        if 'result' in response_dict:
            if isinstance(response_dict['result'], bytes):
                response_dict['result'] = self.decrypt_data(
                    response_dict['result'], context)
            elif isinstance(response_dict['result'], list):
                response_dict['result'] = [self.decrypt_data(
                    r, context) for r in response_dict['result']]

        return response_dict

    def bootstrap_model(self, model_params: Dict[str, bytes]) -> Dict[str, bytes]:
        bootstrapped_params = {}
        for key, value in model_params.items():
            decrypted_value = self.decrypt_data(
                value, self.model_training_context)
            bootstrapped_params[key] = self.encrypt_data(
                decrypted_value, self.model_training_context)
        return bootstrapped_params

    def calculate_mse(self, key: str, predictions: List[float]) -> float:
        return np.mean((self.test_data[key]['y'] - np.array(predictions)) ** 2)

    def train_model(self, server: 'Server', key: str, num_epochs: int):
        for epoch in track(range(num_epochs), description="Training epochs", console=self.console):
            print(f"Epoch {epoch + 1}/{num_epochs}")

            train_request = {
                'action': 'train_epoch',
                'request_type': 'ml',
                'key': key
            }
            response = self.send_request(server, train_request)
            print(f"Epoch {epoch + 1} training response: {response}")

            get_params_request = {
                'action': 'get_model_params',
                'request_type': 'ml',
                'key': key
            }
            response = self.send_request(server, get_params_request)

            if response['status'] == 'success':
                bootstrapped_params = self.bootstrap_model(response['params'])

                set_params_request = {
                    'action': 'set_model_params',
                    'request_type': 'ml',
                    'key': key,
                    'params': bootstrapped_params
                }
                response = self.send_request(server, set_params_request)
                print(f"Epoch {epoch + 1} bootstrapping response: {response}")

            predict_request = {
                'action': 'predict_all',
                'request_type': 'inference',
                'key': key,
                'inference_data': {'x': self.test_data[key]['x']}
            }
            response = self.send_request(server, predict_request)
            if response['status'] == 'success':
                predictions = [pred[0] for pred in response['result']]
                mse = self.calculate_mse(key, predictions)
                print(f"Epoch {epoch + 1} MSE: {mse}")


class EncryptedLR:
    def __init__(self, n_features):
        self.weight = [random.uniform(-0.1, 0.1) for _ in range(n_features)]
        self.bias = random.uniform(-0.1, 0.1)
        self._delta_w = None
        self._delta_b = None
        self._count = 0

    def forward(self, enc_x):
        return enc_x.dot(self.weight) + self.bias

    def backward(self, enc_x, enc_out, enc_y):
        out_minus_y = (enc_out - enc_y)
        self._delta_w = enc_x * \
            out_minus_y if self._delta_w is None else self._delta_w + enc_x * out_minus_y
        self._delta_b = out_minus_y if self._delta_b is None else self._delta_b + out_minus_y
        self._count += 1

    def update_parameters(self):
        if self._count == 0:
            raise RuntimeError("You should at least run one forward iteration")
        lr = 0.01
        self.weight -= lr * self._delta_w
        self.bias -= lr * self._delta_b
        self._delta_w = None
        self._delta_b = None
        self._count = 0

    def get_parameters(self):
        return {
            'weight': self.weight.serialize(),
            'bias': self.bias.serialize()
        }

    def set_parameters(self, params):
        self.weight = ts.ckks_vector_from(params['context'], params['weight'])
        self.bias = ts.ckks_vector_from(params['context'], params['bias'])


class Server:
    def __init__(self):
        self.storage: Dict[str, List[bytes]] = {}
        self.running_sums: Dict[str, ts.CKKSVector] = {}
        self.squared_sums: Dict[str, ts.CKKSVector] = {}
        self.data_counts: Dict[str, int] = {}
        self.training_data: Dict[str, Dict[str, List[ts.CKKSVector]]] = {}
        self.models: Dict[str, EncryptedLR] = {}

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
        elif request_dict['action'] == 'store_training_data':
            return serialize(self.store_training_data(context, request_dict['key'], request_dict['training_data']))
        elif request_dict['action'] == 'initialize_model':
            return serialize(self.initialize_model(context, request_dict['key'], request_dict['n_features']))
        elif request_dict['action'] == 'train_epoch':
            return serialize(self.train_epoch(context, request_dict['key']))
        elif request_dict['action'] == 'get_model_params':
            return serialize(self.get_model_params(request_dict['key']))
        elif request_dict['action'] == 'set_model_params':
            return serialize(self.set_model_params(context, request_dict['key'], request_dict['params']))
        elif request_dict['action'] == 'predict':
            return serialize(self.predict(context, request_dict['key'], request_dict['inference_data']['x']))
        elif request_dict['action'] == 'predict_all':
            return serialize(self.predict_all(context, request_dict['key'], request_dict['inference_data']['x']))
        else:
            return serialize({'status': 'error', 'message': 'Invalid action'})

    def store_data(self, context: ts.Context, key: str, data: bytes, size: int) -> Dict[str, str]:
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

        encrypted_sum = self.running_sums[key]
        data_count = self.data_counts[key]
        encrypted_average = encrypted_sum * (1 / data_count)

        return {'status': 'success', 'result': encrypted_average.serialize()}

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

        return {'status': 'success', 'result': encrypted_variance.serialize()}

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

        return {'status': 'success', 'result': encrypted_average.serialize()}

    def store_training_data(self, context: ts.Context, key: str, training_data: Dict[str, List[bytes]]) -> Dict[str, str]:
        if key in self.training_data:
            return {'status': 'error', 'message': 'Key already exists'}

        self.training_data[key] = {
            'x': [ts.ckks_vector_from(context, data) for data in training_data['x']],
            'y': [ts.ckks_vector_from(context, data) for data in training_data['y']]
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

    def predict(self, context: ts.Context, key: str, x: List[bytes]) -> Dict[str, Any]:
        if key not in self.models:
            return {'status': 'error', 'message': 'Model not found'}

        model = self.models[key]
        enc_x = ts.ckks_vector_from(context, x[0])
        prediction = model.forward(enc_x)

        return {'status': 'success', 'result': prediction.serialize()}

    def predict_all(self, context: ts.Context, key: str, x: List[bytes]) -> Dict[str, Any]:
        if key not in self.models:
            return {'status': 'error', 'message': 'Model not found'}

        model = self.models[key]
        predictions = []
        for enc_x in x:
            enc_x_vector = ts.ckks_vector_from(context, enc_x)
            prediction = model.forward(enc_x_vector)
            predictions.append(prediction.serialize())

        return {'status': 'success', 'result': predictions}


def test_statistical_computations(client, server):
    datasets = [
        [random.uniform(0, 100) for _ in range(347)],
        [random.uniform(0, 100) for _ in range(443)],
        [random.uniform(0, 100) for _ in range(42)]
    ]

    all_data = []

    for i, data in enumerate(datasets):
        print(f"\nDataset {i + 1}:")

        response = client.send_request(server, {
            'action': 'store',
            'key': f'dataset_{i}',
            'request_type': 'normal',
            'data': data,
            'size': len(data)
        })
        print(f"Store response: {response}")

        response = client.send_request(server, {
            'action': 'compute_average',
            'request_type': 'normal',
            'key': f'dataset_{i}'
        })
        print(f"Compute average response: {response}")
        print(f"Actual average: {np.mean(data)}")

        all_data.extend(data)

    keys = [f'dataset_{i}' for i in range(len(datasets))]
    response = client.send_request(server, {
        'action': 'compute_overall_average',
        'request_type': 'normal',
        'keys': keys
    })
    print(f"\nCompute overall average response: {response}")
    print(f"Actual overall average: {np.mean(all_data)}")

    response = client.send_request(server, {
        'action': 'compute_variance',
        'request_type': 'normal',
        'key': 'dataset_1'
    })
    print(f"\nCompute variance response: {response}")
    print(f"Actual variance: {np.var(datasets[1])}")

    response = client.send_request(server, {
        'action': 'sd',
        'request_type': 'normal',
        'key': 'dataset_2'
    })
    print(f"\nCompute standard deviation response: {response}")
    print(f"Actual standard deviation: {np.std(datasets[2])}")


def test_machine_learning(client, server):
    california = fetch_california_housing()
    X, y = california.data, california.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.01, test_size=0.001, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model_key = 'california_model'
    client.test_data[model_key] = {'x': X_test, 'y': y_test}

    store_request = {
        'action': 'store_training_data',
        'request_type': 'ml',
        'key': model_key,
        'training_data': {'x': X_train.tolist(), 'y': y_train.tolist()}
    }

    response = client.send_request(server, store_request)
    print("\nStore training data response:", response)

    init_request = {
        'action': 'initialize_model',
        'request_type': 'ml',
        'key': model_key,
        'n_features': X_train.shape[1]
    }

    response = client.send_request(server, init_request)
    print("Initialize model response:", response)

    num_epochs = 2
    client.train_model(server, model_key, num_epochs)

    sample_test = X_test[0].tolist()
    predict_request = {
        'action': 'predict',
        'request_type': 'inference',
        'key': model_key,
        'inference_data': {'x': [sample_test]}
    }

    response = client.send_request(server, predict_request)
    print("Prediction response:", response)

    if response['status'] == 'success':
        print("Decrypted prediction:", response['result'][0])


if __name__ == "__main__":
    client = Client()
    server = Server()

    print("Testing Statistical Computations:")
    test_statistical_computations(client, server)

    print("\nTesting Machine Learning:")
    test_machine_learning(client, server)
