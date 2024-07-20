import tenseal as ts
from context_gen import TenSEALContext
from rich.console import Console
from rich.progress import track
from server import Server
from utils import serialize, deserialize
from typing import List, Dict, Any
import numpy as np


class Client:
    def __init__(self):
        self.context = TenSEALContext.create_context()
        self.model_training_context = TenSEALContext.create_machine_learning_context()
        self.model_inference_context = TenSEALContext.create_machine_learning_context()
        self.test_data = {}
        self.console = Console()

    def encrypt_data(self, data: List[float], context: ts.Context) -> str:
        encrypted_data = ts.ckks_vector(context, data)
        return encrypted_data.serialize().hex()

    def decrypt_data(self, serialized_data: str, context: ts.Context) -> List[float]:
        encrypted_data = ts.ckks_vector_from(context, bytes.fromhex(serialized_data))
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

        request['context'] = context.serialize().hex()

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
            print(f"====== Epoch {epoch + 1}/{num_epochs} LOG ====== ")

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
