from context_gen import TenSEALContext
from utils import serialize, deserialize
from typing import Any, Dict, List
import tenseal as ts
from server import Server

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
    