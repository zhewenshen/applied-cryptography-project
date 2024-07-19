from Pyfhel import Pyfhel, PyCtxt
import numpy as np
import os
from cryptography.hazmat.primitives.asymmetric import dh
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


from Pyfhel import Pyfhel, PyCtxt
import numpy as np
import os
from cryptography.hazmat.primitives.asymmetric import dh
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


class SecureChannel:
    # For demo
    P = 0xFFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B139B22514A08798E3404DDEF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7EDEE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F83655D23DCA3AD961C62F356208552BB9ED529077096966D670C354E4ABC9804F1746C08CA18217C32905E462E36CE3BE39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF6955817183995497CEA956AE515D2261898FA051015728E5A8AACAA68FFFFFFFFFFFFFFFF
    G = 2

    def __init__(self):
        print("[SECURE CHANNEL] Initializing DH keys...")
        self.parameters = dh.DHParameterNumbers(self.P, self.G).parameters()
        self.private_key = self.parameters.generate_private_key()
        self.public_key = self.private_key.public_key()
        self.shared_key = None
        print("[SECURE CHANNEL] DH keys initialized.")

    def get_public_key(self):
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

    def compute_shared_key(self, other_public_key):
        print("[SECURE CHANNEL] Computing shared key...")
        peer_public_key = serialization.load_pem_public_key(other_public_key)
        self.shared_key = self.private_key.exchange(peer_public_key)
        derived_key = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b'handshake data',
        ).derive(self.shared_key)
        self.aes_key = derived_key
        print("[SECURE CHANNEL] Shared key computed.")

    def encrypt(self, message):
        iv = os.urandom(16)
        encryptor = Cipher(algorithms.AES(self.aes_key),
                           modes.CFB(iv)).encryptor()
        return iv + encryptor.update(message) + encryptor.finalize()

    def decrypt(self, ciphertext):
        iv = ciphertext[:16]
        decryptor = Cipher(algorithms.AES(self.aes_key),
                           modes.CFB(iv)).decryptor()
        return decryptor.update(ciphertext[16:]) + decryptor.finalize()


class Client:
    def __init__(self):
        print("[CLIENT] Initializing Pyfhel session and data...")
        self.HE = Pyfhel(context_params={
            'scheme': 'ckks', 'n': 2**13, 'scale': 2**30, 'qi_sizes': [30]*5
        })
        self.HE.keyGen()
        self.HE.relinKeyGen()
        self.HE.rotateKeyGen()
        self.secure_channel = SecureChannel()

    def prepare_data(self, data):
        print(f"[CLIENT] Preparing data: {data}")
        self.x = np.array(data)
        self.cx = self.HE.encrypt(self.x)

        s_context = self.HE.to_bytes_context()
        s_public_key = self.HE.to_bytes_public_key()
        s_relin_key = self.HE.to_bytes_relin_key()
        s_rotate_key = self.HE.to_bytes_rotate_key()
        s_cx = self.cx.to_bytes()

        return {
            'context': s_context,
            'pk': s_public_key,
            'rlk': s_relin_key,
            'rtk': s_rotate_key,
            'cx': s_cx,
        }

    def process_response(self, c_res):
        res = self.HE.decryptFrac(c_res)
        print(f"[CLIENT] Response received! Result is {np.round(res[3], 4)}")
        return res[3]


class Server:
    def __init__(self):
        self.HE = Pyfhel()
        self.w = np.array([0.5, -1.5, 4, 5])
        self.secure_channel = SecureChannel()

    def process_request(self, request):
        print("[SERVER] Processing request...")
        self.HE.from_bytes_context(request['context'])
        self.HE.from_bytes_public_key(request['pk'])
        self.HE.from_bytes_relin_key(request['rlk'])
        self.HE.from_bytes_rotate_key(request['rtk'])
        cx = PyCtxt(pyfhel=self.HE, bytestring=request['cx'])

        ptxt_w = self.HE.encode(self.w)
        c_mean = (cx * ptxt_w)
        c_mean /= 4
        c_mean += (c_mean >> 1)
        c_mean += (c_mean >> 2)

        print(f"[SERVER] Average computed! Responding: c_mean={c_mean}")
        return c_mean


client = Client()
server = Server()

print("[CLIENT] Initiating key exchange...")
client_public_key = client.secure_channel.get_public_key()
server_public_key = server.secure_channel.get_public_key()

client.secure_channel.compute_shared_key(server_public_key)
server.secure_channel.compute_shared_key(client_public_key)

print("[CLIENT] Key exchange completed.")

data = [1.5, 2, 3.3, 4]
request = client.prepare_data(data)

encrypted_request = client.secure_channel.encrypt(str(request).encode())

decrypted_request = server.secure_channel.decrypt(encrypted_request)
request = eval(decrypted_request.decode())

response = server.process_request(request)

encrypted_response = server.secure_channel.encrypt(response.to_bytes())

decrypted_response = client.secure_channel.decrypt(encrypted_response)
response = PyCtxt(pyfhel=client.HE, bytestring=decrypted_response)

result = client.process_response(response)

print(f"[CLIENT] Final result: {result}")
