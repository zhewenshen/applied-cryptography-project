import base64
import json
from jsonschema import validate
import pysodium as nacl

def serialize(d):
    return json.dumps(d).encode('utf-8')

def deserialize(bs):
    return json.loads(bs.decode('utf-8'))

def dbg_break():
    print("Debug break point...")
    exit(1)

def validate_and_extract_header_from_client_hello(serialized):
    schema = {
    'action': 'hello',
    'request_type': 'normal',
    'data': "^[0-9a-f]*$",
    'size': "number"
    }
    d = deserialize(serialized)
    validate(d, schema)
    assert len(d['data']) == d['size'], 'SIZE does not match size of DATA'
    return bytes.fromhex(d['data'])

def validate_server_hello(serialized):
    schema = {'status': 'success', 'message': 'Client Hello ACKed'}
    d = deserialize(serialized)
    validate(d, schema)

def validate_tag(tag):
    assert tag == nacl.crypto_secretstream_xchacha20poly1305_TAG_REKEY, \
        'must re-key after every message'
