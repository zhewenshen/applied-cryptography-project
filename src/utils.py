import base64
import json
from jsonschema import validate
import pysodium as nacl

def serialize(obj):
    return json.dumps(obj)
    #if isinstance(obj, bytes):
    #    return base64.b64encode(obj).decode()
    #elif isinstance(obj, dict):
    #    return {k: serialize(v) for k, v in obj.items()}
    #elif isinstance(obj, list):
    #    return [serialize(v) for v in obj]
    #else:
    #    return obj

# FIXME: replace 'serialize' with 'serialize2' in the future
def serialize2(d):
    return json.dumps(d).encode('utf-8')

def deserialize(s):
    return json.loads(s)
    #if isinstance(obj, str):
    #    try:
    #        return base64.b64decode(obj.encode())
    #    except:
    #        return obj
    #elif isinstance(obj, dict):
    #    return {k: deserialize(v) for k, v in obj.items()}
    #elif isinstance(obj, list):
    #    return [deserialize(v) for v in obj]
    #else:
    #    return obj

# FIXME: replace 'deserialize' with 'deserialize2' in the future
def deserialize2(bs):
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
    d = deserialize2(serialized)
    validate(d, schema)
    assert len(d['data']) == d['size'], 'SIZE does not match size of DATA'
    return bytes.fromhex(d['data'])

def validate_server_hello(serialized):
    schema = {'status': 'success', 'message': 'Client Hello ACKed'}
    d = deserialize2(serialized)
    validate(d, schema)

def validate_tag(tag):
    assert tag == nacl.crypto_secretstream_xchacha20poly1305_TAG_REKEY, \
        'must re-key after every message'
