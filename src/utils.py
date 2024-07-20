import base64
import json

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

def debug_dict(d):
    print({k: type(v) for k, v in d.items()})
