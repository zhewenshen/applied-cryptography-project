import base64

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
    