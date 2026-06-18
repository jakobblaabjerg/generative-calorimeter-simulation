MODEL_REGISTRY = {}
ENCODER_REGISTRY = {}


def register_model(name):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def register_encoder(name):
    def decorator(cls):
        ENCODER_REGISTRY[name] = cls
        return cls
    return decorator