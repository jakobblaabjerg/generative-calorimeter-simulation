from . import cfm
from . import mdn
from . import encoders

from models.registry import MODEL_REGISTRY, ENCODER_REGISTRY


print(MODEL_REGISTRY)
print(ENCODER_REGISTRY)

def create_model(config):
    return MODEL_REGISTRY[config.name](config.model)

def create_encoder(name):
    return ENCODER_REGISTRY[name]


# not in use