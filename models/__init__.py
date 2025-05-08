# Import all model classes
from .mlp import MLPModel
from .lstm import LSTMModel
from .cnn1d import CNN1DModel
from .tcn import TCNModel
from .cnn_lstm import CNNLSTMModel
from .informer import InformerModel
from .tsmixer import TSMixerModel
from .transformer import TransformerModel
from .itransformer import iTransformerModel
from .mamba import MambaModel

# Register all models
__all__ = [
    'MLPModel',
    'LSTMModel',
    'CNN1DModel',
    'TCNModel',
    'TransformerModel',
    'CNNLSTMModel',
    'InformerModel',
    'TSMixerModel',
    'iTransformerModel',
    'MambaModel',
]
