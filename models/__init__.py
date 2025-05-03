# Import all model classes
from .mlp import MLPModel
from .lstm import LSTMModel
from .cnn1d import CNN1DModel
from .tcn import TCNModel
from .transformer_batchnorm import TransformerBNModel
from .cnn_lstm import CNNLSTMModel
from .informer import InformerModel
from .informer_hf import HFInformerModel
from .tsmixer import TSMixerModel
from .transformer import TransformerModel
from .itransformer import iTransformerModel

# Register all models
__all__ = [
    'MLPModel',
    'LSTMModel',
    'CNN1DModel',
    'TCNModel',
    'TransformerBNModel',
    'TransformerModel',
    'CNNLSTMModel',
    'InformerModel',
    'HFInformerModel',
    'TSMixerModel',
    'iTransformerModel',
]
