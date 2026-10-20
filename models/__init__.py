"""
Model definitions for Event-to-Vibration Prediction.
"""

from .cnn1d import CNN1D, CNN1D_MLP, Simple1DCNN
from .simple1dcnn_windowed import Simple1DCNNWindowed, Simple1DCNNAdaptive
from .cnn2d import CNN2D
from .cnn2d_snn import CNN2D_SNN, create_cnn2d_snn_model
from .cnn2d_snn_eventrate import CNN2D_SNN_EventRate, create_cnn2d_snn_eventrate_model
from .cnn3d import CNN3D, CNN3D_Light
from .cnn_lstm import CNN_LSTM, CNN_LSTM_Attention
from .gnn import GNNModel, build_roi_graph

__all__ = [
    "Simple1DCNN",
    "Simple1DCNNWindowed",
    "Simple1DCNNAdaptive",
    "CNN1D",
    "CNN1D_MLP",
    "CNN2D",
    "CNN2D_SNN",
    "create_cnn2d_snn_model",
    "CNN2D_SNN_EventRate",
    "create_cnn2d_snn_eventrate_model",
    "CNN3D",
    "CNN3D_Light",
    "CNN_LSTM",
    "CNN_LSTM_Attention",
    "GNNModel",
    "build_roi_graph",
]
