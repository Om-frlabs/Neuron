"""NEURON-1: Dendritic Predictive Network — A Hybrid Sparse MoE Architecture (~20M parameters defaults)."""
from neuron1.config import Neuron1Config
from neuron1.model import Neuron1
from neuron1.loss import Neuron1Loss, Neuron1WithHooks
from neuron1.data import SimpleTokenizer, TextDataset

__version__ = "0.1.0"
__all__ = ["Neuron1", "Neuron1Config", "Neuron1Loss", "Neuron1WithHooks", "SimpleTokenizer", "TextDataset"]
