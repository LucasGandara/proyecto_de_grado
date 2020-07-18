from abc import ABCMeta, abstractmethod
from typing import Optional, Tuple, List

import numpy as np


class Layer(ABCMeta):

    @property
    def weights(self):
        """
        Returns weights tensor if layer is trainable.
        Returns None for non-trainable layers.
        """
        return None

    @property
    def gradients(self):
        """
        Returns bias tensor if layer is trainable.
        Returns None for non-trainable layers.
        """
        return None

    @abstractmethod
    def forward_pass(self, a_prev, training):
        """
        Perform layer forward propagation logic.
        """
        pass

    @abstractmethod
    def backward_pass(self, da_curr):
        pass

    def set_wights(self, w, b):
        """
        Perform layer backward propagation logic.
        """
        pass


class Optimizer(ABCMeta):

    @abstractmethod
    def update(self, layers):
        """
        Updates value of weights and bias tensors in trainable layers.
        """
        pass

