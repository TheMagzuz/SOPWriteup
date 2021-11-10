from __future__ import annotations
import numpy as np
import mlmath


class Layer:
    def __init__(
        self,
        nodes: int,
        previous: Layer = None,
        next: Layer = None,
        weights: np.ndarray = None,
    ):
        """Initializes a layer for neural networks
        Parameters:
            nodes (int): The amount nodes in the layer
            previous (Layer): The layer preceding this layer. None if this is the first layer
            next (Layer): The layer after this layer. None if this is the final layer
            weights (np.ndarray): A matrix of the weights going into this layer. The matrix has the key (to, from)

        """
        self.nodeCount = nodes
        self.previous = previous
        self.next = next
        self.outputValues = np.empty(0)
        self.inputValues = np.empty(0)

        if previous != None:
            if weights != None:
                self.weights = weights
            else:
                self.weights = np.empty((nodes, previous.nodeCount))

    def calculateValues(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate the values of the layer, and set them on the layer object

        Returns
        -------
        The calculated values

        """

        # If this is the input layer, the values will simply be the input values
        if self.previous == None:
            self.inputValues = data
            self.outputValues = data
            return self.outputValues

        # Otherwise, calculate the input values as weights · previous
        prevValues = self.previous.calculateValues(data)
        self.inputValues = np.dot(self.weights, prevValues)
        self.outputValues = mlmath.sigmoid(self.inputValues)

        return self.outputValues

    def __repr__(self):
        return str(self.__dict__)
