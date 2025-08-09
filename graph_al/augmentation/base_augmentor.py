from abc import abstractmethod
from graph_al.augmentation.augmentation_functions import BaseAugmentationFunction
from graph_al.augmentation.filters import BaseFilter
from graph_al.model.config import ModelConfig
from graph_al.model.prediction import Prediction
import torch
from graph_al.model.base import BaseModel
from graph_al.acquisition.base import BaseAcquisitionStrategy
from graph_al.acquisition.attribute import AcquisitionStrategyByAttribute
from graph_al.data.base import Dataset, Data
from copy import deepcopy


class BaseAugmentor:
    """
    Base class for augmentors that modify the graph structure or features.
    """

    def __init__(self,  augmentation_function:BaseAugmentationFunction , filter_function:BaseFilter ,model: BaseModel = None):
        """
        Initialize the BaseAugmentor with a graph.

        Args:
            graph: The graph to be augmented.
        """
        self.model = model
        self.augmentation_function = augmentation_function
        self.filter_function = filter_function

    @abstractmethod
    def augment(self, graph:Data) -> Prediction:
        """
        Perform the augmentation on the graph.

        Args:
            graph: The graph to be augmented.

        Returns:
            The augmented graph.
        """
        ...

    def __call__(self, graph: Data) -> Prediction:
        return self.augment(graph)

class GraphSpaceAugmentor(BaseAugmentor):
    """
    An augmentor that modifies the graph structure or features.
    """

    def augment(self, graph: Data) -> Prediction:
        """
        Perform the augmentation on the graph.

        Args:
            graph: The graph to be augmented.

        Returns:
            The augmented graph.
        """
        # Example augmentation logic
        graph_tmp = deepcopy(graph)
        graph_tmp = graph_tmp.to(graph.x.device)
        graph_tmp = self.augmentation_function(graph_tmp)
        return self.model.predict(graph_tmp, acquisition=True), graph_tmp



class LatentSpaceAugmentor(BaseAugmentor):
    """
    An augmentor that modifies the latent space of the graph.
    """

    def augment(self, graph: Data) -> Prediction:
        """
        Perform the augmentation on the graph.

        Args:
            graph: The graph to be augmented.

        Returns:
            The augmented graph.
        """
        # Example augmentation logic
        latent_vector = self.model.encode(graph)
        latent_vector = self.augmentation_function(latent_vector)
        return self.model.decode(latent_vector)
