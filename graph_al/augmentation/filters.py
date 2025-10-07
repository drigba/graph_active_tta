from abc import abstractmethod
from enum import StrEnum
from graph_al.augmentation.config import SoftFilterMode
import torch
from graph_al.model.prediction import Prediction
from graph_al.data.base import Data
from typing import List

class FilterMetric(StrEnum):
    KL_DIVERGENCE = 'kl_divergence'
    BRIER_SCORE = 'brier_score'
    ABSOLUTE_DIFFERENCE = 'absolute_difference'
    COSINE_SIMILARITY = 'cosine_similarity'

class BaseFilter:
    """
    Base class for filtering augmentations.
    """
    @abstractmethod
    def mask(
        self,
        original_output: Prediction,
        augmented_outputs: List[Prediction],
        graph: List[Data]
    ) -> torch.Tensor:
        """
        Mask the graph based on the filter criteria.

        Args:
            original_output: The original model output.
            augmented_outputs: The augmented model output.
            graph: The graph to be filtered.

        Returns:
            A boolean mask indicating which nodes/edges to keep.
        """
        ...
    
    def __call__(
        self,
        original_output: Prediction,
        augmented_output: Prediction,
        graph: List[Data]
    ) -> torch.Tensor:
        return self.mask(original_output, augmented_output, graph)

class NoFilter(BaseFilter):
    """
    A filter that does not apply any filtering.
    """
    def mask(
        self,
        original_output: Prediction,
        augmented_output: Prediction,
        graph: List[Data]
    ) -> torch.Tensor:
        """
        Returns a mask that keeps all nodes/edges in the graph.
        """
        return torch.ones(original_output.probabilities.shape[1], dtype=torch.bool).to(original_output.probabilities.device)

class HardFilter(BaseFilter):
    """
    A filter that applies a hard threshold to the augmented predictions.
    """

    def mask(
        self,
        original_output: Prediction,
        augmented_output: Prediction,
        graph: List[Data]
    ) -> torch.Tensor:
        """
        Returns a mask based on a hard threshold applied to the augmented predictions.
        """
        pred_o = original_output.get_predictions(propagated=True)
        pred_a = augmented_output.get_predictions(propagated=True)
        mask = pred_o == pred_a
        return mask

class SoftFilter(BaseFilter):
    """
    A filter that applies a soft weighting to the augmented predictions.
    """

    def __init__(self, mode: str = SoftFilterMode.PACO, sample: bool = False):
        self.mode = mode
        self.sample = sample
        

    def mask(
        self,
        original_output: Prediction,
        augmented_outputs: List[Prediction],
        graph: List[Data]
    ) -> torch.Tensor:
        """
        Returns a mask based on a hard threshold applied to the augmented predictions.
        """
        return [self.get_mask(original_output, augmented_output) for augmented_output in augmented_outputs]

    def get_mask(self, original_output, augmented_output):
        pred_o = original_output.get_predictions(propagated=True)
        confidence_o = original_output.get_probabilities(propagated=True)[0]
        match self.mode:
            case SoftFilterMode.PACO:
                pred_a = augmented_output.get_predictions(propagated=True)
                mask = confidence_o[torch.arange(confidence_o.shape[0]), pred_a]
            case SoftFilterMode.POCA:
                confidence_a = augmented_output.get_probabilities(propagated=True)[0]
                mask = confidence_a[torch.arange(confidence_a.shape[0]), pred_o]
            case  SoftFilterMode.PACA:
                confidence_a = augmented_output.get_probabilities(propagated=True)[0].max(-1)[0]
                mask = confidence_a
            case _:
                mask = None
        if self.sample:
            mask = torch.bernoulli(mask)
        return mask




def get_metric_fn(metric: FilterMetric) -> BaseFilter:
    match metric:
        case FilterMetric.KL_DIVERGENCE:
            return lambda x,y : torch.nn.functional.kl_div(x, y, reduction='none').mean(-1)
        case FilterMetric.BRIER_SCORE:
            return lambda x,y : torch.nn.functional.mse_loss(x, y,reduction='none').mean(-1)
        case FilterMetric.ABSOLUTE_DIFFERENCE:
            return lambda x,y : torch.abs(x - y)
        case FilterMetric.COSINE_SIMILARITY:
            return lambda x,y : 1 - torch.nn.functional.cosine_similarity(x, y, dim=-1)
        



class FirmFilter(SoftFilter):
    """
    A filter that applies a soft weighting to the augmented predictions.
    """

    def mask(
        self,
        original_output: Prediction,
        augmented_output: Prediction,
        graph: List[Data]
    ) -> torch.Tensor:
        """
        Returns a mask based on a hard threshold applied to the augmented predictions.
        """
        soft_filter = self.get_mask(original_output,augmented_output)
        
        pred_o = original_output.get_predictions(propagated=True)
        pred_a = augmented_output.get_predictions(propagated=True)
        mask = pred_o == pred_a
        mask = mask * soft_filter
        return mask

