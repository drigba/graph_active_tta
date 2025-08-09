from abc import abstractmethod
from enum import StrEnum
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
        augmented_outputs: List[Prediction],
        graph: List[Data]
    ) -> torch.Tensor:
        return self.mask(original_output, augmented_outputs, graph)
    
class NoFilter(BaseFilter):
    """
    A filter that does not apply any filtering.
    """
    def mask(
        self,
        original_output: Prediction,
        augmented_outputs: List[Prediction],
        graph: List[Data]
    ) -> torch.Tensor:
        """
        Returns a mask that keeps all nodes/edges in the graph.
        """
        return [torch.ones(original_output.probabilities.shape[1], dtype=torch.bool).to(original_output.probabilities.device) for _ in augmented_outputs]

class HardFilter(BaseFilter):
    """
    A filter that applies a hard threshold to the augmented predictions.
    """

    def mask(
        self,
        original_output: Prediction,
        augmented_outputs: List[Prediction],
        graph: List[Data]
    ) -> torch.Tensor:
        """
        Returns a mask based on a hard threshold applied to the augmented predictions.
        """
        masks = []
        for augmented_output in augmented_outputs:
            pred_o = original_output.get_predictions(propagated=True)
            pred_a = augmented_output.get_predictions(propagated=True)
            mask = pred_o == pred_a
            masks.append(mask)
        masks = torch.stack(masks, dim=0)
        return masks

class SoftFilter(BaseFilter):
    """
    A filter that applies a soft weighting to the augmented predictions.
    """

    def mask(
        self,
        original_output: Prediction,
        augmented_outputs: List[Prediction],
        graph: List[Data]
    ) -> torch.Tensor:
        """
        Returns a mask based on a hard threshold applied to the augmented predictions.
        """
        masks = []
        for augmented_output in augmented_outputs:
            # pred_o = original_output.get_predictions(propagated=True)
            # confidence_a = original_output.get_probabilities(propagated=True)[0].max(-1)[0]
            # mask = confidence_a
            # mask = confidence_a[torch.arange(confidence_a.shape[0]), pred_o]
            
            pred_a = original_output.get_predictions(propagated=True)
            confidence_o = original_output.get_probabilities(propagated=True)[0].max(-1)[0]
            mask = confidence_o[torch.arange(confidence_o.shape[0]), pred_a]
            masks.append(mask)
        masks = torch.stack(masks, dim=0)
        return masks




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
        



class MetricWeightFilter(BaseFilter):
    """
    A filter that applies a metric divergence threshold to the augmented predictions.
    """
    def __init__(self, metric_name: str = "hard"):
        """
        Args:
            metric: The metric to use for weighting the augmented predictions.
        """
        super().__init__()
        self.metric = get_metric_fn(metric_name)

    def mask(
        self,
        original_output: Prediction,
        augmented_outputs: List[Prediction],
        graph: List[Data]
    ) -> torch.Tensor:
        """
        Returns a weighting based on a metric applied to the augmented predictions.
        """
        masks = []
        probs_o = original_output.get_probabilities(propagated=True)
        for augmented_output in augmented_outputs:
            probs_a = augmented_output.get_probabilities(propagated=True)
            metric_score = self.metric(probs_o, probs_a)
            # Normalize the metric score to [0, 1]
            # Create a mask where the normalized score is below a threshold (e
            masks.append(metric_score)
        masks = torch.stack(masks, dim=0)
        masks = 1 -  (masks - masks.min(0)[0]) / (masks.max(0)[0] - masks.min(0)[0])


        return masks
    
class FirmFilter(MetricWeightFilter):
    """
    A filter that applies a soft weighting to the augmented predictions.
    """

    def mask(
        self,
        original_output: Prediction,
        augmented_outputs: List[Prediction],
        graph: List[Data]
    ) -> torch.Tensor:
        """
        Returns a mask based on a hard threshold applied to the augmented predictions.
        """
        masks = []
        probs_o = original_output.get_probabilities(propagated=True)

        for augmented_output in augmented_outputs:
            probs_a = augmented_output.get_probabilities(propagated=True)
            # confidence_a = augmented_output.get_probabilities(propagated=True)[0].max(-1)[0]
            # confidence_o = original_output.get_probabilities(propagated=True)[0].max(-1)[0]
            # adjusted_confidence_a = 1 + confidence_a - confidence_o
            metric_score = self.metric(probs_o, probs_a)
            pred_o = original_output.get_predictions(propagated=True)
            pred_a = augmented_output.get_predictions(propagated=True)
            mask = pred_o == pred_a
            mask = mask * metric_score
            masks.append(mask)
        masks = torch.stack(masks, dim=0)
        masks = masks * (1 -  (masks - masks.min(0)[0]) / (masks.max(0)[0] - masks.min(0)[0]))
        return masks


class MetricThresholdFilter(BaseFilter):
    """
    A filter that applies a metric-based threshold to the augmented predictions.
    """

    def __init__(self, threshold: float):
        """
        Args:
            threshold: The threshold value for filtering.
        """
        super().__init__()
        if not isinstance(threshold, (int, float)):
            raise ValueError("Threshold must be a numeric value.")
        if threshold < 0:
            raise ValueError("Threshold must be non-negative.")
        if threshold > 1:
            raise ValueError("Threshold must be in the range [0, 1].")
        self.threshold = threshold

    def mask(
        self,
        original_output: Prediction,
        augmented_outputs: List[Prediction],
        graph: List[Data]
    ) -> torch.Tensor:
        """
        Returns a mask based on a metric threshold applied to the augmented predictions.
        """
        pred_o = original_output.get_predictions(propagated=True)
        pred_a = augmented_outputs.get_predictions(propagated=True)
        
        # Example metric: absolute difference
        metric = torch.abs(pred_o - pred_a)
        
        mask = metric < self.threshold
        return mask