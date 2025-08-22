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
        augmented_outputs: List[Prediction],
        graph: List[Data]
    ) -> torch.Tensor:
        return torch.vstack(self.mask(original_output, augmented_outputs, graph))
    
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
        pred_o = original_output.get_predictions(propagated=True)
        for augmented_output in augmented_outputs:
            pred_a = augmented_output.get_predictions(propagated=True)
            mask = pred_o == pred_a
            masks.append(mask)
        return masks

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

class FirmFilter(SoftFilter):
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
            # PACA = POCA -> weight with augmented pred confidence
            # PACA        -> weight with original confidence
            soft_filter = self.get_mask(original_output,augmented_output)
            
            pred_o = original_output.get_predictions(propagated=True)
            pred_a = augmented_output.get_predictions(propagated=True)
            mask = pred_o == pred_a
            mask = mask * soft_filter
            masks.append(mask)
        return masks


class MetricThresholdFilter(BaseFilter):
    """
    A filter that applies a metric divergence threshold to the augmented predictions.
    """
    def __init__(self, metric_name: str = FilterMetric.BRIER_SCORE, threshold: float = 0.2):
        """
        Args:
            metric: The metric to use for weighting the augmented predictions.
        """
        super().__init__()
        self.metric = get_metric_fn(metric_name)
        self.threshold = threshold

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
        metric_scores = []
        for augmented_output in augmented_outputs:
            probs_a = augmented_output.get_probabilities(propagated=True)
            metric_score = self.metric(probs_o, probs_a)
            metric_scores.append(metric_score)

        metric_scores = torch.stack(metric_scores, dim=0)
        metric_mean = metric_scores.mean(dim=0) 
        # cos_dist = get_metric_fn(FilterMetric.COSINE_SIMILARITY)
        brier_fn = get_metric_fn(FilterMetric.BRIER_SCORE)
        metric_dist = brier_fn(metric_mean, metric_scores)
        metric_dist_norm = (metric_dist - metric_dist.min(0)[0]) / (metric_dist.max(0)[0] - metric_dist.min(0)[0] + 1e-8)
        masks = (metric_dist_norm < self.threshold)
        return masks