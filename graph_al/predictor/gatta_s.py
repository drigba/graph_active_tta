from abc import abstractmethod
from typing import Dict, Tuple

from graph_al.augmentation.base_augmentor import BaseAugmentor
from graph_al.model.config import ModelConfig
from graph_al.model.prediction import Prediction
from graph_al.predictor.gatta_p import GattaPPredictor
import torch
from graph_al.model.base import BaseModel
from graph_al.acquisition.base import BaseAcquisitionStrategy
from graph_al.acquisition.attribute import AcquisitionStrategyByAttribute
from graph_al.data.base import Dataset, Data
from graph_al.utils.logging import get_logger
from graph_al.augmentation.filters import BaseFilter, HardFilter, NoFilter, get_metric_fn


class GattaSPredictor(GattaPPredictor):
    """
    A GATTA-S predictor that uses the model to make predictions with an acquisition strategy.
    """

    def __init__(
        self,
        model: BaseModel,
        device: torch.device,
        acquisition_strategy: AcquisitionStrategyByAttribute,
        augmentor: BaseAugmentor,
        number_of_augmentations: int = 1,
        num_to_acquire_per_step: int = 1,
        generator: torch.Generator = None,
    ):
        if not isinstance(acquisition_strategy, AcquisitionStrategyByAttribute):
            raise TypeError(
                "acquisition_strategy must be an instance of AcquisitionStrategyByAttribute"
            )
        super().__init__(
            model,
            device,
            acquisition_strategy,
            augmentor,
            number_of_augmentations,
            num_to_acquire_per_step,
            generator,
        )

    def predict(
        self, dataset: Dataset, acquisition_step: int
    ) -> Tuple[Prediction, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Predict using the model on the provided dataset.

        Args:
            dataset: Input dataset for prediction.

        Returns:
            Predictions made by the model.
        """
        data = dataset.data
        self.model.eval()

        with torch.no_grad():
            prediction = self.model.predict(data, acquisition=True)

            orig_score = self.acquisition_strategy.get_attribute(
                prediction, self.model, dataset, self.generator, self.model.config
            )
            augmented_graphs = []
            masks = []
            augmented_attributes = []
            for _ in range(self.number_of_augmentations):
                p_tmp, data_clone = self.augmentor(data)
                dataset.data = data_clone
                acquisition_attribute = self.acquisition_strategy.get_attribute(
                    p_tmp, self.model, dataset, self.generator, self.model.config
                )
                dataset.data = data

                mask = self.augmentor.filter_function(
                    original_output=prediction,
                    augmented_output=p_tmp,
                    graph=augmented_graphs,
                )
                masks.append(mask)
                augmented_attributes.append(acquisition_attribute)
        
        masks = torch.vstack(masks).float()
        augmented_attributes = torch.vstack(augmented_attributes)
        query_metric = augmented_attributes * masks
        query_metric = query_metric / (masks.sum(dim=0) + 1)
        query_metric = query_metric.sum(dim=0)
        query_metric += orig_score / (masks.sum(dim=0) + 1)

        acquired_idxs = []
        acquisition_attributes = []
        mask_acquired_idxs = torch.zeros_like(dataset.data.mask_train_pool)
        for _ in range(self.num_to_acquire(dataset)):
            idx_sampled, query_metric_info = (
                self.acquisition_strategy.compute_idx_from_metric(
                    query_metric, mask_acquired_idxs, self.model, dataset, self.generator
                )
            )
            acquired_idxs.append(idx_sampled)
            acquisition_attributes.append(query_metric_info["acquisition_attribute"])
            mask_acquired_idxs[idx_sampled] = True
        return prediction, torch.tensor(acquired_idxs), {
            "acquisition_attribute": torch.stack(acquisition_attributes)
        }
