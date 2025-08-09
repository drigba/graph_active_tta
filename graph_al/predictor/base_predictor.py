from abc import abstractmethod
from typing import Dict, Tuple

from graph_al.augmentation.base_augmentor import BaseAugmentor
from graph_al.model.config import ModelConfig
from graph_al.model.prediction import Prediction
import torch
from graph_al.model.base import BaseModel
from graph_al.acquisition.base import BaseAcquisitionStrategy
from graph_al.acquisition.attribute import AcquisitionStrategyByAttribute
from graph_al.data.base import Dataset, Data
from graph_al.utils.logging import get_logger
from graph_al.augmentation.filters import BaseFilter, HardFilter, NoFilter


class BasePredictor:
    """
    Base class for all predictors.
    """

    def __init__(
        self,
        model: BaseModel,
        device: torch.device,
        acquisition_strategy: BaseAcquisitionStrategy,
        generator: torch.Generator = None,
    ):
        """
        Initialize the predictor with a model and device.

        Args:
            model: The model to be used for predictions.
            device: The device on which the model will run (e.g., 'cpu' or 'cuda').
        """
        self.model = model
        self.device = device
        self.acquisition_strategy = acquisition_strategy
        self.generator = (
            generator if generator is not None else torch.Generator(device=device)
        )

    def update_model(self, model: BaseModel):
        """
        Update the model reference in the predictor.

        Args:
            model: The new model to be used for predictions.
        """
        self.model = model

    @abstractmethod
    def predict(self, dataset: Dataset, acquisition_step: int):
        """
        Predict using the model on the provided dataset.

        Args:
            dataset: Input data for prediction.

        Returns:
            Predictions made by the model.
        """
        ...


class NormalPredictor(BasePredictor):
    """
    A normal predictor that uses the model to make predictions.
    """

    def predict(self, dataset: Dataset, acquisition_step: int) -> Tuple[Prediction, torch.Tensor, Dict[str, torch.Tensor]]:
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
            data = data.to(self.device)
            prediction = self.model.predict(data, acquisition=True)
            acquired_idxs, acquisition_metrics = self.acquisition_strategy.acquire(
                model=self.model,
                dataset=dataset,
                prediction=prediction,
                num=1,
                model_config=self.model.config,
                generator=self.generator,
            )

        return prediction, acquired_idxs, acquisition_metrics


class EQSPredictor(BasePredictor):
    """
    An EQS predictor that uses the model to make predictions with an acquisition strategy.
    """

    def __init__(
        self,
        model: BaseModel,
        device: torch.device,
        acquisition_strategy: BaseAcquisitionStrategy,
        augmentor: BaseAugmentor,
        number_of_augmentations: int = 1,
        generator: torch.Generator = None,
    ):
        super().__init__(
            model=model,
            device=device,
            acquisition_strategy=acquisition_strategy,
            generator=generator,
        )
        self.augmentor = augmentor
        self.number_of_augmentations = number_of_augmentations

    def update_model(self, model: BaseModel):
        """
        Update the model reference in the predictor and its augmentor.

        Args:
            model: The new model to be used for predictions.
        """
        super().update_model(model)
        self.augmentor.model = model

    def predict(
        self, dataset: Dataset, acquisition_step: int
    ) -> Tuple[Prediction, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Predict using the model on the provided data.

        Args:
            data: Input data for prediction.

        Returns:
            Predictions made by the model.
        """
        data = dataset.data
        self.model.eval()
        
        
        with torch.no_grad():
            cpu = torch.device("cpu")
            prediction = self.model.predict(data, acquisition=True).to(cpu)
            augmented_predictions = []
            augmented_graphs = []
            for _ in range(self.number_of_augmentations):
                new_param = 0.1 + (0.4-0.1)*self.number_of_augmentations/200
                # print(f"Updating augmentor parameters to {new_param} for acquisition step {acquisition_step}")
                new_params = [new_param] * 2
                self.augmentor.augmentation_function.update_params(new_params)
                p_tmp, data_clone = self.augmentor(data)
                augmented_predictions.append(p_tmp.to(cpu))
                # augmented_graphs.append(data_clone)
            masks = self.augmentor.filter_function(
                original_output=prediction,
                augmented_outputs=augmented_predictions,
                graph=augmented_graphs,
            )
            
            prediction.add_predictions_multiple(
                augmented_predictions, masks
            )
                
            prediction_normalizer = 1 / (masks.sum(dim=0) + 1)
            prediction.multiply_prediction(prediction_normalizer)

            acquired_idxs, acquisition_metrics = self.acquisition_strategy.acquire(
                model=self.model,
                dataset=dataset,
                prediction=prediction,
                num=1,
                model_config=self.model.config,
                generator=self.generator,
            )

        return prediction, acquired_idxs, acquisition_metrics




class QESPredictor(EQSPredictor):
    """
    A QES predictor that uses the model to make predictions with an acquisition strategy.
    """

    def __init__(
        self,
        model: BaseModel,
        device: torch.device,
        acquisition_strategy: AcquisitionStrategyByAttribute,
        augmentor: BaseAugmentor,
        number_of_augmentations: int = 1,
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

            query_metric = self.acquisition_strategy.get_attribute(
                prediction, self.model, dataset, self.generator, self.model.config
            )
            augmented_predictions = []
            augmented_graphs = []
            augmented_attributes = []
            for _ in range(self.number_of_augmentations):
                p_tmp, data_clone = self.augmentor(data)
                augmented_predictions.append(p_tmp)
                augmented_graphs.append(data_clone)
                dataset.data = data_clone
                acquisition_attribute = self.acquisition_strategy.get_attribute(
                    prediction, self.model, dataset, self.generator, self.model.config
                )
                augmented_attributes.append(acquisition_attribute)
                dataset.data = data
                
        masks = self.augmentor.filter_function(
            original_output=prediction,
            augmented_outputs=augmented_predictions,
            graph=augmented_graphs,
        )
        masks = torch.vstack(masks)
        augmented_attributes = torch.vstack(augmented_attributes)
        query_metric = augmented_attributes * masks
        query_metric = query_metric / (masks.sum(dim=0)+1)
        
        mask_acquired_idxs = torch.zeros_like(dataset.data.mask_train_pool)
        idx_sampled, query_metric_info = (
            self.acquisition_strategy.compute_idx_from_metric(
                query_metric, mask_acquired_idxs, self.model, dataset, self.generator
            )
        )
        return prediction, idx_sampled, query_metric_info
