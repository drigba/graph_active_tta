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
from graph_al.augmentation.filters import BaseFilter, HardFilter, NoFilter, get_metric_fn


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

class InitialPredictor(BasePredictor):
    
    def predict(self, dataset, acquisition_step):
        prediction = Prediction()
        acquired_idxs, acquisition_metrics = self.acquisition_strategy.acquire(
                model=self.model,
                dataset=dataset,
                prediction=prediction,
                num=1,
                model_config=self.model.config,
                generator=self.generator,
            )
        return prediction, acquired_idxs, acquisition_metrics





