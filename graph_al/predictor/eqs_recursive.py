from abc import abstractmethod
from typing import Dict, Tuple

from graph_al.augmentation.base_augmentor import BaseAugmentor
from graph_al.model.config import ModelConfig
from graph_al.model.prediction import Prediction
from graph_al.predictor.base_predictor import BasePredictor
from graph_al.predictor.eqs import EQSPredictor
import torch
from graph_al.model.base import BaseModel
from graph_al.acquisition.base import BaseAcquisitionStrategy
from graph_al.acquisition.attribute import AcquisitionStrategyByAttribute
from graph_al.data.base import Dataset, Data
from graph_al.utils.logging import get_logger
from graph_al.augmentation.filters import BaseFilter, HardFilter, NoFilter, get_metric_fn


class EQSRecursivePredictor(BasePredictor):
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
        outer_loops: int = 1,
        metric_name: str = "brier_score",
        generator: torch.Generator = None,
    ):
        super().__init__(
            model=model,
            device=device,
            acquisition_strategy=acquisition_strategy,
            generator=generator,
        )
        print("Using EQS Recursive Predictor")
        self.augmentor = augmentor
        self.number_of_augmentations = number_of_augmentations
        if number_of_augmentations % outer_loops != 0:
            raise ValueError("number_of_augmentations must be a multiple of outer_loops")
        self.number_of_outer_loops = outer_loops
        self.metric = get_metric_fn(metric_name)

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
        from copy import deepcopy

        data_o = deepcopy(data)

        with torch.no_grad():
            cpu = torch.device("cpu")
            prediction = self.model.predict(data, acquisition=True).to(cpu)
            augmented_predictions = []
            augmented_graphs = []
            best_score, best_graph = float("inf"), None
            for j in range(self.number_of_outer_loops):
                for i in range(
                    self.number_of_augmentations // self.number_of_outer_loops
                ):
                    p_tmp, data_clone = self.augmentor(data)
                    augmented_predictions.append(p_tmp.to(cpu))
                    score = self.metric(
                        p_tmp.get_probabilities(propagated=True),
                        prediction.get_probabilities(propagated=True),
                    ).sum()
                    if score < best_score:
                        best_score = score
                        best_graph = data_clone
                data.x = best_graph.x
                best_score = float("inf")
                # augmented_graphs.append(data_clone)
            masks = self.augmentor.filter_function(
                original_output=prediction,
                augmented_outputs=augmented_predictions,
                graph=augmented_graphs,
            )

            prediction.add_predictions_multiple(augmented_predictions, masks)

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
        dataset.data = data_o.to(self.device)
        return prediction, acquired_idxs, acquisition_metrics
