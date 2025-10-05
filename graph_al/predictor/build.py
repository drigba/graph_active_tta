from graph_al.predictor.base_predictor import  BasePredictor, NormalPredictor

from graph_al.predictor.config import PredictorType
from graph_al.model.base import BaseModel
from graph_al.acquisition.base import BaseAcquisitionStrategy
from graph_al.augmentation.base_augmentor import BaseAugmentor
from graph_al.data.base import Dataset
from graph_al.predictor.config import PredictorConfig
from graph_al.predictor.eqs import EQSPredictor
from graph_al.predictor.qes import QESPredictor
from torch import Generator
from typing import Optional
from graph_al.augmentation.build import get_augmentor
import torch

def get_predictor(
    config: PredictorConfig,
    model: BaseModel,
    device: torch.device,
    acquisition_strategy: BaseAcquisitionStrategy,
    generator: Optional[Generator],
) -> BasePredictor:
    """
    Build and return the appropriate predictor instance based on config.
    """
    match config.type_:
        case PredictorType.NORMAL:
            return NormalPredictor(
                model=model,
                device=device,
                acquisition_strategy=acquisition_strategy,
                generator=generator,
            )
        case PredictorType.EQS:
            augmentor = get_augmentor(
                config.augmentor,
                model=model,
            ) 
            return EQSPredictor(
                model=model,
                device=device,
                acquisition_strategy=acquisition_strategy,
                augmentor=augmentor,
                number_of_augmentations=getattr(config, "number_of_augmentations", 1),
                generator=generator,
            )
        case PredictorType.QES:
            augmentor = get_augmentor(
                config.augmentor,
                model=model,
            ) 
            return QESPredictor(
                model=model,
                device=device,
                acquisition_strategy=acquisition_strategy,
                augmentor=augmentor,
                number_of_augmentations=getattr(config, "number_of_augmentations", 1),
                generator=generator,
            )
        case _:
            raise ValueError(f"Unsupported predictor type {config.type_}")