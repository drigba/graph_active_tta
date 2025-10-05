from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from typing import Any
from graph_al.augmentation.config import AugmentorConfig

class PredictorType:
    NORMAL = "normal"
    EQS = "eqs"
    QES = "qes"

@dataclass
class PredictorConfig:
    type_: str = MISSING

@dataclass
class NormalPredictorConfig(PredictorConfig):
    type_: str = PredictorType.NORMAL

@dataclass
class EQSPredictorConfig(PredictorConfig):
    type_: str = PredictorType.EQS
    number_of_augmentations: int = 1
    augmentor: AugmentorConfig = field(default_factory=AugmentorConfig)

@dataclass
class QESPredictorConfig(EQSPredictorConfig):
    type_: str = PredictorType.QES
    

cs = ConfigStore.instance()
cs.store(name="base", node=PredictorConfig, group="predictor")
cs.store(name="normal", node=NormalPredictorConfig, group="predictor")
cs.store(name="eqs", node=EQSPredictorConfig, group="predictor")
cs.store(name="qes", node=QESPredictorConfig, group="predictor")
