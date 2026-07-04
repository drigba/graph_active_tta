from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from typing import Any
from graph_al.augmentation.config import AugmentorConfig

class PredictorType:
    NORMAL = "normal"
    GATTA_P = "gatta_p"
    GATTA_S = "gatta_s"

@dataclass
class PredictorConfig:
    type_: str = MISSING

@dataclass
class NormalPredictorConfig(PredictorConfig):
    type_: str = PredictorType.NORMAL

@dataclass
class GattaPPredictorConfig(PredictorConfig):
    type_: str = PredictorType.GATTA_P
    number_of_augmentations: int = 1
    augmentor: AugmentorConfig = field(default_factory=AugmentorConfig)

@dataclass
class GattaSPredictorConfig(GattaPPredictorConfig):
    type_: str = PredictorType.GATTA_S
    

cs = ConfigStore.instance()
cs.store(name="base", node=PredictorConfig, group="predictor")
cs.store(name="normal", node=NormalPredictorConfig, group="predictor")
cs.store(name="gatta_p", node=GattaPPredictorConfig, group="predictor")
cs.store(name="gatta_s", node=GattaSPredictorConfig, group="predictor")
