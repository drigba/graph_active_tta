from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from typing import List, Any

class AugmentationType:
    EDGE_DROPOUT = "edge_dropout"
    FEATURE_MASKING = "feature_masking"
    FEATURE_NOISING = "feature_noising"
    ADAPTIVE = "adaptive"
    COMBINED = "combined"
    NONE = "none"

class AugmentorType:
    GRAPH_SPACE = "graph_space"
    LATENT_SPACE = "latent_space"

@dataclass
class AugmentationConfig:
    type_: str = MISSING
    p: float = 0.1  # Probability for dropout/mask/noise
    adaptive_params: dict = field(default_factory=dict)

@dataclass
class EdgeDropoutAugmentationConfig(AugmentationConfig):
    type_: str = AugmentationType.EDGE_DROPOUT

@dataclass
class FeatureMaskingAugmentationConfig(AugmentationConfig):
    type_: str = AugmentationType.FEATURE_MASKING

@dataclass
class FeatureNoisingAugmentationConfig(AugmentationConfig):
    type_: str = AugmentationType.FEATURE_NOISING

@dataclass
class AdaptiveAugmentationConfig(AugmentationConfig):
    type_: str = AugmentationType.ADAPTIVE

@dataclass
class CombinedAugmentationConfig(AugmentationConfig):
    type_: str = AugmentationType.COMBINED
    augmentations: List[Any] = field(default_factory=list)  # List of augmentation configs
    

    
class FilterType:
    NO_FILTER = "no_filter"
    HARD_FILTER = "hard_filter"
    METRIC_FILTER = "metric_filter"
    SOFT_FILTER = "soft_filter"
    FIRM_FILTER = "firm_filter"

@dataclass
class FilterConfig:
    type_: str = MISSING

@dataclass
class NoFilterConfig(FilterConfig):
    type_: str = FilterType.NO_FILTER

@dataclass
class HardFilterConfig(FilterConfig):
    type_: str = FilterType.HARD_FILTER

@dataclass
class MetricFilterConfig(FilterConfig):
    type_: str = FilterType.METRIC_FILTER
    metric: str = MISSING

@dataclass
class SoftFilterConfig(FilterConfig):
    type_: str = FilterType.SOFT_FILTER

@dataclass
class FirmFilterConfig(MetricFilterConfig):
    type_: str = FilterType.FIRM_FILTER

@dataclass
class AugmentorConfig:
    type_: str = MISSING
    augmentations: List[AugmentationConfig] = field(default_factory=list)  # Changed from single augmentation to list
    filter: FilterConfig = field(default_factory=FilterConfig)


@dataclass
class GraphSpaceAugmentorConfig(AugmentorConfig):
    type_: str = AugmentorType.GRAPH_SPACE

@dataclass
class LatentSpaceAugmentorConfig(AugmentorConfig):
    type_: str = AugmentorType.LATENT_SPACE

cs = ConfigStore.instance()
cs.store(name="base_filter", node=FilterConfig, group="predictor/augmentor/filter")
cs.store(name="no_filter", node=NoFilterConfig, group="predictor/augmentor/filter")
cs.store(name="hard_filter", node=HardFilterConfig, group="predictor/augmentor/filter")
cs.store(name="metric_filter", node=MetricFilterConfig, group="predictor/augmentor/filter")
cs.store(name="soft_filter", node=SoftFilterConfig, group="predictor/augmentor/filter")
cs.store(name="firm_filter", node=FirmFilterConfig, group="predictor/augmentor/filter")

cs = ConfigStore.instance()
cs.store(name="base", node=AugmentationConfig, group="predictor/augmentor/augmentation")
cs.store(name="edge_dropout", node=EdgeDropoutAugmentationConfig, group="predictor/augmentor/augmentation")
cs.store(name="feature_masking", node=FeatureMaskingAugmentationConfig, group="predictor/augmentor/augmentation")
cs.store(name="feature_noising", node=FeatureNoisingAugmentationConfig, group="predictor/augmentor/augmentation")
# cs.store(name="adaptive", node=AdaptiveAugmentationConfig, group="augmentation")
cs.store(name="combined", node=CombinedAugmentationConfig, group="predictor/augmentor/augmentation")

cs.store(name="base", node=AugmentorConfig, group="predictor/augmentor")
cs.store(name="graph_space", node=GraphSpaceAugmentorConfig, group="predictor/augmentor")
cs.store(name="latent_space", node=LatentSpaceAugmentorConfig, group="predictor/augmentor")