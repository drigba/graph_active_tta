from graph_al.augmentation.base_augmentor import GraphSpaceAugmentor, LatentSpaceAugmentor
from graph_al.augmentation.augmentation_functions import EdgeDropoutAugmentation, FeatureMaskingAugmentation, FeatureNoisingAugmentation, CombinedAugmentation
from graph_al.augmentation.config import AugmentationType
from typing import Any, Optional
from graph_al.augmentation.base_augmentor import BaseAugmentor
from graph_al.augmentation.augmentation_functions import BaseAugmentationFunction
from graph_al.augmentation.base_augmentor import BaseAugmentor
from graph_al.augmentation.config import AugmentorConfig, AugmentationConfig, FilterConfig, AugmentorType,AugmentationType, FilterType
from graph_al.model.base import BaseModel
from graph_al.augmentation.filters import NoFilter, HardFilter, BaseFilter, MetricWeightFilter, SoftFilter, FirmFilter




def get_augmentor(
    config: AugmentorConfig,
    model: BaseModel
) -> BaseAugmentor:
    """
    Get the augmentor based on the configuration.

    Args:
        config: Configuration object containing augmentation settings.
        model: The model to be used for augmentation.
        augmentation_function: The specific augmentation function to be used.
        filter_function: The specific filter function to be used.

    Returns:
        An instance of the augmentation function.
    """
    
    augmentation_functions = [
            get_augmentation_function(aug_config) 
            for aug_config in config.augmentations
        ]
    augmentation_function = CombinedAugmentation(augmentation_functions)
    
    # Get filter function from nested config
    filter_function = get_filter(config.filter)
    
    match config.type_:
        case AugmentorType.GRAPH_SPACE:
            return GraphSpaceAugmentor(model=model, 
                                      augmentation_function=augmentation_function, 
                                      filter_function=filter_function)
        case AugmentorType.LATENT_SPACE:
            return LatentSpaceAugmentor(model=model, 
                                       augmentation_function=augmentation_function, 
                                       filter_function=filter_function)
        case _:
            raise ValueError(f'Unsupported augmentation type {config.type_}')

def get_augmentation_function(config: AugmentationConfig) -> BaseAugmentationFunction:
    """
    Get the augmentation function based on the configuration.

    Args:
        config: Configuration object containing augmentation settings.
    
    Returns:
        An instance of the augmentation function.
    """
    if config is None or config.type_ == AugmentationType.NONE:
        return None
    match config.type_:
        case AugmentationType.EDGE_DROPOUT:
            return EdgeDropoutAugmentation(dropout_rate=config.p)
        case AugmentationType.FEATURE_NOISING:
            return FeatureNoisingAugmentation(noise_level=config.p)
        case AugmentationType.FEATURE_MASKING:
            return FeatureMaskingAugmentation(mask_rate=config.p)
        case AugmentationType.COMBINED:
            augmentations = [get_augmentation_function(aug_cfg) for aug_cfg in config.augmentations]
            return CombinedAugmentation(augmentations)
        case _:
            raise ValueError(f"Unknown augmentation type: {config.type_}")
        


def get_filter(config: FilterConfig) -> BaseFilter:
    if config is None:
        return None
    match config.type_:
        case FilterType.NO_FILTER:
            return NoFilter()
        case FilterType.HARD_FILTER:
            return HardFilter()
        case FilterType.METRIC_FILTER:
            return MetricWeightFilter(metric_name=config.metric)
        case FilterType.SOFT_FILTER:
            return SoftFilter()
        case FilterType.FIRM_FILTER:
            return FirmFilter(metric_name=config.metric)
        case _:
            raise ValueError(f"Unknown filter type: {config.type_}")