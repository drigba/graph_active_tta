from abc import abstractmethod
from graph_al.acquisition.config import AcquisitionStrategyConfig
from graph_al.model.base import BaseModel
from graph_al.data.base import Dataset
from graph_al.model.prediction import Prediction
from graph_al.model.config import ModelConfig
from graph_al.utils.logging import get_logger
from graph_al.data.config import DatasetSplit
from collections import defaultdict

import torch
import torch_scatter
from torch import Tensor, Generator
from jaxtyping import Int, Bool, jaxtyped
from typeguard import typechecked
from typing import Tuple, Dict, Any, List

from copy import deepcopy
from torch_geometric.utils import to_dense_adj, dropout_edge, mask_feature
import torch.nn.functional as F

from graph_al.model.base import Ensemble, BaseModel
from graph_al.model.trainer.config import TrainerConfig
from graph_al.utils.logging import get_logger
from graph_al.model.trainer.build import get_trainer
from graph_al.model.build import get_model
from graph_al.acquisition.enum import NodeAugmentation, EdgeAugmentation


class BaseAcquisitionStrategy:
    """Base class for acquisition strategies.""" 
    
    def __init__(self, config: AcquisitionStrategyConfig):
        self.name = config.name
        self.balanced = config.balanced
        self.requires_model_prediction = config.requires_model_prediction
        self.verbose = config.verbose
        self.scale = config.scale
        self.probs_list = []
        self.probs_o_list = []
        self.probs_unfiltered_list = []
        self.probs_filtered_list = []
        self.num_steps = config.num_steps
        if config.tta_enabled:
            print("TTA ENABLED")
            self.tta = True
            
            self.tta_strat_node = config.tta.strat_node
            self.tta_strat_edge = config.tta.strat_edge
            
            self.tta_norm = config.tta.norm
            self.num = config.tta.num
            if self.tta_strat_edge == EdgeAugmentation.ADAPTIVE:
                self.drop_weights = None
            if self.tta_strat_node == NodeAugmentation.ADAPTIVE:
                self.feature_weights = None
            self.tta_filter = config.tta.filter
            self.probs = config.tta.probs
            self.p_edge = config.tta.p_edge
            self.p_node = config.tta.p_node
            print("p_node: ", self.p_node)
            print("p_edge: ", self.p_edge)
        else:
            self.tta = False
        if config.adaptation:
            self.adaptation = config.adaptation

    @property
    def retrain_after_each_acquisition(self) -> bool | None:
        """ If the model should be retrained after each acquisition. """
        return None

    @abstractmethod
    def is_stateful(self) -> bool:
        """ If the acquisition strategy is stateful, i.e. has a state that persists over multiple acquisitions. """
        return False

    def reset(self):
        """ Resets the acquisition strategies state. """
        ...

    def update(self, idxs_acquired: List[int], prediction: Prediction | None, dataset: Dataset, model: BaseModel):
        """ Updates the acquisition strategy state after acquisition.

        Args:
            idxs_acquired (List[int]): acquired indices
            prediction (Prediction | None): The model predictions that were used in the acquisition
            dataset (Dataset): the dataset
            model (BaseModel): the model in its state before `idxs_acquires` were acquired.
        """
        ...
        
    @abstractmethod
    def acquire_one(self, mask_acquired: Bool[Tensor, 'num_nodes'], prediction: Prediction | None, model: BaseModel, dataset: Dataset, model_config: ModelConfig, 
            generator: Generator) -> Tuple[int, Dict[str, Tensor | None]]:
        """ Acquires one label

        Args:
            mask_acquired (Bool[Tensor, &#39;num_nodes&#39;]): which nodes have been acquired in this iteration already
            prediction (Prediction | None): an optional model prediction if the acquisition needs that
            model (BaseModel): the classifier 
            dataset (Dataset): the dataset
            generator (Generator): a rng

        Returns:
            int: the acquired node label
            Dict[str, Tensor | None]: meta information from this aggregation
        """
        ...
    
    def augment_data_node(self, data, generator):
        """
        Augment the data using the model and generator.

        Args:
            data (torch_geometric.data.Data): The input data.
            generator (torch.Generator): The random number generator.

        Returns:
            torch_geometric.data.Data: The augmented data.
        """
        x = data.x
        i = data.num_train - data.num_classes
        # p_node = self.p_node - (self.p_node-0.1)*i/self.num_steps
        p_node = self.p_node
        # p_node = torch.empty(1).uniform_(0.1, 0.6).item()
        match self.tta_strat_node:
            case "dropout":
                return torch.nn.functional.dropout(x, p=p_node, training=True)
            case "mask":
                return mask_feature(x, p=p_node, mode='col')[0]
            case "noise":
                noise_mask = torch.randn_like(x) * p_node
                return x + noise_mask
            case "adaptive":
                return self.drop_feature_weighted_2(x, self.feature_weights, p=p_node).to(device=x.device)
            case _:
                return x
    
    def augment_data_edge(self, data, generator):
        """
        Augment the data using the model and generator.

        Args:
            data (torch_geometric.data.Data): The input data.
            generator (torch.Generator): The random number generator.

        Returns:
            torch_geometric.data.Data: The augmented data.
        """
        edge_index = data.edge_index
        i = data.num_train - data.num_classes
        # p_edge = self.p_edge - (self.p_edge-0.1)*i/self.num_steps
        p_edge = self.p_edge
        # p_edge = torch.empty(1).uniform_(0.1, 0.6).item()
        match self.tta_strat_edge:
            case "mask":
                edge_index, edge_mask = dropout_edge(edge_index, p=p_edge)
            case "adaptive":
                edge_index = self.drop_edge_weighted(edge_index, self.drop_weights, p=p_edge, threshold=0.7).to(device=edge_index.device)
            case "train_connection":
                train_mask = data.mask_train[edge_index[0]] | data.mask_train[edge_index[1]]
                edge_probs = torch.full((edge_index.size(1),), p_edge, device=edge_index.device)
                edge_probs[train_mask] = 0.8
                edge_to_drop = torch.bernoulli(edge_probs).to(torch.bool)
                edge_index = edge_index[:, ~edge_to_drop]
            case _:
                edge_index = edge_index
        return edge_index
    
    def augment_data(self, data, generator):
        """
        Augment the data using the model and generator.

        Args:
            data (torch_geometric.data.Data): The input data.
            generator (torch.Generator): The random number generator.

        Returns:
            torch_geometric.data.Data: The augmented data.
        """
        
        if self.tta_strat_node == "adaptive" and self.feature_weights is None:
            self.feature_weights = torch.load('feature_weights.pt')
        if self.tta_strat_edge == "adaptive" and self.drop_weights is None:
            self.drop_weights = torch.load('other_data/drop_weights.pt')
        
        data_clone = deepcopy(data)
        data_clone.x = self.augment_data_node(data_clone, generator)
        data_clone.edge_index = self.augment_data_edge(data_clone, generator)
        return data_clone
        
    def cosine_similarity_matrix(self, p1, p2):
    # p1 and p2 shape: [num_nodes, num_classes]
        return F.cosine_similarity(p1, p2, dim=2)  # shape: [num_nodes]

    def drop_feature_weighted(self, x, w, p: float, threshold: float = 0.7):
        w = w / w.mean() * p
        w = w.where(w < threshold, torch.ones_like(w) * threshold)
        drop_prob = w

        drop_mask = torch.bernoulli(drop_prob).to(torch.bool).to(x.device)

        x = x.clone()
        x[:, drop_mask] = 0.

        return x
    
    def drop_edge_weighted(self,edge_index, edge_weights, p: float, threshold: float = 1.):
        edge_weights = edge_weights / edge_weights.mean() * p
        edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
        sel_mask = torch.bernoulli(1. - edge_weights).to(torch.bool).to(edge_index.device)

        return edge_index[:, sel_mask]

    def acquire(self, model: BaseModel, dataset: Dataset, prediction: Prediction, num: int, model_config: ModelConfig, generator: Generator) -> Tuple[Int[Tensor, 'num'], Dict[str, Any]]:
        """ Computes the nodes to acquire in this iteration. It iteratively calls `acquire_one`.
        
        Returns:
            Tuple[Int[Tensor, 'num']: the indices of acquired nodes
            Dict[str, int | float]: Metrics over the acquistion
        """
        acquired_idxs = []
        acquired_meta = defaultdict(list)
        mask_acquired_idxs = torch.zeros_like(dataset.data.mask_train_pool)

        for _ in range(num):
            idx, acquired_meta_iteration = self.acquire_one(mask_acquired_idxs, prediction, model, dataset, model_config, generator)
            for k, v in acquired_meta_iteration.items():
                acquired_meta[k].append(v)
            acquired_idxs.append(idx)
            mask_acquired_idxs[idx] = True

        if self.is_stateful:
            self.update(acquired_idxs, prediction, dataset, model)
        
        
        return torch.tensor(acquired_idxs), self._aggregate_acquired_meta(acquired_meta)
    
    def _aggregate_acquired_meta(self, acquired_meta: Dict[str, Any]):
        # Filter out Nones
        acquired_meta = {k : [vi for vi in v if vi is not None] for k, v in acquired_meta.items()}
        acquired_meta = {k : v for k, v in acquired_meta.items() if len(v) > 0}
        
        aggregated = {}
        for k, v in acquired_meta.items():
            if all(isinstance(vi, Tensor) for vi in v):
                if len(set([tuple(vi.size()) for vi in v])) == 1: # Homogeneous tensors, we can stack
                    aggregated[k] = torch.stack(v)
                else:
                    aggregated[k] = v
            else:
                raise ValueError(f'Unsupported acquisition meta attribute of type(s) {list(type(vi) for vi in v)}')
        return aggregated
    
    def base_sampling_mask(self, model: BaseModel, dataset: Dataset, generator: torch.Generator) -> Bool[Tensor, 'num_nodes']:
        """ A mask from which it is legal to sample from."""
        return torch.ones_like(dataset.data.mask_train_pool)
    
    def pool(self, mask_sampled: Bool[Tensor, 'num_nodes'], model: BaseModel, dataset: Dataset, generator: Generator) -> Bool[Tensor, 'num_nodes']:
        """ Provides the pool to sample from according to the acquisition strategy realization.
        
        Args:
            mask_sampled: Bool[Tensor, 'num_nodes']: Mask for all nodes that are already selected in this current acquisition iteration (in case more than one labels are acquired in one iteration)

        Returns:
            Bool[Tensor, 'num_nodes']: Mask for the pool from which to sample
        """
        mask = dataset.data.mask_train_pool & (~mask_sampled) & self.base_sampling_mask(model, dataset, generator)
        if self.balanced:
            y = dataset.data.y[dataset.data.mask_train | mask_sampled]
            counts = torch_scatter.scatter_add(torch.ones_like(y), y, dim_size=dataset.data.num_classes)
            for label in torch.argsort(counts).detach().cpu().tolist():
                mask_class = mask & (dataset.data.y == label)
                if mask_class.sum() > 0:
                    return mask_class
                else:
                    # get_logger().warn(f'Can not sample balanced from class {label}. Not enough instances!')
                    ...
            else:
                raise RuntimeError(f'Could not sample from any class!')
        else:
            return mask
    
    @property
    def mask_not_in_val(self) -> Bool[Tensor, 'num_nodes'] | None:
        """ An optional mask of indices that should never be in the validation set and thus
        always available in the training pool. Needed for acquisitions with a fixed pool set. """
        return None

def mask_not_in_val(*acquisition_strategies) -> Bool[Tensor, 'num_nodes'] | None:
    """
    Returns the optional mask of idxs that should not be in validation masks given multiple acquisition strategies.
    """
    masks = [strategy.mask_not_in_val for strategy in acquisition_strategies if strategy.mask_not_in_val is not None]
    if len(masks) == 0:
        return None
    else:
        mask = masks[0]
        for other_mask in masks[1:]:
            mask |= other_mask
    return mask
