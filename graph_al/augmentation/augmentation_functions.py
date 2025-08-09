from torch_geometric.utils import to_dense_adj, dropout_edge, mask_feature
import torch

class BaseAugmentationFunction:
    def __init__(self):
        pass

    def augment(self, data):
        raise NotImplementedError("This method should be overridden by subclasses")

    def __call__(self, data):
        return self.augment(data)
    
    def update_params(self, new_param):
        pass
    
class EdgeDropoutAugmentation(BaseAugmentationFunction):
    def __init__(self, dropout_rate=0.1):
        super().__init__()
        self.dropout_rate = dropout_rate

    def augment(self, graph):
        # Implement edge dropout logic here
        # For example, randomly drop edges based on the dropout rate
        edge_index = graph.edge_index
        edge_index, _ = dropout_edge(edge_index, p=self.dropout_rate)
        graph.edge_index = edge_index
        return graph

    def update_params(self, new_param):
        self.dropout_rate = new_param

class FeatureMaskingAugmentation(BaseAugmentationFunction):
    def __init__(self, mask_rate=0.1, mask_mode='col'):
        super().__init__()
        self.mask_rate = mask_rate
        self.mask_mode = mask_mode

    def augment(self, graph):
        # Implement feature masking logic here
        # For example, randomly mask node features based on the mask rate
        graph.x, _ = mask_feature(graph.x, p=self.mask_rate, mode=self.mask_mode)
        return graph
    
    def update_params(self, new_param):
        self.mask_rate = new_param

class FeatureNoisingAugmentation(BaseAugmentationFunction):
    def __init__(self, noise_level=0.1):
        super().__init__()
        self.noise_level = noise_level

    def augment(self, graph):
        # Implement feature noising logic here
        # For example, add Gaussian noise to node features
        noise = torch.randn_like(graph.x) * self.noise_level
        graph.x += noise
        return graph
    
    def update_params(self, new_param):
        self.noise_level = new_param
    
class CombinedAugmentation(BaseAugmentationFunction):
    def __init__(self, augmentations):
        super().__init__()
        self.augmentations = augmentations

    def augment(self, graph):
        for augmentation in self.augmentations:
            graph = augmentation(graph)
        return graph
    
    def update_params(self, new_params):
        for augmentation, new_param in zip(self.augmentations, new_params):
            augmentation.update_params(new_param)