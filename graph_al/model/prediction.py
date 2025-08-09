from dataclasses import dataclass

import torch.nn.functional as F
from jaxtyping import Float, Int
from typing import List
from torch import Tensor
import torch
from typeguard import typechecked
from jaxtyping import jaxtyped, Shaped
from numpy import ndarray
from graph_al.model.enum import *

class PredictionAttributeNotFound(BaseException):
    ...

@dataclass
class Prediction:
    """ Class for the model predictions. """
    
    logits: Float[Tensor, 'num_samples num_nodes num_classes'] | None = None
    logits_unpropagated: Float[Tensor, 'num_samples num_nodes num_classes'] | None = None
    
    embeddings: Float[Tensor, 'num_samples num_nodes embedding_dim'] | None = None
    embeddings_unpropagated: Float[Tensor, 'num_samples num_nodes embedding_dim'] | None = None
    
    probabilities: Float[Tensor, 'num_samples num_nodes num_classes'] | None = None
    probabilities_unpropagated: Float[Tensor, 'num_samples num_nodes num_classes'] | None = None
    
    aleatoric_confidence: Float[Tensor, 'num_nodes num_classes'] | None = None
    total_confidence: Float[Tensor, 'num_nodes num_classes'] | None = None
    epistemic_confidence: Float[Tensor, 'num_nodes num_classes'] | None = None
    approximation_error: Float[Tensor, 'num_nodes num_classes'] | None = None
    confidence_residual: Float[Tensor, 'num_nodes num_classes'] | None = None
    
    # KL divergence to prior is only predicted by a Bayesian model
    kl_divergence: Float[Tensor, 'num_samples'] | None = None
    num_kl_terms: Int[Tensor, ''] | None = None
    
    # Evidence only for evidential models (GPN)
    alpha: Float[Tensor, 'num_samples num_nodes num_classes'] | None = None
    alpha_unpropagated: Float[Tensor, 'num_samples num_nodes num_classes'] | None = None
    log_beta: Float[Tensor, 'num_samples num_nodes num_classes'] | None = None
    log_beta_unpropagated: Float[Tensor, 'num_samples num_nodes num_classes'] | None = None
    
    # Adversarial training, SEAL
    discriminator_embeddings: Float[Tensor, 'num_samples num_nodes num_discriminator_features'] | None = None
    discriminator_logits: Float[Tensor, 'num_samples num_nodes num_classes'] | None = None
    
    # Approximation from ground truth bayesian classifier

    def __init__(self, **kwargs):
        """
        Initialize the Prediction object with given attributes.
        
        Args:
            **kwargs: Attributes to initialize the Prediction object.
        """
        for key, value in kwargs.items():
            if value is not None:
                setattr(self, key, value)
        if "probabilities" not in kwargs:
            self.probabilities = self.get_probabilities(propagated=True)  # Ensure probabilities are computed if not provided
        if "probabilities_unpropagated" not in kwargs:
            self.probabilities_unpropagated = self.get_probabilities(propagated=False)
    @property
    def num_classes(self) -> int:
        """ For how many classes predictions are made. """
        # It is cheaper to not "compute" `self.get_predictions()` right away but instead infer the number
        # of classes from the "raw" data tensors
        for attribute in ('logits', 'logits_unpropagated', 'probabilities', 'probabilities_unpropagated',
            'aleatoric_confidence', 'total_confidence', 'epistemic_confidence', 'alpha', 'alpha_unpropagated',
            'beta', 'beta_unpropagated',):
            value = getattr(self, attribute, None)
            if value is not None:
                return value.size(-1)
        prediction = self.get_predictions()
        if prediction is not None:
            return int((prediction.max() + 1).item())
        raise ValueError(f'Can not infer the number of predicted classes from the prediction')

    @jaxtyped(typechecker=typechecked)
    def get_adversarial_is_labeled_probabilities(self) -> Float[Tensor, 'num_samples num_nodes'] | None:
        """ Adverserial learning for SEAL: Get the probability that a node is labeled. """
        logits = self.discriminator_logits
        if logits is None:
            return None
        # This may be numerically instable...
        exp_logits = logits.exp().sum(-1)
        return exp_logits / (exp_logits + 1)
        

    @jaxtyped(typechecker=typechecked)
    def get_predictions(self, propagated: bool = True) -> Float[Tensor, 'num_samples num_nodes'] | None:

        probabilities = self.get_probabilities(propagated=propagated)
        if probabilities is not None:
            return probabilities.mean(0).max(dim=-1)[1]
        # If we have logits, we also have probabilities
        return None

    @jaxtyped(typechecker=typechecked)
    def get_probabilities(self, propagated: bool = True) -> Float[Tensor, 'num_samples num_nodes num_classes'] | None:
        probabilities = self.probabilities if propagated else self.probabilities_unpropagated
        if probabilities is not None:
            return probabilities
        logits = self.get_logits(propagated=propagated)
        if logits is not None:
            return F.softmax(logits, dim=-1)
        return None

    @jaxtyped(typechecker=typechecked)
    def get_logits(self, propagated: bool = True) -> Float[Tensor, 'num_samples num_nodes num_classes'] | None:
        return self.logits if propagated else self.logits_unpropagated

    @jaxtyped(typechecker=typechecked)
    def get_max_score(self, propagated: bool) -> Float[Tensor, 'num_nodes']:
        """ Gets the maximum score of a prediction: max_c E_theta[p(y)]_c

        Args:
            propagated (bool): Whether to return the propagated version

        Returns:
            Float[Tensor, 'num_nodes']: the max score
        """
        probabilities = self.get_probabilities(propagated=propagated)
        if probabilities is None:
            raise PredictionAttributeNotFound('No probabilities are predicted to compute mutual information')
        probabilities = probabilities.mean(0) # average over multiple samples
        labels = probabilities.max(dim=1)[1]
        return probabilities[torch.arange(probabilities.size(0)), labels]
    
    @jaxtyped(typechecker=typechecked)
    def get_expected_predictive_entropy(self, propagated: bool, eps: float = 1e-12) -> Float[Tensor, 'num_nodes']:
        """ Gets the expected entropy of a prediction: E_theta[ H[ p(y, theta) ] ]

        Args:
            propagated (bool): Whether to return the propagated version

        Returns:
            Float[Tensor, 'num_nodes']: the expected entropy
        """
        probabilities = self.get_probabilities(propagated=propagated)
        if probabilities is None:
            raise PredictionAttributeNotFound('No probabilities are predicted to compute mutual information')
        entropy = -(probabilities * torch.log(probabilities + eps)).sum(-1)
        return entropy.mean(0)
    
    @jaxtyped(typechecker=typechecked)
    def get_mutual_information(self, propagated: bool, eps: float=1e-12) -> Float[Tensor, 'num_nodes']:
        """ Gets the mutual information between model weights and the class labels: H[ E_theta[ p(y, theta) ] ] - E_theta[ H[p(y, theta)] ]

        Args:
            propagated (bool): Whether to return the propagated version

        Returns:
            Float[Tensor, 'num_nodes']: the mutual information
        """
        probabilities = self.get_probabilities(propagated=propagated)
        if probabilities is None:
            raise PredictionAttributeNotFound('No probabilities are predicted to compute mutual information')
        expected_entropy = self.get_expected_predictive_entropy(propagated, eps=eps)
        expected_propabilities = probabilities.mean(0)
        predictive_entropy = -(expected_propabilities * torch.log(expected_propabilities + eps)).sum(-1)
        return predictive_entropy - expected_entropy
    
    @jaxtyped(typechecker=typechecked)
    def get_predicted_variance(self, propagated: bool) -> Float[Tensor, 'num_nodes']:
        """ Gets the variance of the predicted class labels: var_theta[ p(y, theta)_c' ], where c' = argmax_c E[ p(y, theta) ]

        Args:
            propagated (bool): Whether to return the propagated version

        Returns:
            Float[Tensor, 'num_nodes']: the variance of the predicted class
        """
        probabilities = self.get_probabilities(propagated=propagated)
        if probabilities is None:
            raise PredictionAttributeNotFound('No probabilities are predicted to compute mutual information')
        labels = probabilities.mean(0).max(-1)[1]
        if probabilities.size(0) <= 1:
            var = torch.zeros_like(probabilities[0])
        else:
            var = probabilities.var(0)
        return var[torch.arange(var.size(0)), labels]
    
    @jaxtyped(typechecker=typechecked)
    def get_total_variance(self, propagated: bool, eps: float=1e-12) -> Float[Tensor, 'num_nodes']:
        """ Gets the total variance of the prediction: sum_c var_theta[ p(y, theta)_c ]

        Args:
            propagated (bool): Whether to return the propagated version

        Returns:
            Float[Tensor, 'num_nodes']: the variance of the prediction
        """
        probabilities = self.get_probabilities(propagated=propagated)
        if probabilities is None:
            raise PredictionAttributeNotFound('No probabilities are predicted to compute mutual information')
        if probabilities.size(0) <= 1:
            var = torch.zeros_like(probabilities[0])
        else:  
            var = probabilities.var(0)
        return var.sum(-1)
    
    @jaxtyped(typechecker=typechecked)
    def get_log_evidence(self, propagated: bool) -> Float[Tensor, 'num_nodes']:
        """ Gets the total evidence of samples collected by a Dirichlet-based method (e.g. GPN)
        Args:
            propagated (bool): Whether to return the propagated version

        Returns:
            Float[Tensor, 'num_nodes']: the evidence of the prediction
        """
        if propagated:
            log_evidence = self.log_beta
        else:
            log_evidence = self.log_beta_unpropagated
        if log_evidence is None:
            raise PredictionAttributeNotFound('No evidence is predicted')
        log_evidence = torch.logsumexp(log_evidence, dim=-1)
        assert log_evidence.size(0) == 1, f'There should not be more than one sample for evidential methods, but got {log_evidence.size(0)}'
        return log_evidence.mean(0) # ensemble average

    @jaxtyped(typechecker=typechecked)
    def get_energy(self, propagated: bool, temperature: float = 1.0) -> Float[Tensor, 'num_nodes']:
        """ Gets the logit energy
        Args:
            propagated (bool): Whether to return the propagated version
            temperature (float): Temperature parameter

        Returns:
            Float[Tensor, 'num_nodes']: the energy
        """
        logits = self.get_logits(propagated=propagated)
        if logits is None:
            raise PredictionAttributeNotFound(f'Prediction does not have logits, propagated={propagated}')
        energy = -temperature * torch.logsumexp(logits / temperature, dim=-1).mean(0)
        return energy

    @jaxtyped(typechecker=typechecked)
    def get_attribute(self, attribute: PredictionAttribute, propagated: bool, **kwargs) -> Shaped[Tensor, 'num_nodes']:
        """ Gets an attribute of this prediction.

        Args:
            attribute (PredictionAttribute): which attribute to get
            propagated (bool): if the attribute should be calculated from predictions made with propagation

        Returns:
            attribute (Shaped[Tensor, 'num_nodes']): the attribute
        """
        match attribute:
            case PredictionAttribute.MAX_SCORE:
                values = self.get_max_score(propagated)
            case PredictionAttribute.ENTROPY:
                values = self.get_expected_predictive_entropy(propagated)
            case PredictionAttribute.MUTUAL_INFORMATION:
                values = self.get_mutual_information(propagated)
            case PredictionAttribute.PREDICTED_VARIANCE:
                values = self.get_predicted_variance(propagated)
            case PredictionAttribute.TOTAL_VARIANCE:
                values = self.get_total_variance(propagated)
            case PredictionAttribute.LOG_EVIDENCE:
                values = self.get_log_evidence(propagated)
            case PredictionAttribute.ENERGY:
                values = self.get_energy(propagated, **kwargs)
            case attribute:
                raise ValueError(f'Unsupported prediction attribute {attribute}')
        return values

    
    @classmethod
    @typechecked
    def collate(cls, predictions: List['Prediction']) -> 'Prediction':
        """ Collates multiple predictions into one by concatenating tensors along the first axis ('num_samples')

        Args:
            predictions (List[&#39;Prediction&#39;]): predictions to concatenate

        Returns:
            Prediction: the collated prediction
        """
        @jaxtyped(typechecker=typechecked)        @typechecked
        def _collate_attribute(predictions: List['Prediction'], attribute: str) -> Shaped[Tensor, 'num_samples ...'] | None:
            attributes = [getattr(prediction, attribute) for prediction in predictions]
            if all(attribute is None for attribute in attributes):
                return None
            elif all(attribute is not None for attribute in attributes):
                return torch.cat(attributes, dim=0)
            else:
                raise RuntimeError(f'Cant collate {attribute} as it is inhomogenous for Nones among predictions.')
            
        return cls(**{attribute : _collate_attribute(predictions, attribute) for attribute in (
            'logits', 'logits_unpropagated', 'probabilities', 'probabilities_unpropagated', 'kl_divergence',
            'embeddings', 'embeddings_unpropagated', 'alpha', 'alpha_unpropagated', 'log_beta',
            'log_beta_unpropagated', 'aleatoric_confidence', 'aleatoric_confidence', 'epistemic_confidence',
            'discriminator_logits',
        )})

    def add_prediction(self, prediction: 'Prediction', weight: Float[Tensor, 'num_nodes'] = None) -> None:
        """ Adds a prediction to this prediction by summing the attributes.

        Args:
            prediction (Prediction): the prediction to add
            weight (Optional[Float[Tensor, 'num_nodes']]): the weight to apply to the prediction. If None, no weighting is applied.
        """
        for attribute in ('logits', 'logits_unpropagated', 'probabilities', 'probabilities_unpropagated'):
            value = getattr(prediction, attribute, None)
            if value is not None:
                current_value = getattr(self, attribute, None)
                if weight is not None:
                    value = value * weight.unsqueeze(-1)
                if current_value is None:
                    setattr(self, attribute, value)
                else:
                    setattr(self, attribute, current_value + value)
                    
    def add_predictions_multiple(self, predictions: List['Prediction'], weights: Float[Tensor, 'num_nodes'] | None = None) -> None:
        """ Adds multiple predictions to this prediction by summing the attributes.

        Args:
            predictions (List[Prediction]): the predictions to add
            weights (Optional[Float[Tensor, 'num_nodes']]): the weights to apply to the predictions. If None, no weighting is applied.
        """
        if len(weights) != len(predictions) and weights is not None:
            raise ValueError(f'Weights must have the same length as predictions, but got {len(weights)} and {len(predictions)}')
        for prediction, weight in zip(predictions, weights):
            self.add_prediction(prediction, weight=weight)

    def mask_prediction(self, mask: Float[Tensor, 'num_nodes']) -> 'Prediction':
        """ Masks the prediction by multiplying the attributes with the mask.

        Args:
            mask (Float[Tensor, 'num_nodes']): the mask to apply

        Returns:
            Prediction: the masked prediction
        """
        for attribute in ('logits', 'logits_unpropagated', 'probabilities', 'probabilities_unpropagated'):
            value = getattr(self, attribute, None)
            if value is not None:
                setattr(self, attribute, value * mask.unsqueeze(-1))
        return self
    
    def divide_prediction(self, divisor: Float[Tensor, 'num_nodes']) -> 'Prediction':
        """ Divides the prediction by the divisor.

        Args:
            divisor (Float[Tensor, 'num_nodes']): the divisor to apply

        Returns:
            Prediction: the divided prediction
        """
        for attribute in ('logits', 'logits_unpropagated', 'probabilities', 'probabilities_unpropagated'):
            value = getattr(self, attribute, None)
            if value is not None:
                setattr(self, attribute, value / divisor.unsqueeze(-1))
        return self

    def multiply_prediction(self, multiplier: Float[Tensor, 'num_nodes']) -> 'Prediction':
        """ Multiplies the prediction by the multiplier .

        Args:
            divisor (Float[Tensor, 'num_nodes']): the divisor to apply

        Returns:
            Prediction: the Multiplyd prediction
        """
        for attribute in ('logits', 'logits_unpropagated', 'probabilities', 'probabilities_unpropagated'):
            value = getattr(self, attribute, None)
            if value is not None:
                setattr(self, attribute, value * multiplier.unsqueeze(-1))
        return self
    
    
    def to(self, device: torch.device) -> 'Prediction':
        """ Moves the prediction to the specified device.

        Args:
            device (torch.device): the device to move the prediction to

        Returns:
            Prediction: the moved prediction
        """
        for attribute in ('logits', 'logits_unpropagated', 'probabilities', 'probabilities_unpropagated',
                          'embeddings', 'embeddings_unpropagated', 'aleatoric_confidence', 'total_confidence',
                          'epistemic_confidence', 'alpha', 'alpha_unpropagated', 'log_beta',
                          'log_beta_unpropagated', 'discriminator_logits'):
            value = getattr(self, attribute, None)
            if value is not None:
                setattr(self, attribute, value.to(device))
        return self
        
        
        
    