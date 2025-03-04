import torch.nn as nn
import torch
from typing import Literal, Tuple
from src.inference import InferenceBase
from laplace import Laplace
import time
#from laplace.utils import RunningNLLMetric
from torchmetrics import MeanSquaredError
from src.logging import WandBLogger
from src.utils import FullLLLaplaceWrapper, DiagLLLaplaceWrapper
from torch.utils.data import DataLoader
from torch.nn.utils import parameters_to_vector, vector_to_parameters
#from laplace.curvature import AsdlGGN
import math
from .laplace_for_painn import LLLaplaceForPaiNN


class LastLayerLaplace(InferenceBase):
    """
    Laplace approximation for the last layer in a neural network.
    """
    def __init__(
        self,
        model: nn.Module,
        likelihood: Literal['classification', 'regression'],
        hessian_structure: Literal['full', 'kron', 'lowrank', 'diag'] = 'full', #Curreently do not use this.
        link_approx: Literal["mc", 'probit', 'bridge'] = 'mc',
        pred_type: Literal["glm", 'nn', 'gp'] = 'nn',
    ):
        super().__init__(model, likelihood=likelihood)


        self.likelihood = likelihood
        self.link_approx = link_approx
        self.pred_type = pred_type
        self.hessian_structure = hessian_structure

        self.num_params = sum(p.numel() for p in self.model.head.parameters())
        self.register_buffer('mean', torch.zeros(self.num_params))
        self.register_buffer('covariance', torch.zeros(self.num_params, self.num_params))

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        logger: WandBLogger, # Never used, just to not have to change train setup.
        num_outputs: int = 1, # Only used regression likelihood, can be used for multi output regression tasks.
        num_posterior_samples: int = 100,
        prior_fit_method: str = "CV",
        grid_size: int=20
    ):
        # Eval mode has effect on models with batch norm layers (and dropout) but not on filter response norm.
        self.model.eval()

        if self.model.name == 'painn':
            self.llla_model = LLLaplaceForPaiNN(self.model)
        elif self.hessian_structure == "full":
            self.llla_model = FullLLLaplaceWrapper(
                self.model, 
                likelihood=self.likelihood,
                prior_precision = 1.0 if prior_fit_method == "CV" else 40
            #    backend = AsdlGGN,
            )
        elif self.hessian_structure == "diag":
            self.llla_model = DiagLLLaplaceWrapper(
                self.model, 
                likelihood=self.likelihood,
                prior_precision = 1.0 if prior_fit_method == "CV" else 40
            #    backend = AsdlGGN,
            )
        else:
            raise Exception("This hessian structure does not exist.")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.llla_model.fit(train_loader)

        #if self.likelihood == "classification":
        #    loss = RunningNLLMetric()
        #elif self.likelihood == "regression":
        #    loss = MeanSquaredError(num_outputs=num_outputs)

        if prior_fit_method == "CV":
            self.llla_model.optimize_prior_precision(
            method='CV',
            val_loader=val_loader,
            pred_type=self.pred_type,
            link_approx=self.link_approx,
            n_samples = num_posterior_samples,
            grid_size = grid_size,
            log_prior_prec_min = -4,
            log_prior_prec_max = 4,
            #progress_bar = True,
            #loss = loss,
            )
        else:
            self.llla_model.optimize_prior_precision(
                pred_type = self.pred_type,
                link_approx = self.link_approx,
                n_samples = num_posterior_samples,
                init_prior_prec  = 40,
                n_steps=1000)

        self.mean = self.llla_model.mean
        if self.hessian_structure == "full":
            self.covariance = self.llla_model.posterior_covariance
        else:
            self.covariance = self.llla_model.posterior_variance


    def predict(
        self,
        dataloader: DataLoader,
        num_posterior_samples: int = 100,
        stratified: bool = False,
        covariance_scale_factor: float = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
       Predicts with the Laplace LL posterior approximation.

        Args:
            dataloader: A PyTorch dataloader containing the data to predict.
            num_posterior_samples: Number of posterior samples to compute
                predictions with.
            stratified: Boolean indicating if sampling should be stratified in
                case of a multimodal posterior. This argument is unused for the
                last-layer laplace approximation.

        Returns:
            A tuple of tensors with predictions for the data and the
            corresponding targets. The predictions has shape
            (num_posterior_samples, num_datapoints, output_size) where 
            output_size is the size of a single output from the model. The 
            targets has shape (num_datapoints,).
        """

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if covariance_scale_factor != 1:
            if isinstance(self.llla_model, FullLLLaplaceWrapper):
                self.llla_model.posterior_scale # To invoke computation of posterior scale
                self.llla_model._posterior_scale = (
                    self.llla_model._posterior_scale*math.sqrt(covariance_scale_factor)
                )

        outputs = list()
        targets = list()
        hidden_layer_outputs = list()
        #The none MC functionality here returns probs not logits, which is funky with train script. Needs fixing.
        if self.pred_type == "nn":
            if covariance_scale_factor != 1 and isinstance(self.llla_model, DiagLLLaplaceWrapper):
                posterior_samples = self.llla_model.sample(n_samples=num_posterior_samples, covariance_scale_factor=covariance_scale_factor)
            else:
                posterior_samples = self.llla_model.sample(n_samples=num_posterior_samples)

            with torch.no_grad():
                for i, params_sample in enumerate(posterior_samples):
                    params_sample = params_sample.to(device)
                    vector_to_parameters(params_sample, self.model.head.parameters())

                    outputs_sample = list()
                    self.model.eval()
                    for k, (x, y) in enumerate(dataloader):
                        x, y = x.to(device), y.to(device)
                        if i == 0:
                            hidden_outs = self.model.backbone(x)
                            output = self.model.head(hidden_outs)
                            hidden_layer_outputs.append(hidden_outs)
                        else:
                            output = self.model.head(hidden_layer_outputs[k])
                        outputs_sample.append(output.detach().cpu()) # batch_size x output_size
                        if i == 0:
                            targets.append(y.detach().cpu()) # batch_size

                    outputs_sample = torch.concat(outputs_sample, dim=0) # num_datapoints x output_size
                    outputs.append(outputs_sample)
            outputs = torch.stack(outputs, dim=0) # num_posterior_samples x num_datapoints x output_size
            targets = torch.concat(targets, dim=0) # num_datapoints
            vector_to_parameters(self.mean, self.model.head.parameters())
        else:
            with torch.no_grad():

                outputs = list()
                self.model.eval()
                for k, (x, y) in enumerate(dataloader):
                    x, y = x.to(device), y.to(device)
                    preds = self.llla_model(x, link_approx=self.link_approx, pred_type=self.pred_type)
                    outputs.append(preds.detach().cpu()) # batch_size x output_size
                    targets.append(y.detach().cpu()) # batch_size


            outputs = torch.concat(outputs, dim=0) # num_datapoints x output_size
            outputs = outputs.unsqueeze(0) # batch_Size x num_datapoints x output_size
            targets = torch.concat(targets, dim=0) # num_datapoints

        return outputs, targets

    def state_dict(
        self,
    ):
        state_dict = {}
        state_dict['model'] = self.model.state_dict()
        state_dict['llla'] = self.llla_model.state_dict()
        state_dict['mean'] = self.mean
        state_dict['covariance'] = self.covariance
        state_dict['hessian_structure'] = self.hessian_structure
        return state_dict

    #TODO: Need to load Laplace and load model to Laplace.
    def load_state_dict(
        self,
        state_dict,
        base: bool = True,
    ):
        #OBS: This "hacky" solution is to handle loading pretrained MAP model vs. loading fully trained Laplace (First is for training second is for testing)
        if base:
            for key in list(state_dict.keys()):
                state_dict[key.lstrip("model.")] = state_dict.pop(key)
            self.model.load_state_dict(state_dict)
        else:
            if 'hessian_structure' not in state_dict.keys():
                state_dict['hessian_structure'] = "full"
            self.model.load_state_dict(state_dict['model'])
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model.to(device)

            if self.model.name == 'painn':
                self.llla_model = LLLaplaceForPaiNN(self.model)
            elif state_dict['hessian_structure'] == "full":
                self.llla_model = FullLLLaplaceWrapper(
                    self.model, 
                    likelihood=self.likelihood,
                #    backend = AsdlGGN,
                )
            elif state_dict['hessian_structure'] == "diag":
                self.llla_model = DiagLLLaplaceWrapper(
                    self.model, 
                    likelihood=self.likelihood,
                #    backend = AsdlGGN,
                )

            self.llla_model.load_state_dict(state_dict['llla'])
            
            if state_dict['hessian_structure'] == "diag":
                self.llla_model.last_layer = self.model.head
                self.llla_model.n_params = len(parameters_to_vector(self.llla_model.last_layer.parameters()))
                self.llla_model.n_layers = len(list(self.llla_model.last_layer.parameters()))
            self.mean = state_dict['mean']
            self.covariance = state_dict['covariance']
