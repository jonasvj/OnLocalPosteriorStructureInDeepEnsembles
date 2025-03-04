import math
import pyro
import torch
import torch.nn as nn
from tqdm import trange
from typing import Literal
from torch.optim import Optimizer
from src.logging import WandBLogger
from src.inference import InferenceBase, LastLayerLaplace
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from omegaconf import OmegaConf
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.nn import PyroModule, PyroSample
import pyro.distributions as dist
from pyro.distributions.transforms import Radial
from src.posterior_refinement.pyro import *
from src.posterior_refinement.autoguide import *
from src.utils.utils import *
import copy
from torch_geometric.data import Data
from src.data.qm9_utils import DataLoader as QM9Loader

class PosteriorRefinedLastLayerLaplace(InferenceBase):
    """
    Last Layer Laplace approximation refined with normalizing flows.
    Based on code from https://github.com/runame/laplace-refinement/
    """
    def __init__(
        self,
        model: LastLayerLaplace,
        likelihood: Literal['classification', 'regression'],
        data_augmentation: bool = False,
        transform: str = "radial",
        num_transforms: int = 10,
        n_classes: int = 10,
        n_features: int = 256,
        dataset: str = "cifar10",
        prior_precision: int = 0,
    ):
        super().__init__(model, likelihood=likelihood)
        #self.model -> Laplace Module
        #self.laplace -> Laplace Module
        #self.laplace and self.model -> same
        #self.laplace.model.eval()
        self.laplace = model
        self.data_augmentation = data_augmentation
        self.transform = transform
        self.num_transforms = num_transforms
        self.n_classes = n_classes
        self.n_features = n_features
        self.dataset = dataset
        self.prior_precision = prior_precision
        self.num_posterior_samples = 100


    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        logger: WandBLogger,
        num_epochs: int = 100,
        num_posterior_samples: int = 100
    ):
        pyro.clear_param_store()
        

        #Use -1 to toggle CV on prior precision.
        if self.prior_precision == -1:
            #if self.dataset == 'qm9':
            #    self.prior_precisions = torch.logspace(-4, 4, 21)
            #else:
            self.prior_precisions = [1, 5, 10, 20, 30, 40, 50, 70, 90, 100, 125, 150, 175, 200, 500]
            for i, prior_precision in enumerate(self.prior_precisions):
                self.prior_precision = prior_precision
                pyro.clear_param_store()
                self.get_fit(train_loader, val_loader, optimizer, num_epochs)
                outputs, targets = self.predict(val_loader, num_posterior_samples)
                eval_stats = self.compute_stats(outputs, targets)[0]
                #loss = eval_stats['loss']
                loss = -eval_stats['lpd']
                print('Prior precision:', self.prior_precision)
                print('Loss:', loss)
                if i == 0:
                    best_loss = loss
                if loss <= best_loss:
                    best_loss = loss
                    best_guide = copy.deepcopy(self.guide)
                    best_pyro_model = copy.deepcopy(self.pyro_model)
                    best_precision = self.prior_precision
            self.guide = best_guide
            self.pyro_model = best_pyro_model
            self.prior_precision = best_precision
            print('Best loss:', best_loss)
            print('Best prior precision:', self.prior_precision)

        else:
            self.get_fit(train_loader, val_loader, optimizer, num_epochs, logger)
            
    
    def get_fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        num_epochs: int = 100,
        logger: WandBLogger = None,
    ):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.n_data = len(train_loader.dataset) if not self.data_augmentation else len(train_loader.dataset.indices)
        self.laplace.model.eval()

        if self.dataset == 'qm9':
            self.pyro_model = PaiNNModelLL(
                n_data = self.n_data,
                n_features = self.n_features,
                feature_extractor=self.laplace.model.backbone,
                scale=self.laplace.llla_model.scale,
                shift=self.laplace.llla_model.shift,
                atom_references=self.laplace.llla_model.atom_references,
                prior_prec=self.laplace.llla_model.prior_precision if self.prior_precision == 0 else self.prior_precision,
                cuda = True if device == 'cuda' else False
            )
        else:
            self.pyro_model = ClassificationModelLL(
                n_data = self.n_data,
                n_classes = self.n_classes,
                n_features = self.n_features,
                feature_extractor = self.laplace.model.backbone,
                prior_prec = self.laplace.llla_model.prior_precision if self.prior_precision == 0 else self.prior_precision,
                cuda = True if device == 'cuda' else False
            )
        if len(self.laplace.covariance.shape) > 1:
            base_dist = dist.MultivariateNormal(self.laplace.mean, self.laplace.covariance)
            diag=False
        else:
            base_dist = dist.Normal(self.laplace.mean, torch.sqrt(self.laplace.covariance))
            diag=True

        self.guide = AutoNormalizingFlowCustom(
            self.pyro_model.model,
            self.laplace.mean.detach().to(device),
            self.laplace.covariance.detach().to(device),
            diag=diag,
            flow_type=self.transform,
            flow_len=self.num_transforms,
            cuda = True if device == 'cuda' else False
        )

        # OBS: This is hard coded and if we make other NF modules, we should make getter for this.
        optimizer = pyro.optim.CosineAnnealingLR({
            'optimizer': torch.optim.Adam, 
            'optim_args': OmegaConf.to_container(optimizer.optim_args), 
            'T_max': num_epochs*len(train_loader),
        })

        elbo = Trace_ELBO()

        svi = SVI(
            self.pyro_model.model,
            self.guide,
            optimizer,
            elbo
        )

        #TODO: Check dimensionality of y in posterior refinement paper.
        self.laplace.model.backbone.to(device)
        if not self.data_augmentation:
            features, targets = list(), list()
            with torch.no_grad():
                for x, y in train_loader:
                    x = x.to(device)

                    # Handle QM9 data separately
                    if self.dataset == 'qm9':
                        output = self.laplace.model.backbone(x)
                        for i, graph_idx in enumerate(output['graph_indexes'].unique()):
                            idx_vector = output['graph_indexes'] == graph_idx
                            features_ = output['scalar_features'][idx_vector].detach().cpu()
                            atoms = output['atoms'][idx_vector].detach().cpu()
                            features.append(
                                Data(x=features_, z=atoms, y=y[i].unsqueeze(0).cpu())
                            )

                    else:
                        features.append(self.laplace.model.backbone(x).detach().cpu())
                        targets.append(y.detach().cpu())

            # Handle QM9 data separately
            if self.dataset == 'qm9':
                train_loader = QM9Loader(features, batch_size=train_loader.batch_size, shuffle=True)
            else:
                features = torch.concat(features, dim=0)
                targets = torch.concat(targets, dim=0)
                train_loader = DataLoader(TensorDataset(features, targets), batch_size=train_loader.batch_size, shuffle=True)

        self.laplace.model.backbone.to('cpu')

        pbar = trange(num_epochs, desc="Fitting flow")
        step = 0
        for epoch in pbar:
            loss_epoch = 0.
            
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                if self.data_augmentation:
                    data = self.laplace.model.backbone(data)
                
                # Handle QM9 data separately
                if self.dataset == 'qm9':
                    data = {
                        'scalar_features': data.x,
                        'atoms': data.z,
                        'graph_indexes': data.batch,
                    }

                loss = svi.step(data, target, X_is_features=True)
                loss_step = loss / len(y)
                loss_epoch += loss

                if logger is not None:
                    logger.log('train_loss_step', loss_step, step)
                step += 1

                optimizer.step() #TODO: Check that this only needed for updating learning rate
            loss_epoch /= len(train_loader.dataset) #TODO: Check this length again

            if logger is not None:
                logger.log('train_loss_epoch', loss_epoch, step)

            pbar.set_postfix_str(
                f'train_loss={loss_epoch:.3e}, '
                #f'lr={lr:.3e}'
            )

    def predict(
        self,
        dataloader: DataLoader,
        num_posterior_samples: int = 100,
        stratified: bool = False,
        covariance_scale_factor: float = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        outputs = list()
        targets = list()
        self.laplace.model.eval()
        #samples = self.guide.get_posterior().sample((n_hmc_samples,))
        predictive = Predictive(self.pyro_model.model, guide=self.guide, num_samples=num_posterior_samples, return_sites=('_RETURN',))

        with torch.no_grad():
            self.laplace.model.backbone.to(device)
            for k, (x, y) in enumerate(dataloader):
                x, y = x.to(device), y.to(device)
                hidden_outs = self.laplace.model.backbone(x)
                output = predictive(hidden_outs, X_is_features=True)['_RETURN'] #samples x batch_size x output_size
                outputs.append(output.detach().cpu())
                targets.append(y.detach().cpu()) # batch_size
        outputs = torch.concat(outputs, dim=1) # num_posterior_samples x num_datapoints x output_size
        targets = torch.concat(targets, dim=0) # num_datapoints
        return outputs, targets

    def load_state_dict(self, state_dict, base=False):
        if base == True:
            self.laplace.load_state_dict(state_dict, base=False)
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.laplace.load_state_dict(state_dict['llla_inference'], base=False)
            self.transform = state_dict['transform']
            self.num_transforms = state_dict['num_transforms']
            self.n_classes = state_dict['n_classes']
            self.n_features = state_dict['n_features']
            self.data_aug = state_dict['data_aug']
            self.n_data = state_dict['n_data']
            self.dataset = state_dict['dataset']
            self.prior_precision = state_dict['prior_precision']
            #pyro.get_param_store().set_state(state_dict['param_store'])
            self.laplace.model.backbone.to(device)

            if self.dataset == 'qm9':
                self.pyro_model = PaiNNModelLL(
                    n_data = self.n_data,
                    n_features = self.n_features,
                    feature_extractor=self.laplace.model.backbone,
                    scale=self.laplace.llla_model.scale,
                    shift=self.laplace.llla_model.shift,
                    atom_references=self.laplace.llla_model.atom_references,
                    prior_prec=self.laplace.llla_model.prior_precision if self.prior_precision == 0 else self.prior_precision,
                    cuda = True if device == 'cuda' else False
                )
            else:
                self.pyro_model = ClassificationModelLL(
                    n_data = self.n_data,
                    n_classes = self.n_classes,
                    n_features = self.n_features,
                    feature_extractor = self.laplace.model.backbone,
                    prior_prec = self.laplace.llla_model.prior_precision if self.prior_precision == 0 else self.prior_precision,
                    cuda = True if device == 'cuda' else False
                )

            if len(self.laplace.covariance.shape) > 1:
                diag=False
            else:
                diag=True
            self.guide = AutoNormalizingFlowCustom(
                self.pyro_model.model,
                self.laplace.mean.to(device),
                self.laplace.covariance.to(device),
                diag = diag,
                flow_type = self.transform,
                flow_len = self.num_transforms,
                cuda = True if device == 'cuda' else False
            )

            #WE need to initalize the guide before we can load the state dict...
            svi = SVI(self.pyro_model.model, self.guide, optim=ClippedAdam({'lr': 1e-3}), loss=Trace_ELBO())
            X_train, y_train = get_data_batch(dataset=self.dataset)
            X_train, y_train = X_train.to(device), y_train.to(device)
            self.guide.to(device)
            svi.step(X_train, y_train)

            self.guide.load_state_dict(state_dict['guide_state_dict'])



        #else:

    def state_dict(self):
        #In principle for Pyro we should only need the nf_module, base_distr and flow_distr
        state_dict = {}
        state_dict['llla_inference'] = self.laplace.state_dict()
        state_dict['guide_state_dict'] = self.guide.state_dict()
        #state_dict['param_store'] = pyro.get_param_store().get_state()
        state_dict['transform'] = self.transform
        state_dict['num_transforms'] = self.num_transforms
        state_dict['n_classes'] = self.n_classes
        state_dict['n_features'] = self.n_features
        state_dict['data_aug'] = self.data_augmentation
        state_dict['n_data'] = self.n_data
        state_dict['dataset'] = self.dataset
        state_dict['prior_precision'] = self.prior_precision
        return state_dict
