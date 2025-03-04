import torch
import torch.nn as nn
from ivon import IVON
from tqdm import trange
from src.utils import EarlyStopping
from src.logging import WandBLogger
from src.inference import InferenceBase
from torch.utils.data import DataLoader
from typing import Literal, Tuple, Callable
from torch.optim.lr_scheduler import LRScheduler


class IVONFromScratch(InferenceBase):
    """
    Model training with IVON.
    """
    def __init__(
        self,
        model: nn.Module,
        likelihood: Literal['classification', 'regression'],
        optimizer: IVON,
    ):
        super().__init__(model, likelihood=likelihood)
        self.partial_optimizer = optimizer

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)
        self.optimizer = optimizer(params=self.model.parameters())


    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        scheduler: LRScheduler,
        scheduler_warmup: LRScheduler,
        logger: WandBLogger,
        num_epochs: int = 200,
        warmup_epochs: int = 5,
        patience: int = 200,
        min_epochs: int = 200,
        lam_schedule: Callable = None,
        es_criterion: str = 'loss',
    ):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)

        scheduler_after_warmup = scheduler
        scheduler = scheduler_warmup(optimizer=self.optimizer)

        early_stopping = EarlyStopping(
            patience=patience,
            min_epochs=min_epochs,
        )

        step = 0
        pbar = trange(num_epochs)
        for epoch in pbar:
            if epoch == warmup_epochs:
                scheduler = scheduler_after_warmup(optimizer=self.optimizer)

            loss_epoch = 0
            self.model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)

                with self.optimizer.sampled_params(train=True):
                    self.optimizer.zero_grad()
                    output = self.model(x)

                    if self.likelihood == 'classification':
                        loss = self.loss_func(output, y, reduction='sum')
                    elif self.likelihood == 'regression':
                        self.lam = lam_schedule(step)
                        logger.log('lambda', self.lam, step)
                        loss = self.loss_func(
                            output,
                            y,
                            reduction='sum',
                            lam=self.lam,
                            y_var=self.y_var,
                        )

                    loss_step = loss / len(y)
                    loss_step.backward()            
                self.optimizer.step()

                # Metrics for tracking
                loss_step = loss_step.detach().item()
                loss_epoch += loss.detach().item()

                # Track step metrics
                logger.log('train_loss_step', loss_step, step)

                step += 1
            
            # Update learning rate
            lr = scheduler.get_last_lr()[0]
            scheduler.step()

            # Train metrics
            loss_epoch /= len(train_loader.dataset)

            # Validation metrics
            outputs_val, targets_val = self.predict(
                dataloader=val_loader, num_posterior_samples=1
            )
            val_stats = self.compute_stats(
                outputs=outputs_val, targets=targets_val
            )[0]

            # Track metrics
            logger.log('train_loss_epoch', loss_epoch, step)
            logger.log('learning_rate', lr, step)
            for key, value in val_stats.items():
                logger.log('val_' + key + '_epoch', value, step)

            pbar.set_postfix_str(
                f'train_loss={loss_epoch:.3e}, '
                f'val_loss={val_stats["loss"]:.3e}, '
                f'lr={lr:.3e}'
            )

            stop = early_stopping.check((self.model, self.optimizer.state_dict()), val_stats[es_criterion], epoch)
            if stop:
                break

        self.model, optim_state_dict = (
            early_stopping.best_model if early_stopping.best_model is not None 
            else (self.model,  self.optimizer.state_dict())
        )
        self.optimizer = self.partial_optimizer(params=self.model.parameters())
        self.optimizer.load_state_dict(optim_state_dict)


    def predict(
        self,
        dataloader: DataLoader,
        num_posterior_samples: int = 1,
        stratified: bool = False,
        covariance_scale_factor: float = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts with the MAP solution.

        Args:
            dataloader: A PyTorch dataloader containing the data to predict.
            num_posterior_samples: Number of posterior samples to compute
                predictions with. This argument is unused as the MAP solution
                is used for making predictions.
            stratified: Boolean indicating if sampling should be stratified in
                case of a multimodal posterior. This argument is unused for the
                MAP solution.

        Returns:
            A tuple of tensors with predictions for the data and the
            corresponding targets. The predictions has shape
            (num_posterior_samples, num_datapoints, output_size) where 
            output_size is the size of a single output from the model and
            num_posterior_samples is equal to 1. The targets has shape 
            (num_datapoints,).
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)
        self.model.eval()

        outputs = list()
        targets = list()

        with torch.no_grad():
            # Evaluate at MAP
            if num_posterior_samples == 1:
                for x, y in dataloader:
                    x, y = x.to(device), y.to(device)
                    output = self.model(x)

                    outputs.append(output.detach().cpu()) # batch_size x output_size
                    targets.append(y.detach().cpu())      # batch_size

                outputs = torch.concat(outputs, dim=0) # num_datapoints x output_size
                outputs = outputs.unsqueeze(dim=0) # num_posterior_samples x num_datapoints x output_size
                targets = torch.concat(targets, dim=0) # num_datapoints
            else:
                for s in range(num_posterior_samples):
                    with self.optimizer.sampled_params():
                        fixed = None
                        outputs_sample = list()
                        for x, y in dataloader:
                            x, y = x.to(device), y.to(device)

                            output = self.model(x)
                            outputs_sample.append(output.detach().cpu()) # batch_size x output_size

                            if s == 0:
                                targets.append(y.detach().cpu()) # batch_size

                    outputs_sample = torch.concat(outputs_sample, dim=0) # num_datapoints x output_size
                    outputs.append(outputs_sample)

                outputs = torch.stack(outputs, dim=0) # num_posterior_samples x num_datapoints x output_size
                targets = torch.concat(targets, dim=0) # num_datapoints

        return outputs, targets


    def state_dict(self):
        state_dict = {}
        state_dict['model'] = self.model.state_dict()
        state_dict['optim'] = self.optimizer.state_dict()

        return state_dict


    def load_state_dict(self, state_dict, base: bool = False):
        self.model.load_state_dict(state_dict['model'])
        self.optimizer = self.partial_optimizer(params=self.model.parameters())
        self.optimizer.load_state_dict(state_dict['optim'])