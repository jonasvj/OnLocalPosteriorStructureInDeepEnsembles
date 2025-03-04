import torch
import torch.nn as nn
from tqdm import trange
from torch.optim import Optimizer
from src.utils import EarlyStopping
from src.logging import WandBLogger
from src.inference import InferenceBase
from torch.utils.data import DataLoader
from typing import Literal, Tuple, Callable
from torch.optim.lr_scheduler import LRScheduler


class MAP(InferenceBase):
    """
    Maximum a posterior estimation of a model.
    """
    def __init__(
        self,
        model: nn.Module,
        likelihood: Literal['classification', 'regression']
    ):
        super().__init__(model, likelihood=likelihood)


    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        logger: WandBLogger,
        num_epochs: int = 100,
        patience: int = 100,
        min_epochs: int = 100,
        lam_schedule: Callable = None,
        es_criterion: str = 'loss',
    ):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        optimizer = optimizer(params=self.model.parameters())
        scheduler = scheduler(optimizer=optimizer)

        early_stopping = EarlyStopping(
            patience=patience,
            min_epochs=min_epochs,
        )

        step = 0
        pbar = trange(num_epochs)
        for epoch in pbar:
            loss_epoch = 0

            self.model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                
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
                optimizer.zero_grad()
                loss_step.backward()
                optimizer.step()

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

            stop = early_stopping.check(self.model, val_stats[es_criterion], epoch)
            if stop:
                break

        self.model = (
            early_stopping.best_model if early_stopping.best_model is not None 
            else self.model
        )


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

        outputs = list()
        targets = list()

        self.model.eval()
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                output = self.model(x)
                
                outputs.append(output.detach().cpu()) # batch_size x output_size
                targets.append(y.detach().cpu())      # batch_size
        
        outputs = torch.concat(outputs, dim=0) # num_datapoints x output_size
        outputs = outputs.unsqueeze(dim=0) # num_posterior_samples x num_datapoints x output_size
        targets = torch.concat(targets, dim=0) # num_datapoints

        return outputs, targets


    def load_state_dict(self, state_dict, base: bool = False):
        for key in list(state_dict.keys()):
            state_dict[key.lstrip('model.')] = state_dict.pop(key)
        self.model.load_state_dict(state_dict)