import torch
import hydra
from src.utils import *
from omegaconf import OmegaConf
from src.inference import *
from src.inference import all_regression_metrics

@hydra.main(
    config_path='conf/',
    config_name='config.yaml',
    version_base=None,
)
def main(cfg):
    set_seed(cfg.seed)
    print(cfg)
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Logging
    print('Initializing logger.')
    logger = hydra.utils.instantiate(cfg.logger)
    hparams = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    logger.init_run(hparams)
    print('Initializing logger DONE.')

    # Data
    print('Initializing data.')
    dm = hydra.utils.instantiate(cfg.data)
    dm.prepare_data()
    dm.setup()
    print('Initializing data DONE.')

    # Model
    print('Initializing model.')
    if cfg.model.name == 'painn':
        y_mean, y_std, atom_refs = dm.get_target_stats(
            remove_atom_refs=True,
            divide_by_atoms=True,
        )
        head = hydra.utils.instantiate(cfg.model.head)
        head = head(mean=y_mean.item(), std=y_std.item(), atom_refs=atom_refs)
        model = hydra.utils.instantiate(cfg.model, head=head)
    else:
        model = hydra.utils.instantiate(cfg.model)

    if cfg.inference.name == 'refined_llla' or cfg.inference.name == "posterior_refined_llla":
        model = LastLayerLaplace(model, cfg.inference.init.likelihood)
    print('Initializing model DONE.')

    # Initialize inference method
    print('Initializing inference.')
    inference = hydra.utils.instantiate(cfg.inference.init, model=model)
    if cfg.model.name == 'painn':
        inference.y_var = 0.02**2

    # Possibly load pretrained model
    if cfg.pretrained_model is not None:
        inference.load_state_dict(
            torch.load(cfg.pretrained_model)['state_dict'],
            base=cfg.inference.load_base
        )
    inference.model.to(device)
    print('Initializing inference DONE.')

    # Fit model with inference method
    print('Fitting model.')
    hydra.utils.call(
        cfg.inference.fit,
        inference,
        train_loader=dm.train_dataloader(),
        val_loader=dm.val_dataloader(),
        logger=logger,
    )
    print('Fitting model DONE.')

    # Compute validation statistics
    if cfg.inference.init.likelihood == 'classification':
        keys_to_log = ['loss', 'lpd', 'acc', 'avg_conf', 'avg_entropy', 'ece', 'mce', 'brier']
    elif cfg.inference.init.likelihood == 'regression':
        keys_to_log = all_regression_metrics

    if not cfg.disable_eval:
        print('Computing validation statistics.')
        set_seed(0)
        outputs_val, targets_val = inference.predict(
            dataloader=dm.val_dataloader(),
            num_posterior_samples=cfg.num_posterior_samples
        )
        val_stats = inference.compute_stats(
            outputs=outputs_val, targets=targets_val
        )
        # Log validation statics
        logger.log_table(
            val_stats[0],
            keys=keys_to_log,
            table_name='val_metrics'
        )
        print('Validation statistics:')
        print(val_stats[0])
        print('Computing validation statistics DONE.')

    # Compute test statistics
    if not cfg.disable_eval:
        print('Computing test statistics.')
        set_seed(0)
        outputs_test, targets_test = inference.predict(
            dataloader=dm.test_dataloader(),
            num_posterior_samples=cfg.num_posterior_samples
        )
        test_stats = inference.compute_stats(
            outputs=outputs_test, targets=targets_test
        )
        # Log test statics
        logger.log_table(
            test_stats[0],
            keys=keys_to_log,
            table_name='test_metrics'
        )
        print('Test statistics:')
        print(test_stats[0])
        print('Computing test statistics DONE.')

    # Save model
    print('Saving inference.')
    model.to('cpu')
    inference.to('cpu')
    torch.save(
        {'cfg': cfg, 'state_dict': inference.state_dict(), 'val_lpd': val_stats[0]['lpd'],  'test_lpd': test_stats[0]['lpd']},
        f"{cfg.model_dir}/{cfg.model_name}".lower()
    )
    print('Saving inference DONE.')

    logger.end_run()
    print('Logging DONE.')

if __name__ == '__main__':
    main()