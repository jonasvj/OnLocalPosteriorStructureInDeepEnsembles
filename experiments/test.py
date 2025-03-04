import os
import hydra
import torch
import pandas as pd
from src.inference import *
from src.utils import set_seed
from omegaconf import open_dict


def load_model(path, prepare_data=True, ood=True, shift=True):
    model_dict = torch.load(path)
    cfg = model_dict['cfg']
    state_dict = model_dict['state_dict']

    with open_dict(cfg):
        cfg.data.ood = ood
        cfg.data.shift = shift

    # Data
    dm = hydra.utils.instantiate(cfg.data)
    if cfg.model.name == 'painn':
        dm.prepare_data()
        dm.setup()
    elif prepare_data:
        dm.prepare_data()
        dm.setup()

    # Model
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

    if 'refined_llla' in path:
        model = LastLayerLaplace(model, cfg.inference.init.likelihood)

    # Infererence
    inference = hydra.utils.instantiate(cfg.inference.init, model=model)
    inference.load_state_dict(state_dict, base=False)

    return dm, inference


def load_ensemble(paths, ood=True, shift=True):
    inference_list = list()
    for i, path in enumerate(paths):
        dm, inference = load_model(
            path, prepare_data=(i == len(paths) - 1), ood=ood, shift=shift
        )
        inference_list.append(inference)

    likelihood = inference_list[0].likelihood
    if all(type(model) == MAP for model in inference_list):
        inference = DeepEnsemble(model=inference_list, likelihood=likelihood)
    elif all(type(model) == SWAG for model in inference_list):
        inference = MultiSWAG(model=inference_list, likelihood=likelihood)
    elif all(type(model) == LastLayerLaplace for model in inference_list):
        inference = MoLA(model=inference_list, likelihood=likelihood)
    elif all(type(model) == PosteriorRefinedLastLayerLaplace for model in inference_list):
        inference = MoFlowLA(model=inference_list, likelihood=likelihood)
    elif all(type(model) == SWA for model in inference_list):
        inference = MultiSWA(model=inference_list, likelihood=likelihood)
    elif all(type(model) == SampleSWAG for model in inference_list):
        inference = MultiSampleSWAG(model=inference_list, likelihood=likelihood)
    elif all(type(model) == IVONFromScratch for model in inference_list):
        inference = MultiIVONFromScratch(model=inference_list, likelihood=likelihood)
    elif all(type(model) == MonteCarloDropout for model in inference_list):
        inference = MultiMCDO(model=inference_list, likelihood=likelihood)
    else:
        raise ValueError('Invalid ensemble.')
    return dm, inference


@hydra.main(
    config_path='conf/',
    config_name='test_config.yaml', 
    version_base=None,
)
def main(cfg):
    set_seed(cfg.test_seed)
    model_seeds = [str(seed) for seed in cfg.model_seeds]

    if len(model_seeds) == 1:
        model_path = cfg.model_prefix + model_seeds[0] + cfg.model_suffix
        dm, inference = load_model(
            model_path,
            ood=cfg.compute_ood_stats,
            shift=cfg.compute_shift_stats,
        )
        K = 1
    elif len(model_seeds) > 1:
        model_paths = [
            cfg.model_prefix + seed + cfg.model_suffix for seed in model_seeds
        ]
        dm, inference = load_ensemble(
            model_paths,
            ood=cfg.compute_ood_stats,
            shift=cfg.compute_shift_stats,
        )
        K = len(model_seeds)

    results = list()
    if cfg.compute_test_stats:
        set_seed(cfg.test_seed)
        # In-distribution metrics
        outputs_test, targets_test = inference.predict(
            dataloader=dm.test_dataloader(),
            num_posterior_samples=cfg.num_posterior_samples,
            stratified=cfg.stratified_sampling,
            covariance_scale_factor=cfg.covariance_scale_factor,
        )
        stats_test = inference.compute_stats(
            outputs=outputs_test,
            targets=targets_test,
            subsample_sizes=cfg.subsample_sizes,
        )
        for row in stats_test:
            row['split'] = 'test'
        results.extend(stats_test)

    if cfg.compute_shift_stats:
        for dataset, dataloader in dm.shift_dataloaders().items():
            set_seed(cfg.test_seed)
            # Shift-distribution metrics
            outputs_shift, targets_shift = inference.predict(
                dataloader=dataloader,
                num_posterior_samples=cfg.num_posterior_samples,
                stratified=cfg.stratified_sampling,
                covariance_scale_factor=cfg.covariance_scale_factor,
            )
            stats_shift = inference.compute_stats(
                outputs=outputs_shift,
                targets=targets_shift,
                subsample_sizes=cfg.subsample_sizes,
            )
            for row in stats_shift:
                row['split'] = dataset
            results.extend(stats_shift)

    if cfg.compute_ood_stats and cfg.compute_test_stats:
        for dataset, dataloader in dm.ood_dataloaders().items():
            set_seed(cfg.test_seed)
            # Out-of-distribution metrics
            outputs_ood, targets_ood = inference.predict(
                dataloader=dataloader,
                num_posterior_samples=cfg.num_posterior_samples,
                stratified=cfg.stratified_sampling,
                covariance_scale_factor=cfg.covariance_scale_factor,
            )
            stats_ood = inference.compute_ood_stats(
                outputs_id=outputs_test,
                outputs_ood=outputs_ood,
                subsample_sizes=cfg.subsample_sizes,
            )
            for row in stats_ood:
                row['split'] = dataset
            results.extend(stats_ood)

    for row in results:
        row['inference'] = inference.__class__.__name__
        row['K'] = K
        row['stratified'] = cfg.stratified_sampling
        row['model_seeds'] = ','.join(model_seeds)
        row['test_seed'] = cfg.test_seed
        row['name'] = cfg.stats_name

    df = pd.DataFrame(results)
    print(df)
    print(df.columns)

    torch.save(
        {'cfg': cfg, 'results': results},
        os.path.join(cfg.stats_dir, cfg.stats_name + '.pt')
    )


if __name__ == '__main__':
    main()
