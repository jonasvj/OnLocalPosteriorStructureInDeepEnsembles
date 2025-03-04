import os
import hydra
import torch
import pandas as pd
from tqdm import tqdm
from src.utils import set_seed
from omegaconf import open_dict


inference_to_name = {
    'src.inference.SWA': 'swa',
    'src.inference.SampleSWAG': 'sample_swag',
}


def load_model(path, new_inference, prepare_data=True, ood=False, shift=False):
    model_dict = torch.load(path)
    cfg = model_dict['cfg']
    state_dict = model_dict['state_dict']

    with open_dict(cfg):
        cfg.data.ood = ood
        cfg.data.shift = shift
        cfg.inference.init._target_ = new_inference

    # Data
    dm = hydra.utils.instantiate(cfg.data)
    if prepare_data:
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

    # Infererence
    inference = hydra.utils.instantiate(cfg.inference.init, model=model)
    inference.load_state_dict(state_dict, base=False)

    return cfg, dm, inference


@hydra.main(
    config_path='conf/',
    config_name='select_lr.yaml',
    version_base=None,
)
def main(cfg):
    print(cfg)
    # SWAG models for different seeds and learning rates
    val_metrics = list()
    files = [f for f in os.listdir(cfg.swag_dir) if cfg.model_name in f]
    for file in tqdm(files):
        val_metrics_run = dict()

        # Load SWAG model but with another inference class (SWA or SampleSWAG)
        path = os.path.join(cfg.swag_dir, file)
        try:
            original_cfg, dm, inference = load_model(path, cfg.new_inference_class)
        except RuntimeError as e:
            print(e)
            print(path)
            continue

        set_seed(original_cfg.seed)
        if original_cfg.inference.init.rank < original_cfg.num_posterior_samples:
            num_posterior_samples = original_cfg.inference.init.rank
        else:
            num_posterior_samples = original_cfg.num_posterior_samples
        print(num_posterior_samples)
        outputs_val, targets_val = inference.predict(
            dataloader=dm.val_dataloader(),
            num_posterior_samples=num_posterior_samples
        )
        val_stats = inference.compute_stats(
            outputs=outputs_val, targets=targets_val
        )[0]

        val_metrics_run['path'] = path
        val_metrics_run['lpd'] = val_stats['lpd']
        val_metrics_run['seed'] = original_cfg['seed']
        val_metrics_run['lr'] = original_cfg['inference']['fit']['optimizer']['lr']
        val_metrics.append(val_metrics_run)    
    print(len(val_metrics))

    val_metrics = pd.DataFrame(val_metrics)
    val_metrics = val_metrics.reset_index(drop=True) 
    val_metrics.to_csv(cfg.val_metrics_dest.lower(), index=False)
    print(val_metrics)

    selection_idx = val_metrics.groupby('seed').lpd.idxmax()
    selected_models = val_metrics.loc[selection_idx]
    selected_models.to_csv(cfg.selected_dest.lower(), index=False)
    print(selected_models)

    # Save models
    for index, row in selected_models.iterrows():
        path = row['path']
        new_path = path.replace(
            '/swag/',
            f'/selected_{inference_to_name[cfg.new_inference_class]}/'
        ).split('_lr=')[0] + '.pt'

        model_dict = torch.load(path)
        with open_dict(model_dict['cfg']):
            model_dict['cfg'].inference.init._target_ = cfg.new_inference_class
            model_dict['cfg'].num_posterior_samples = num_posterior_samples

        if not os.path.exists(new_path.rsplit('/', 1)[0]):
            os.makedirs(new_path.rsplit('/', 1)[0])

        torch.save(model_dict, new_path)


if __name__ == '__main__':
    main()
