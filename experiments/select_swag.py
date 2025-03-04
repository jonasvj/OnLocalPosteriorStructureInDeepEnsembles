import os
import json
import wandb
import hydra
import pandas as pd
from requests.exceptions import HTTPError


def get_val_metrics(run) -> pd.DataFrame:
    artifacts = run.logged_artifacts()

    # Search for artifact with validation metrics
    val_metrics_artifact = None
    for artifact in artifacts:
        if 'val_metrics' in artifact.name and 'latest' in artifact.aliases:
            val_metrics_artifact = artifact

    if val_metrics_artifact is None:
        raise ValueError(f'Validation metrics not found for run {run.name}.')    

    # Download validation metrics
    artifact_path = val_metrics_artifact.download(
        f'wandb/artifacts/{val_metrics_artifact.source_name}'
    )

    # Load validation metrics into a pandas DataFrame
    with open(f'{artifact_path}/val_metrics.table.json') as f:
        json_dict = json.load(f)
        df = pd.DataFrame(json_dict['data'], columns=json_dict['columns'])
    
    return df


@hydra.main(
    config_path='conf/',
    config_name='select_swag.yaml',
    version_base=None,
)
def main(cfg):
    api = wandb.Api(timeout=999)
    runs = api.runs(
        f'{cfg.logger.entity}/{cfg.logger.project_name}',
        filters={
            'state':  "finished",
            'config.inference.name': 'swag',
            'config.data.name': cfg.data.name,
            'config.model.name': cfg.model.name,
            'config.data.data_augmentation': cfg.data_aug,
            "$and": [{'created_at': {"$gt": cfg.created_after}}]
        }
    )
    print(len(runs))

    failed_runs = list()
    val_metrics = list()
    for i, run in enumerate(runs):
        print(f'Run {i}')
        # Get validation metrics
        try:
            val_metrics_run = get_val_metrics(run)
            val_metrics_run['seed'] = run.config['seed']
            val_metrics_run['lr'] = run.config['inference']['fit']['optimizer']['lr']
            val_metrics_run['model_dir'] = run.config['model_dir']
            val_metrics_run['model_name'] = run.config['model_name']
            val_metrics.append(val_metrics_run)
        except (ValueError, HTTPError):
            failed_runs.append(run.name)
    print(len(val_metrics))
    print('Runs with missing validation metrics:')
    print('\n'.join(failed_runs))

    val_metrics = pd.concat(val_metrics)
    val_metrics = val_metrics.reset_index(drop=True) 
    val_metrics.to_csv(cfg.swag_val_metrics_dest.lower(), index=False)
    print(val_metrics)

    selection_idx = val_metrics.groupby('seed').lpd.idxmax()
    selected_swag = val_metrics.loc[selection_idx]
    selected_swag.to_csv(cfg.selected_swag_dest.lower(), index=False)
    print(selected_swag)

    # Create symbolic links to the selected models
    if cfg.create_links:
        save_dir = f'models/{cfg.data.name}/aug_{cfg.data_aug}/selected_swag'
        os.makedirs(save_dir.lower(), exist_ok=True)
        for index, row in selected_swag.iterrows():
            print(os.path.join(row['model_dir'], row['model_name']).lower())
            print(os.path.join(save_dir, f'{cfg.model.name}_seed={row["seed"]}.pt').lower())

            source = os.path.join(row['model_dir'], row['model_name']).lower()
            os.symlink(
                src=source,
                dst=os.path.join(save_dir, f'{cfg.model.name}_seed={row["seed"]}.pt').lower()
            )


if __name__ == '__main__':
    main()
