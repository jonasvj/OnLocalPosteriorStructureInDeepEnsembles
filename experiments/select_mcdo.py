import os
import torch
import argparse
import pandas as pd
from src import get_norm_name


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--model', type=str, default='wrn-28-10')

    args = parser.parse_args()
    return args


def main():
    args = cli()

    val_metrics = list()
    files = sorted(os.listdir(args.model_dir))
    for file in files:
        if args.model not in file:
            continue
        model_dict = torch.load(os.path.join(args.model_dir, file))
        row = {}
        row['seed'] = model_dict['cfg'].seed
        if args.model == 'bert':
            row['do'] = model_dict['cfg'].model.backbone.hidden_dropout_prob
        else:
            row['do'] = model_dict['cfg'].model.backbone.dropout_rate
        row['model_dir'] = model_dict['cfg'].model_dir
        row['model'] = model_dict['cfg'].model.name
        row['model_name'] = model_dict['cfg'].model_name
        row['val_lpd'] = model_dict['val_lpd']
        val_metrics.append(row)

    val_metrics = pd.DataFrame(val_metrics)
    print(val_metrics)

    selection_idx = val_metrics.groupby('seed').val_lpd.idxmax()
    selected_mcdo = val_metrics.loc[selection_idx]
    print(selected_mcdo)
   
    # Create symbolic links to the selected models
    for index, row in selected_mcdo.iterrows():
        old_dir = row['model_dir'].lower()
        new_dir = old_dir.replace('/mcdo', '/selected_mcdo').lower()
        os.makedirs(new_dir, exist_ok=True)
        print(old_dir)
        print(new_dir)
        old_name = row['model_name'].lower()
        new_name = f'{row["model"]}_seed={row["seed"]}.pt'.lower()
        print(old_name)
        print(new_name)
        old_path = os.path.join(old_dir, old_name).lower()
        new_path = os.path.join(new_dir, new_name).lower()
        print(old_path)
        print(new_path)
        print()
        os.symlink(src=old_path, dst=new_path)


if __name__ == '__main__':
    main()