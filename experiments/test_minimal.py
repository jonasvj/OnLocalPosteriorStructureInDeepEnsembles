import torch
import hydra
import argparse
from src.inference import *
from src.utils import set_seed
from omegaconf import open_dict


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_posterior_samples', type=int, default=1)

    args = parser.parse_args()
    return args


def load_model(path):
    model_dict = torch.load(path)
    cfg = model_dict['cfg']
    state_dict = model_dict['state_dict']

    # Data
    with open_dict(cfg):
        cfg.data.data_dir = 'data/'
    dm = hydra.utils.instantiate(cfg.data)
    dm.prepare_data()
    dm.setup()

    # Model
    model = hydra.utils.instantiate(cfg.model)

    # Infererence
    inference = hydra.utils.instantiate(cfg.inference.init, model=model)
    inference.load_state_dict(state_dict, base=False)

    return dm, inference


def main():
    args = cli()
    set_seed(args.seed)

    # Load data module and inference
    dm, inference = load_model(args.model_path)

    # Compute predictions for the test split of the data the model/inference was 
    # trained on.
    # Alternatively, provide another dataloader with the data you want predictions 
    # for. It is important that the dataloader does not shuffle the data.
    outputs_test, targets_test = inference.predict(
        dataloader=dm.test_dataloader(),
        num_posterior_samples=args.num_posterior_samples,
    )

    # The outputs has shape (num_posterior_samples, num_datapoints, output_size)
    # and the targets has shape (num_datapoints,).
    # Here we print the predicted output (i.e. logits for classification) for the 
    # first posterior sample and first datapoint and the corresponding target.
    print(outputs_test[0,0])
    print(targets_test[0])


if __name__ == '__main__':
    main()