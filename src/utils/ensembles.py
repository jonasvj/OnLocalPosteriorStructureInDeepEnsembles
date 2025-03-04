import sys
import random
import pandas as pd
from typing import List


def generate_ensembles(
    ensemble_sizes: List[int],
    model_ids: List[int],
    num_ensembles: 30, 
    save_path: str = 'ensembles.csv',
) -> None:
    ensembles = list()

    for ensemble_size in ensemble_sizes:
        for ensemble_id in range(1, num_ensembles + 1):
            # Generate unique seed from size and id (using Cantor's pairing function)
            seed = int(
                (ensemble_size + ensemble_id)
                * (ensemble_size + ensemble_id + 1)/2 
                + ensemble_id
            )
            random.seed(seed)
            ensemble = sorted(random.sample(model_ids, ensemble_size))
            
            ensembles.append({
                'K': ensemble_size, 
                'ensemble_id': ensemble_id,
                'seed': seed,
                'model_ids': ','.join(str(e) for e in ensemble)
            })

    ensembles = pd.DataFrame(ensembles)
    ensembles.to_csv(save_path, index=False, sep='\t')


def get_ensemble_str(
    ensemble_size: int,
    ensemble_id: int,
    load_path: str = 'ensembles.csv',
) -> str:
    ensembles = pd.read_csv(load_path, sep='\t')
    ensemble = ensembles[
        (ensembles.K == ensemble_size) & (ensembles.ensemble_id == ensemble_id)
    ].model_ids.item()
    
    return ensemble


def get_ensemble(
    ensemble_size: int,
    ensemble_id: int,
    load_path: str = 'ensembles.csv',
)-> None:
    ensemble = get_ensemble_str(ensemble_size, ensemble_id, load_path)
    print(ensemble, file=sys.stdout)


if __name__ == '__main__':
    ensemble_sizes = list(range(2, 21))
    model_ids = list(range(1, 31))
    num_ensembles = 30
    #generate_ensembles(ensemble_sizes, model_ids, num_ensembles)

    print(get_ensemble_str(5, 20))
    get_ensemble(5, 20)
