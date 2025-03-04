import argparse
import numpy as np
import pandas as pd
from itertools import product
from src.utils import load_stats

INFERENCES_TO_KEEP = [
    'DE',
    'SWA',
    'SWAG',
    'LLLA',
    'LLLA-NF-10',
]

K = [1, 2, 5, 10, 20]


def highlight_max(s, props='', precision=3):
    return  np.where(
        s.round(precision) == np.nanmax(s.round(precision).values), props, ''
    )


def highlight_min(s, props='', precision=3):
    return np.where(
        s.round(precision) == np.nanmin(s.round(precision).values), props, ''
    )


def cli():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--directory', type=str, nargs='+')
    parser.add_argument('--table_dest', type=str)

    args = parser.parse_args()
    return args


def main():
    args = cli()
    
    df_wrn_16_4 = load_stats(args.directory, 'wrn-16-4', '/cifar10/')
    df_wrn_28_10 = load_stats(args.directory, 'wrn-28-10', '/cifar10/')
    df_wrn_16_4_cifar100 = load_stats(args.directory, 'wrn-16-4', '/cifar100/')
    df_painn = load_stats(args.directory, 'painn', '/qm9/')
    df_painn['acc'] = -df_painn['mae']
    df_bert = load_stats(args.directory, 'bert-256-4', '/sst2/')
    df_wrn_16_4['data'] = df_wrn_16_4['data'] + ' (WRN-16-4)'
    df_wrn_28_10['data'] = df_wrn_28_10['data'] + ' (WRN-28-10)'
    df_wrn_16_4_cifar100['data'] = df_wrn_16_4_cifar100['data'] + ' (WRN-16-4)'
    df = pd.concat([df_wrn_16_4, df_wrn_28_10, df_wrn_16_4_cifar100, df_painn, df_bert])
    df.to_csv('all_results.csv', index=False)
    
    #df = pd.read_csv('all_results.csv', low_memory=False)
    #df.loc[df.data == 'QM9', 'acc'] = 1000*df.loc[df.data == 'QM9', 'acc']  # ONLY FOR SEM TABLE

    # Filter results
    df = df[df.inference.isin(INFERENCES_TO_KEEP)]
    df = df[df.K.isin(K)]
    df = df[
        (df.inference == 'DE') | # MAP / deep ensembles
        (df.inference == 'SWA') | # SWA / SWA ensembles
        ((df.K == 1) & (df.stratified == False) & (df.num_posterior_samples == 200)) | # BNNs
        ((df.stratified == True) & (df.K > 1) & (df.num_posterior_samples == 200)) # BNN ensembles
    ]

    datasets = df.data.unique()
    ks = df.K.unique()

    df = df[['K', 'data', 'inference', 'acc', 'lpd', 'ece']] 
    df = df.rename(columns={'inference': 'Inference', 'data': 'Data', 'acc': 'Acc.', 'lpd': 'ELPD', 'ece': 'ECE'})
    df = df.replace('LLLA-NF-10', 'LA-NF')
    df.Inference = pd.Categorical(df.Inference, categories=['DE', 'SWA', 'SWAG', 'LLLA', 'LA-NF'])
    df.Data = pd.Categorical(df.Data, categories=['CIFAR-10 (WRN-16-4)', 'CIFAR-10 (WRN-28-10)', 'CIFAR-100 (WRN-16-4)', 'SST-2', 'QM9'])

    df_mean = df.pivot_table(values=['Acc.', 'ELPD', 'ECE'], index=['K', 'Inference'], columns=['Data'])#, sort=False, aggfunc='sem') # AGGFUNC ONLY FOR SEM TABLE
    df_mean= df_mean.reorder_levels(order=[1,0], axis=1)
    df_mean = df_mean.sort_index(axis=1, level=0, sort_remaining=False)

    # Format table
    min_cols = list(product(datasets, ['ECE']))
    max_cols = list(product(datasets, ['Acc.', 'ELPD']))
    max_cols.remove(('QM9', 'Acc.'))
    min_cols.append(('QM9', 'MAE'))
    df_mean.columns = df_mean.columns.map(lambda x: ('QM9', 'MAE') if x == ('QM9', 'Acc.') else x)
    df_mean[('QM9', 'MAE')] = -df_mean[('QM9', 'MAE')]*1000 # THIS LINE SHOULD ONLY BE USED IN THE MEAN TABLE
    print(df_mean)

    df_mean = df_mean.style.format(precision=3, na_rep='', escape='latex')
    idx = pd.IndexSlice
    for k in ks:
        slice_ = idx[idx[k,:], idx[max_cols]]
        df_mean = df_mean.apply(
            highlight_max, props="textbf:--rwrap;", axis=0, subset=slice_
        )

        slice_ = idx[idx[k,:], idx[min_cols]]
        df_mean = df_mean.apply(
            highlight_min, props="textbf:--rwrap;", axis=0, subset=slice_
        )
    df_mean.to_latex(
        args.table_dest,
        hrules=True,
        clines='skip-last;data',
        sparse_index=True,
        sparse_columns=True,
        multicol_align='c'
    )

if __name__ == '__main__':
    main()