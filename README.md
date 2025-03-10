# On Local Posterior Structure in Deep Ensembles
Code repository for the paper "On Local Posterior Structure in Deep Ensembles" accepted to AISTATS 2025.

All trained models are available at https://data.dtu.dk/projects/On_Local_Posterior_Structure_in_Deep_Ensembles_-_Trained_Models/236885.

For each dataset there are placed `selected_<METHODS>.txt` files for SWAG and MCDO models. These contain names of the best performing models (as chosen based on validation loss) with cross-validated hyperparameters of the respective methods for a given seed. 

## Setup
To setup the code environment execute the following commands:
1. `git clone git@github.com:jonasvj/OnLocalPosteriorStructureInDeepEnsembles.git`
2. `cd olpside/`
3. `conda env create -f environment.yml`
4. `conda activate olpside`
5. `pip install -e .`

## Training
Below we provide instructions for training the models used to produce the main results in the paper.

**Note on logging**\
The training script uses `wandb` for logging by default. Your `wandb` "entity" and "project" can be specified by adding the following arguments to `experiments/train.py`
```
logger.entity=YOUR_WANDB_ENTITY
logger.project_name=YOUR_WANDB_PROJECT
```
Logging can also be completely disabled by adding
```
logger.disable=true
```
`wandb` logging is, however, currently needed for hyperparameter selection in SWAG.


### Maximum-a-posterior estimation
Use the following commands to train MAP models for the different architectures / datasets.

Note that `SEED`$\in \{1,2,\dots,30\}$.

#### CIFAR-10 (WRN-16-4 / WRN-28-10)
For the WRN-16-4 architecture:
```
mkdir -p models/cifar10/aug_true/map

python3 experiments/train.py \
    seed=${SEED}
```
For the WRN-28-10 architecture:
```
mkdir -p models/cifar10/aug_true/map

python3 experiments/train.py \
    seed=${SEED} \
    data.batch_size_inference=256 \
    model.backbone.depth=28 \
    model.backbone.widen_factor=10 \
    model.head.input_dim=640 \
    inference.fit.num_epochs=200
```

#### CIFAR-100 (WRN-16-4)
```
mkdir -p models/cifar100/aug_true/map

python3 experiments/train.py \
    seed=${SEED} \
    data=cifar100 \
    model.head.num_classes=100
```

#### SST-2
```
mkdir -p models/sst2/aug_false/map

python3 experiments/train.py \
    seed=${SEED} \
    data=SST2 \
    data.batch_size_train=64 \
    data.batch_size_inference=64 \
    model=bert \
    model.head.num_classes=2 \
    inference.fit.optimizer.weight_decay=3e-3 \
    inference.fit.optimizer.lr=5e-3
```

#### QM9
```
mkdir -p models/qm9/aug_false/map

python3 experiments/train.py \
    seed=${SEED} \
    data=qm9 \
    model=painn \
    inference.init.likelihood=regression \
    inference.fit.num_epochs=650 \
    inference.fit.patience=50 \
    inference.fit.min_epochs=650 \
    inference.fit.es_criterion=nll \
    inference.fit.optimizer._target_=torch.optim.AdamW \
    inference.fit.optimizer.lr=5e-4 \
    inference.fit.optimizer.weight_decay=0.01 \
    ~inference.fit.optimizer.momentum \
    ~inference.fit.optimizer.nesterov \
    inference.fit.lam_schedule._target_=src.utils.const_lam_schedule \
    +inference.fit.lam_schedule.const=0 \
    ~inference.fit.lam_schedule.num_warmup_steps \
    ~inference.fit.lam_schedule.num_interpolation_steps
```

### Stochastic Weight Averaging(-Gaussian) (SWA / SWAG)
Use the following commands to train SWA and SWAG models for the different architectures / datasets.

Note that `SEED`$\in \{1,2,\dots,30\}$ and `LR`$\in\{1\mathrm{e}{-1}, 5\mathrm{e}{-2}, 4\mathrm{e}{-2}, 3\mathrm{e}{-2}, 2\mathrm{e}{-2}, 1\mathrm{e}{-2}, 9\mathrm{e}{-3}, 8\mathrm{e}{-3}, 7\mathrm{e}{-3}, 6\mathrm{e}{-3}, 5\mathrm{e}{-3}, 4\mathrm{e}{-3}, 3\mathrm{e}{-3}, 2\mathrm{e}{-3}, 1\mathrm{e}{-3}, 9\mathrm{e}{-4}, 8\mathrm{e}{-4}, 7\mathrm{e}{-4}, 6\mathrm{e}{-4}, 5\mathrm{e}{-4}, 1\mathrm{e}{-4}\}$.

#### CIFAR-10 (WRN-16-4 / WRN-28-10)
For the WRN-16-4 architecture:
```
mkdir -p models/cifar10/aug_false/swag

python3 experiments/train.py \
    seed=${SEED} \
    data.data_augmentation=false \
    inference=swag \
    inference.fit.optimizer.lr=${LR} \
    model_name=\"wrn-16-4_seed=${SEED}_lr=${LR}.pt\" \
    pretrained_model=\"models/cifar10/aug_true/map/wrn-16-4_seed=${SEED}.pt\"
```
For the WRN-28-10 architecture:
```
mkdir -p models/cifar10/aug_false/swag

python3 experiments/train.py \
    seed=${SEED} \
    data.data_augmentation=false \
    data.batch_size_inference=256 \
    model.backbone.depth=28 \
    model.backbone.widen_factor=10 \
    model.head.input_dim=640 \
    inference=swag \
    inference.init.rank=25 \
    inference.fit.optimizer.lr=${LR} \
    model_name=\"wrn-28-10_seed=${SEED}_lr=${LR}.pt\" \
    pretrained_model=\"models/cifar10/aug_true/map/wrn-28-10_seed=${SEED}.pt\"
```

#### CIFAR-100 (WRN-16-4)
```
mkdir -p models/cifar100/aug_false/swag

python3 experiments/train.py \
    seed=${SEED} \
    data=cifar100 \
    data.data_augmentation=false \
    model.head.num_classes=100 \
    inference=swag \
    inference.fit.optimizer.lr=${LR} \
    model_name=\"wrn-16-4_seed=${SEED}_lr=${LR}.pt\" \
    pretrained_model=\"models/cifar100/aug_true/map/wrn-16-4_seed=${SEED}.pt\"
```

#### SST-2
```
mkdir -p models/sst2/aug_false/swag

python3 experiments/train.py \
    seed=${SEED} \
    data=SST2 \
    data.batch_size_train=64 \
    data.batch_size_inference=64 \
    model=bert \
    model.head.num_classes=2 \
    inference=swag \
    inference.fit.optimizer.weight_decay=3e-3 \
    inference.fit.optimizer.lr=${LR} \
    model_name=\"bert-256-4_seed=${SEED}_lr=${LR}.pt\" \
    pretrained_model=\"models/sst2/aug_false/map/bert-256-4_seed=${SEED}.pt\"
```

#### QM9
For the PaiNN model on the QM9 dataset, we use the learning rate grid `LR`$\in\{1\mathrm{e}{-6}, 5\mathrm{e}{-7}, 4\mathrm{e}{-7}, 3\mathrm{e}{-7}, 2\mathrm{e}{-7}, 1\mathrm{e}{-7}, 9\mathrm{e}{-8}, 8\mathrm{e}{-8}, 7\mathrm{e}{-8}, 6\mathrm{e}{-8}, 5\mathrm{e}{-8}, 4\mathrm{e}{-8}, 3\mathrm{e}{-8}, 2\mathrm{e}{-8}, 1\mathrm{e}{-8}, 9\mathrm{e}{-9}, 8\mathrm{e}{-9}, 7\mathrm{e}{-9}, 6\mathrm{e}{-9}, 5\mathrm{e}{-9}, 4\mathrm{e}{-9}, 3\mathrm{e}{-9}, 2\mathrm{e}{-9}, 1\mathrm{e}{-9}, 9\mathrm{e}{-10}, 8\mathrm{e}{-10}, 7\mathrm{e}{-10}, 6\mathrm{e}{-10}, 5\mathrm{e}{-10}, 1\mathrm{e}{-10}\}$.
```
mkdir -p models/qm9/aug_false/swag

python3 experiments/train.py \
    seed=${SEED} \
    data=qm9 \
    model=painn \
    inference=swag \
    inference.init.likelihood=regression \
    inference.fit.optimizer.weight_decay=0.01 \
    inference.fit.optimizer.lr=${LR} \
    model_name=\"painn_seed=${SEED}_lr=${LR}.pt\" \
    pretrained_model=\"models/qm9/aug_false/map/painn_seed=${SEED}.pt\"
```

### Last-layer Laplace Approximation (LLLA)
Use the following commands to train LLLA models for the different architectures / datasets.

Note that `SEED`$\in \{1,2,\dots,30\}$.

We tune the precision of a zero-mean isotropic Gaussian prior based on the validation expected log predictive density (ELPD). The precision is tuned using a grid of 21 values evenly spaced in the interval $[-4, 4]$ in log space. This grid search happens internally during fitting of the LLLA.

#### CIFAR-10 (WRN-16-4 / WRN-28-10)
For the WRN-16-4 architecture:
```
mkdir -p models/cifar10/aug_false/llla

python3 experiments/train.py
    seed=${SEED} \
    data.data_augmentation=false \
    inference=llla \
    inference.init.link_approx=mc \
    inference.init.pred_type=nn \
    inference.fit.prior_fit_method=CV \
    pretrained_model=\"models/cifar10/aug_true/map/wrn-16-4_seed=${SEED}.pt\"
```

For the WRN-28-10 architecture:
```
mkdir -p models/cifar10/aug_false/llla

python3 experiments/train.py
    seed=${SEED} \
    data.data_augmentation=false \
    model.backbone.depth=28 \
    model.backbone.widen_factor=10 \
    model.head.input_dim=640 \
    inference=llla \
    inference.init.hessian_structure=diag \
    inference.init.link_approx=mc \
    inference.init.pred_type=nn \
    inference.fit.prior_fit_method=CV \
    pretrained_model=\"models/cifar10/aug_true/map/wrn-28-10_seed=${SEED}.pt\"
```

#### CIFAR-100 (WRN-16-4)
```
mkdir -p models/cifar100/aug_false/llla

python3 experiments/train.py
    seed=${SEED} \
    data=cifar100 \
    data.data_augmentation=false \
    model.head.num_classes=100 \
    inference=llla \
    inference.init.link_approx=mc \
    inference.init.pred_type=nn \
    inference.fit.prior_fit_method=CV \
    pretrained_model=\"models/cifar100/aug_true/map/wrn-16-4_seed=${SEED}.pt\"
```

#### SST-2
```
mkdir -p models/sst2/aug_false/llla

python3 experiments/train.py \
    seed=${SEED} \
    data=SST2 \
    data.batch_size_train=64 \
    data.batch_size_inference=64 \
    model=bert \
    model.head.num_classes=2 \
    inference=llla \
    inference.init.link_approx=mc \
    inference.init.pred_type=nn \
    inference.fit.prior_fit_method=CV \
    pretrained_model=\"models/sst2/aug_false/map/bert-256-4_seed=${SEED}.pt\"
```

#### QM9
```
mkdir -p models/qm9/aug_false/llla

python3 experiments/train.py \
    seed=${SEED} \
    data=qm9 \
    model=painn \
    inference=llla \
    inference.init.likelihood=regression \
    inference.init.link_approx=mc \
    inference.init.pred_type=nn \
    pretrained_model=\"models/qm9/aug_false/map/painn_seed=${SEED}.pt\"
```

### LLLA refined by normalizing flows (LA-NF)
Use the following commands to train LA-NF models for the different architectures / datasets.

Note that `SEED`$\in \{1,2,\dots,30\}$ and `NT`$\in\{1, 5, 10, 30\}$, where `NT` is the number of transforms, i.e., the number of normalizing flow layers.

We tune the precision of a zero-mean isotropic Gaussian prior based on the validation ELPD. The precision is tuned using the grid $\{1, 5, 10, 20, 30, 40, 50, 70, 90, 100, 125, 150, 175, 200, 500\}$. This grid search happens internally during fitting of the LA-NF.

#### CIFAR-10 (WRN-16-4 / WRN-28-10)
For the WRN-16-4 architecture:
```
mkdir -p models/cifar10/aug_false/posterior_refined_llla

python3 experiments/train.py
    seed=${SEED} \
    data.data_augmentation=false \
    inference=posterior_refined_llla \
    inference.init.prior_precision=-1 \
    inference.init.num_transforms=${NT} \
    inference.fit.num_epochs=20 \
    model_name=\"wrn-16-4_num-transform=${NT}_seed=${SEED}.pt\" \
    pretrained_model=\"models/cifar10/aug_false/llla/wrn-16-4_seed=${SEED}.pt\"
```

For the WRN-28-10 architecture:
```
mkdir -p models/cifar10/aug_false/posterior_refined_llla

python3 experiments/train.py
    seed=${SEED} \
    data.data_augmentation=false \
    model.backbone.depth=28 \
    model.backbone.widen_factor=10 \
    model.head.input_dim=640 \
    inference=posterior_refined_llla \
    inference.init.prior_precision=-1 \
    inference.init.num_transforms=${NT} \
    inference.fit.num_epochs=20 \
    model_name=\"wrn-28-10_num-transform=${NT}_seed=${SEED}.pt\" \
    pretrained_model=\"models/cifar10/aug_false/llla/wrn-28-10_seed=${SEED}.pt\"
```

#### CIFAR-100 (WRN-16-4)
```
mkdir -p models/cifar100/aug_false/posterior_refined_llla

python3 experiments/train.py
    seed=${SEED} \
    data=cifar100 \
    data.data_augmentation=false \
    model.head.num_classes=100 \
    inference=posterior_refined_llla \
    inference.init.prior_precision=-1 \
    inference.init.num_transforms=${NT} \
    inference.init.n_classes=100 \
    inference.fit.num_epochs=20 \
    model_name=\"wrn-16-4_num-transform=${NT}_seed=${SEED}.pt\" \
    pretrained_model=\"models/cifar100/aug_false/llla/wrn-16-4_seed=${SEED}.pt\"
```

#### SST-2
```
mkdir -p models/sst2/aug_false/posterior_refined_llla

python3 experiments/train.py \
    seed=${SEED} \
    data=SST2 \
    data.batch_size_train=64 \
    data.batch_size_inference=64 \
    model=bert \
    model.head.num_classes=2 \
    inference=posterior_refined_llla \
    inference.init.prior_precision=-1 \
    inference.init.num_transforms=${NT} \
    inference.init.n_classes=2 \
    inference.fit.num_epochs=20 \
    model_name=\"bert-256-4_num-transform=${NT}_seed=${SEED}.pt\" \
    pretrained_model=\"models/sst2/aug_false/llla/bert-256-4_seed=${SEED}.pt\"
```

#### QM9
```
mkdir -p models/qm9/aug_false/posterior_refined_llla

python3 experiments/train.py \
    seed=${SEED} \
    data=qm9 \
    model=painn \
    inference=posterior_refined_llla \
    inference.init.likelihood=regression \
    inference.init.prior_precision=-1 \
    inference.init.num_transforms=${NT} \
    inference.init.n_classes=null \
    +inference.init.n_features=64 \
    inference.fit.num_epochs=20 \
    model_name=\"painn_num-transform=${NT}_seed=${SEED}.pt\" \
    pretrained_model=\"models/qm9/aug_false/llla/painn_seed=${SEED}.pt\"
```

## Hyperparameter selection
As described above, the hyperparameter selection for LLLA and LA-NF happens internally during model fitting. 

Above, we provided instructions for training SWAG for all combinations of seeds and learning rates. Here we provide instructions for seleting the optimal learning rate (based on the validation ELPD) for each seed for both SWA and SWAG.
To select learning rates for SWAG run
```
mkdir -p tables

python3 experiments/select_swag.py \
    data_aug=false \
    created_after=2024-10-11T00
    swag_val_metrics_dest=tables/swag_${DATA}_${MODEL}_val_metrics.csv \
    selected_swag_dest=tables/selected_swag_${DATA}_${MODEL}_val_metrics.csv \
    create_links=true \
    data=${DATA} \
    model=${MODEL}
```
where `DATA` and `MODEL` are one of the dataset and model combinations trained above, e.g., 'cifar10' and 'wrn-16-4' or 'qm9' and 'painn'. Note that this scripts fetches data from `wandb`, so the `wandb` logger needs to be setup during model training.
The selected SWAG models will be saved in `models/${DATA}/aug_false/selected_swag`.

To select learning rates for SWA run
```
mkdir -p tables

python3 experiments/select_lr.py \
    swag_dir=models/${DATA}/aug_false/swag/ \
    model_name=${MODEL} \
    new_inference_class=src.inference.SWA \
    val_metrics_dest=tables/swa_${DATA}_${MODEL}_val_metrics.csv \
    selected_dest=tables/selected_swa_${DATA}_${MODEL}_val_metrics.csv \
```
where `DATA` and `MODEL` again are one of the dataset and model combinations trained above. The selected SWA models will be saved in `models/${DATA}/aug_false/selected_swa`.

## Evaluation
Here we provide instructions for evaluating ensembles of the models trained above.
We denote the ensemble size with `K`$\in\{1, 2, 5, 10, 20\}$.

We provide separate instructions for evaluating single models (`K`$=1$) and ensembles (`K` > 1).

### `K`$=1$
To evaluate single models from the model and dataset combinations trained above, run:
```
mkdir -p stats/${DATA}$/aug_${AUG}/${INFERENCE}

python3 experiments/test.py
    test_seed=${SEED} \
    compute_test_stats=true \
    compute_shift_stats=false \
    compute_ood_stats=${OOD} \
    num_posterior_samples=200 \
    subsample_sizes=[5,10,15,20,40,60,80,100,120,140,160,180,200] \
    stratified_sampling=false \
    model_prefix=\"models/${DATA}/aug_${AUG}/${INFERENCE}/${MODEL}_seed=\" \
    model_suffix=\".pt\" \
    model_seeds=[${SEED}] \
    stats_dir=\"stats/${DATA}$/aug_${AUG}/${INFERENCE}\" \
    stats_name=\"{MODEL}_seed=${SEED}\" \
```
where: \
`SEED`$\in\{1,2,\dots,30\}$, \
`DATA`$\in\{\textrm{cifar10},\textrm{cifar100}, \textrm{sst2}, \textrm{qm9} \}$, \
`MODEL`$\in\{\textrm{wrn-16-4}, \textrm{wrn-28-10}, \textrm{bert-256-4}, \textrm{painn} \}$, \
`INFERENCE`$\in\{\textrm{map}, \textrm{selected\_swa}, \textrm{selected\_swag}, \textrm{llla}, \textrm{posterior\_refined\_llla} \}$, \
`OOD` is `true` if `DATA`$\in\{\textrm{cifar10},\textrm{cifar100}\}$ and otherwise `false`, \
`AUG` is `true` if `INFERENCE` is `map` and otherwise `false`, \
`num_posterior_samples` and `subsample_sizes` is set to `1` if `INFERENCE`$\in\{\textrm{map}, \textrm{selected\_swa}\}$, \
`_num-transform=${NT}` is appended to `MODEL` for `NT`$\in\{1,5,10,30\}$ if `INFERENCE` is `posterior_refined_llla`.


### `K`$>1$

The file `ensembles.csv` provides for each `K`$>1$, 30 ensemble IDs `ENSEMBLE_ID`$\in\{1,2,\dots,30\}$ where each `ENSEMBLE_ID` has `K` associated `MODEL_ID`s (the `SEED`s used above) and a `ENSEMBLE_SEED`.

To evaluate ensembles from the model and dataset combinations trained above, run:
```
mkdir -p stats/${DATA}$/aug_${AUG}/${INFERENCE}_ensemble

MODEL_IDS=`grep -P "\b${K}\t${ENSEMBLE_ID}\b" ensembles.csv | awk '{print $4}'`
ENSEMBLE_SEED=`grep -P "\b${K}\t${ENSEMBLE_ID}\b" ensembles.csv | awk '{print $3}'`

python3 experiments/test.py
    test_seed=${ENSEMBLE_SEED} \
    compute_test_stats=true \
    compute_shift_stats=false \
    compute_ood_stats=${OOD} \
    num_posterior_samples=200 \
    subsample_sizes=[5,10,15,20,40,60,80,100,120,140,160,180,200] \
    stratified_sampling=true \
    model_prefix=\"models/${DATA}/aug_${AUG}/${INFERENCE}/${MODEL}_seed=\" \
    model_suffix=\".pt\" \
    model_seeds=[${MODEL_IDS}] \
    stats_dir=\"stats/${DATA}$/aug_${AUG}/${INFERENCE}_ensemble\" \
    stats_name=\"{MODEL}_K=${K}_combination=${ENSEMBLE_ID}\"
```
where: \
`K`$\in\{2,5,10,20\}$, \
`ENSEMBLE_ID`$\in\{1,2,\dots,30\}$, \
`DATA`$\in\{\textrm{cifar10},\textrm{cifar100}, \textrm{sst2}, \textrm{qm9} \}$, \
`MODEL`$\in\{\textrm{wrn-16-4}, \textrm{wrn-28-10}, \textrm{bert-256-4}, \textrm{painn} \}$, \
`INFERENCE`$\in\{\textrm{map}, \textrm{selected\_swa}, \textrm{selected\_swag}, \textrm{llla}, \textrm{posterior\_refined\_llla} \}$, \
`OOD` is `true` if `DATA`$\in\{\textrm{cifar10},\textrm{cifar100}\}$ and otherwise `false`, \
`AUG` is `true` if `INFERENCE` is `map` and otherwise `false`, \
`subsample_sizes` and `num_posterior_samples` is set to `K` if `INFERENCE`$\in\{\textrm{map}, \textrm{selected\_swa}\}$, \
`_num-transform=${NT}` is appended to `MODEL` for `NT`$\in\{1,5,10,30\}$ if `INFERENCE` is `posterior_refined_llla`.


## Results
When the models and ensembles has been trained and evaluated as described above, all the material needed to reproduce the main figures and tables in the paper is available.
For example, to generate Table C.3 in Appendix C (all the main results) run:
```
python3 metric_table.py \
    --directory \
    stats/cifar10/aug_true/map \
    stats/cifar10/aug_false/selected_swa \
    stats/cifar10/aug_false/selected_swag \
    stats/cifar10/aug_false/llla \
    stats/cifar10/aug_false/posterior_refined_llla \
    stats/cifar10/aug_true/map_ensemble \
    stats/cifar10/aug_false/selected_swa_ensemble \
    stats/cifar10/aug_false/selected_swag_ensemble \
    stats/cifar10/aug_false/llla_ensemble \
    stats/cifar10/aug_false/posterior_refined_llla_ensemble \
    stats/cifar100/aug_true/map \
    stats/cifar100/aug_false/selected_swa \
    stats/cifar100/aug_false/selected_swag \
    stats/cifar100/aug_false/llla \
    stats/cifar100/aug_false/posterior_refined_llla \
    stats/cifar100/aug_true/map_ensemble \
    stats/cifar100/aug_false/selected_swa_ensemble \
    stats/cifar100/aug_false/selected_swag_ensemble \
    stats/cifar100/aug_false/llla_ensemble \
    stats/cifar100/aug_false/posterior_refined_llla_ensemble \
    stats/qm9/aug_false/map \
    stats/qm9/aug_false/selected_swa \
    stats/qm9/aug_false/selected_swag \
    stats/qm9/aug_false/llla \
    stats/qm9/aug_false/posterior_refined_llla \
    stats/qm9/aug_false/map_ensemble \
    stats/qm9/aug_false/selected_swa_ensemble \
    stats/qm9/aug_false/selected_swag_ensemble \
    stats/qm9/aug_false/llla_ensemble \
    stats/qm9/aug_false/posterior_refined_llla_ensemble \
    stats/sst2/aug_false/map \
    stats/sst2/aug_false/selected_swa \
    stats/sst2/aug_false/selected_swag \
    stats/sst2/aug_false/llla \
    stats/sst2/aug_false/posterior_refined_llla \
    stats/sst2/aug_false/map_ensemble \
    stats/sst2/aug_false/selected_swa_ensemble \
    stats/sst2/aug_false/selected_swag_ensemble \
    stats/sst2/aug_false/llla_ensemble \
    stats/sst2/aug_false/posterior_refined_llla_ensemble \
    --table_dest metric_table.tex
```
