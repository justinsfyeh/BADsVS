## Combining Machine Learning and Quantum Chemistry to Discover New Energetic Materials in Bis-Azole Derivatives
This repository contains the data and model for the virtual screening framework of bis-azole derivatives.

The manuscript of this repository is in preparation.

<!-- ![alt text](assets/Graphical_Abstract.png) -->
<img src="assets/Graphical_Abstract.png" style="width: 100%; height: auto;">

## OS Requirements
This repository requires to operate on **Linux** operating system.

## Python Dependencies
* Python (version >= 3.8)
* chemprop  (version >= 1.3.1)
* rdkit (version >= 2020.03.6)
* scikit-learn (version >= 1.0.2)
* torch (versioin >= 1.12.0)
* matplotlib (version >=3.7.4)
* numpy (version >= 1.24.4)
* pandas (version >= 1.1.3)


## Installation
```bash
conda env create -f environment.yml
```

## Datasets
This study explores various featurization methods and ML models, including the D-MPNN model implemented using the chemprop package.

For the Chemprop model (D-MPNN), the randomly split datasets for evaluation can be accessed here:
[`D-MPNN_train.csv`](https://github.com/justinsfyeh/BADsVS/blob/main/dataset/train/D-MPNN_train.csv), [`D-MPNN_test.csv`](https://github.com/justinsfyeh/BADsVS/blob/main/dataset/test/D-MPNN_test.csv).

For the regression models (RR, SVR, KRR), three different featurization methods have been implemented and evaluated. The datasets containing the pre-generated features, ready for training and testing, can be found below:
* Dressed Atoms (DA) model: [`DA_train.csv`](https://github.com/justinsfyeh/BADsVS/blob/main/dataset/train/DA_train.csv), [`DA_test.csv`](https://github.com/justinsfyeh/BADsVS/blob/main/dataset/test/DA_test.csv).
* Sum-over-Bonds (SoB) model: [`SoB_train.csv`](https://github.com/justinsfyeh/BADsVS/blob/main/dataset/train/SoB_train.csv), [`SoB_test.csv`](https://github.com/justinsfyeh/BADsVS/blob/main/dataset/test/SoB_test.csv).
* Bag-of-Bonds (BoB) model: [`BoB_train.csv`], [`BoB_test.csv`]

Additionally, the training and testing sets, as well as the model checkpoints for the BoB model, and the optimized molecule structure files (.xyz) used for feature generation, can be downloaded using gdown:

```bash
gdown https://drive.google.com/drive/folders/1ln6K561FxiqqGZ2n-LmJaAnrJtuleE-U?usp=drive_link --folder
```
## Training Chemprop models for heat of formation predictions.
1. Atomic fingerprint model
```
python chemprop_atom_fp/train.py \
    --data_path dataset/train/D-MPNN_train.csv \
    --separate_test_path dataset/test/D-MPNN_test.csv \
    --extra_metrics mae \
    --dataset_type regression \
    --save_dir saves/atom_fp \
    --warmup_epochs 2 --max_lr 0.007606892138090938 --init_lr 0.00015213784276181874 \
    --epochs 50 --final_lr 1.5213784276181875e-05 --no_features_scaling \
    --dropout 0.05 --hidden_size 500 --ffn_num_layers 1 \
    --save_preds --fp_method atomic --activation PReLU \
    --batch_size 64 \
    --aggregation sum
```
2. Molecular fingerprint model
python chemprop_atom_fp/train.py \
    --data_path dataset/train/D-MPNN_train.csv \
    --separate_test_path dataset/test/D-MPNN_test.csv \
    --extra_metrics mae \
    --dataset_type regression \
    --save_dir saves/mol_fp \
    --warmup_epochs 2 --max_lr 0.003327655204331969 --init_lr 6.655310408663938e-05 \
    --epochs 1 --final_lr 6.655310408663938e-06 --no_features_scaling \
    --dropout 0 --hidden_size 400 --ffn_num_layers 1 \
    --save_preds --fp_method molecular --activation PReLU \
    --batch_size 64 \
    --aggregation sum
```