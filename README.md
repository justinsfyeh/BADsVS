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

The randomly split datasets for evaluating the chemprop model can be accessed here:
[`D-MPNN_train.csv`](https://github.com/justinsfyeh/BADsVS/blob/main/dataset/train/D-MPNN_train.csv), [`D-MPNN_test.csv`](https://github.com/justinsfyeh/BADsVS/blob/main/dataset/test/D-MPNN_test.csv).

For the regression models (RR, SVR, KRR), three different featurization methods have been implemented and evaluated. The datasets containing the pre-generated features, ready for training and testing, can be found below:
* Dressed Atoms (DA) model: [`DA_train.csv`](https://github.com/justinsfyeh/BADsVS/blob/main/dataset/train/DA_train.csv), [`DA_test.csv`](https://github.com/justinsfyeh/BADsVS/blob/main/dataset/test/DA_test.csv).
* Sum-over-Bonds (SoB) model: [`SoB_train.csv`](https://github.com/justinsfyeh/BADsVS/blob/main/dataset/train/SoB_train.csv), [`SoB_test.csv`](https://github.com/justinsfyeh/BADsVS/blob/main/dataset/test/SoB_test.csv).
* Bag-of-Bonds (BoB) model: [`BoB_train.csv`], [`BoB_test.csv`]

Additionally, the training and testing sets, as well as the model checkpoints for the BoB model, and the optimized molecule structure files (.xyz) used for feature generation, can be downloaded using gdown:

```bash
gdown https://drive.google.com/drive/folders/1ln6K561FxiqqGZ2n-LmJaAnrJtuleE-U?usp=drive_link --folder
```

