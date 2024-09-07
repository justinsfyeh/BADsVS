#!/bin/bash
#PBS -l select=1:vnode=c8-eno1:ncpus=8:mpiprocs=8

source ~/.bashrc
export PYTHONPATH="$PYTHONPATH:/home/lungyi/chemprop"
cd /home/lungyi/chemprop/chemprop/visualization

conda activate chemprop

python visualize_atomic_contribution.py --test_path /home/lungyi/chemprop/heat_formation_data/visulaize_moles/some_test.txt \
    --preds_path /home/lungyi/chemprop/chemprop/visualization/figures \
    --checkpoint_path /home/lungyi/chemprop/save_models/feature_4/transfer_CCSD_exp_atomic_size_split_12/fold_0/model_0/model.pt \
    --no_features_scaling

conda deactivate
