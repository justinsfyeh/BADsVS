import numpy as np
from rdkit import Chem
import torch
import torch.nn as nn
import os
from chemprop.visualization import MolToImage
import pandas as pd

from chemprop.args import PredictArgs
from chemprop.models import MoleculeModel
from chemprop.train import load_model

# export PYTHONPATH="$PYTHONPATH:/home/lungyi/chemprop"
# python visualize_atomic_contribution.py --test_path ~/chemprop/heat_formation_data/visulaize_moles/compare.txt \
#     --preds_path ~/chemprop/chemprop/visualization/figures \
#     --checkpoint_path ~/chemprop/save_models/feature_4/transfer_CCSD_exp_atomic_size_split_12/fold_0/model_0/model.pt --no_features_scaling

if __name__ == "__main__":
    parser = PredictArgs()
    args = parser.parse_args()
    args, train_args, models, scalers, num_tasks, task_names = load_model(args, generator=True)
    os.makedirs(args.preds_path, exist_ok=True)
    os.makedirs(os.path.join(args.preds_path, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(args.preds_path, 'embeddings'), exist_ok=True)
    f = open(os.path.join(args.preds_path, 'smiles.txt'), 'w')

    model = next(models)
    model.eval()
    print(args.fp_method)
    # load smiles
    data = pd.read_csv(args.test_path)
    smiles_list = [ [s.strip('\n')] for s in data['smiles'] ]

    #smiles_list = [['CCCCCCC']]
    for i, smiles in enumerate(smiles_list):
        subdf = pd.DataFrame()
        mol = Chem.MolFromSmiles(smiles[0])
        num_atoms = mol.GetNumAtoms()

        output = model([smiles], sum_contribution = False)
        contributions = output[0,0:num_atoms,0]
        subdf['atomic contribution'] = contributions.cpu().detach()
        # extract latent feature
        atomic_fingerprints = model.fingerprint([smiles])
        # print(atomic_fingerprints.shape)
        # print(atomic_fingerprints)
        # print(atomic_fingerprints[0,0:num_atoms,:])

        embedding = atomic_fingerprints[0,0:num_atoms,:].cpu().detach().numpy()
        # print(embedding.shape)

        subdf = pd.concat([subdf, pd.DataFrame(embedding)],axis = 1 )

        for j, atom in enumerate(mol.GetAtoms()):
            atomic_contribution = contributions[j]
            atom.SetProp('atomNote','{:.2f}'.format(atomic_contribution))
            atom.SetAtomMapNum(j+1)
        
        new_smiles = Chem.MolToSmiles(mol)
        print('molecule', i+1,':', new_smiles)
        print(float(sum(contributions)))

        try:
            subdf.to_csv(os.path.join(os.path.join(args.preds_path, 'embeddings'), '{}.csv'.format(i+1)))
            f.write(new_smiles +'\n')
            # img = MolToImage(mol)
            # img.save(os.path.join(os.path.join(args.preds_path, 'figures'), '{}.png'.format(i+1)))
        except:
            pass
    f.close()