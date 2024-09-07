import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

import torch
import torch.nn as nn
import os
from chemprop.visualization import MolToImage

from chemprop.args import PredictArgs
from chemprop.models import MoleculeModel
from chemprop.train import load_model

# export PYTHONPATH="$PYTHONPATH:/home/lungyi/chemprop"
# python visualize_atomic_contribution.py --test_path ~/chemprop/heat_formation_data/visulaize_moles/compare.txt \
#     --preds_path ~/chemprop/chemprop/visualization/figures_updated_2nd_review \
#     --checkpoint_path /home/lungyi/chemprop/save_models/feature_4/multi/multi_afp_PReLU_transfer_newLR_gpu/afp_6/fold_0/model_0/model.pt --no_features_scaling

if __name__ == "__main__":
    parser = PredictArgs()
    args = parser.parse_args()
    args, train_args, models, scalers, num_tasks, task_names = load_model(args, generator=True)
    os.makedirs(args.preds_path, exist_ok=True)
    
    model = next(models)
    model.eval()
    print(args.fp_method)
    # load smiles
    f = open(args.test_path)
    smiles_list = f.readlines()
    f.close()
    smiles_list = [ [s.strip('\n')] for s in smiles_list ]

    #smiles_list = [['CCCCCCC']]
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles[0])
        output = model([smiles], sum_contribution = False)
        contributions = output[0,:,0]
        
        print('molecule', i+1,':',smiles[0])
        print(float(sum(contributions)))
        
        AllChem.EmbedMolecule(mol)
        AllChem.UFFOptimizeMolecule(mol)
        mol.GetConformer()

        with open(os.path.join(args.preds_path,'{}_{}.txt'.format(i+1, Chem.MolToSmiles(mol))), 'w') as file:
            file.write(str(mol.GetNumAtoms())+'\n')
            contributions = list(contributions.detach().numpy())
            file.write(str(contributions)+'\n')
            for j, atom in enumerate(mol.GetAtoms()):
                atomic_contribution = contributions[j]
                atom.SetAtomMapNum(j+1)
                atom.SetProp('atomNote','{:.2f}'.format(atomic_contribution))

                positions = mol.GetConformer().GetAtomPosition(j)

                line = [str(atom.GetSymbol()), str(positions.x), str(positions.y), str(positions.z), str(float(atomic_contribution)), '\n']
                line = '     '.join(line)
                file.write(line)



        img = MolToImage(mol)
        img.save(os.path.join(args.preds_path, './{}_{}.png'.format(i+1, Chem.MolToSmiles(mol))))
