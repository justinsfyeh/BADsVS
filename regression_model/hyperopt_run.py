import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from hyperopt import fmin, tpe
import pickle
import pyfiglet
import os
import json
from models import get_model, RR_SPACE, BRR_SPACE, KRR_SPACE, SVR_SPACE
from utils import makedirs, plot_results

def run_hyperopt(args, X_train, X_test, y_train, y_test):
    """
    Run hyperparameter optimization using Hyperopt.
    :param args: Arguments containing model architecture, iterations, etc.
    :param X_train: Training features.
    :param X_test: Testing features.
    :param y_train: Training labels.
    :param y_test: Testing labels.
    """
    results = []

    def objective(hyperparams):
        model = get_model(args.model_arch, hyperparams)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        results.append({
            'mae': mae,
            'rmse': mean_squared_error(y_test, preds, squared=False),
            'hyperparams': hyperparams
        })
        return (1 if args.minimize_score else -1) * mae

    # Define search space
    if args.model_arch == 'RR':
        space = RR_SPACE
    elif args.model_arch == 'BRR':
        space = BRR_SPACE
    elif args.model_arch == 'KRR':
        space = KRR_SPACE
    elif args.model_arch == 'SVR':
        space = SVR_SPACE

    # Run hyperparameter optimization
    fmin(objective, space, algo=tpe.suggest, max_evals=args.num_iters)
    
    print(pyfiglet.figlet_format('Hyperopt Finished'))
    best_result = min(results, key=lambda r: (1 if args.minimize_score else -1) * r['mae'])

    # Train the best model
    final_model = get_model(args.model_arch, best_result['hyperparams'])
    final_model.fit(X_train, y_train)
    pickle.dump(final_model, open(os.path.join(args.save_dir, 'models', f'{args.feat}_{args.model_arch}.sav'), 'wb'))

    # Evaluate and plot results
    preds = final_model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    plot_results(y_test, preds, mae, rmse, args)
