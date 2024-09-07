from hyperopt import hp
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR

# Define hyperparameter spaces
RR_SPACE = {'alpha': hp.loguniform('alpha', low=-8, high=3)}

BRR_SPACE = {
    'alpha_1': hp.loguniform('alpha_1', low=-6, high=2),
    'lambda_1': hp.loguniform('lambda_1', low=-6, high=2),
    'alpha_2': hp.loguniform('alpha_2', low=-6, high=2),
    'lambda_2': hp.loguniform('lambda_2', low=-6, high=2)
}

KRR_SPACE = {
    'alpha': hp.loguniform('alpha', low=-10, high=-1),
    'gamma': hp.loguniform('gamma', low=-10, high=-1)
}

SVR_SPACE = {
    'C': hp.uniform('C', 1, 4000),
    'epsilon': hp.loguniform('epsilon', low=-3, high=2)
}

def get_model(model_arch, hyperparams):
    """
    Get the model based on architecture and hyperparameters.
    :param model_arch: Model architecture string (RR, BRR, KRR, SVR).
    :param hyperparams: Hyperparameters for model initialization.
    """
    if model_arch == 'RR':
        return Ridge(**hyperparams)
    elif model_arch == 'BRR':
        return BayesianRidge(**hyperparams)
    elif model_arch == 'KRR':
        return KernelRidge(kernel='laplacian', **hyperparams)
    elif model_arch == 'SVR':
        return SVR(kernel='rbf', **hyperparams)
