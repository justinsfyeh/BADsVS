import argparse

def argbuilder():
    """
    Builds and parses command-line arguments for the model training and hyperparameter optimization process.
    
    Returns:
        args (Namespace): Parsed arguments including:
            feat (str): The feature name to be used.
            target_column (str): The column to be predicted in the dataset.
            first_columns_to_exclude (int): Number of initial columns to exclude from the feature set.
            model_arch (str): The architecture of the model to be used (RR, BRR, KRR, SVR).
            num_iters (int): The number of iterations for hyperparameter optimization.
            save_dir (str): Directory where models, results, and configurations are saved.
            minimize_score (bool): Flag to indicate if the score should be minimized (default is True).
            config_save_path (str): Path where the best hyperparameters and results will be saved.
    """
    parser = argparse.ArgumentParser(description="Arguments for model training and hyperparameter optimization.")
    
    parser.add_argument('--feat', type=str, required=True, help='The featurization method (DA, SoB, BoB).')
    parser.add_argument('--train_dir', default='../dataset/train/', type=str, help='Directory for training data.')
    parser.add_argument('--test_dir', default='../dataset/test', type=str, help='Directory for training data.')
    parser.add_argument('--target_column', default='Hf(R)', type=str, help='Target column to be predicted.')
    parser.add_argument('--first_columns_to_exclude', default=8, type=int, help='Number of columns to exclude from the feature set.')
    parser.add_argument('--model_arch', type=str, required=True, help='Model architecture (e.g., RR, KRR, SVR).')
    parser.add_argument('--num_iters', default=50, type=int, help='Number of iterations for hyperparameter optimization.')
    parser.add_argument('--save_dir', default='../',type=str, help='Directory to save models and results.')
    parser.add_argument('--minimize_score', default=True, type=bool, help='Minimize the score (True) or maximize (False).')
    parser.add_argument('--config_save_path', default='../', type=str, help='Path to save hyperparameter configurations.')
    
    return parser.parse_args()

# Optional: You can create a function that just returns the arguments
def get_config():
    """
    Returns the parsed command-line arguments by calling the argbuilder function.
    This function provides an abstraction in case any future configuration logic is needed.
    
    Returns:
        args (Namespace): Parsed command-line arguments.
    """
    return argbuilder()

