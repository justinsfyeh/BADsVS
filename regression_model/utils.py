import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def makedirs(path: str, isfile: bool = False) -> None:
    """
    Creates a directory or the parent directory for a file.
    :param path: Path to a directory or file.
    :param isfile: Whether the provided path is a directory or file.
    """
    if isfile:
        path = os.path.dirname(path)
    if path != '':
        os.makedirs(path, exist_ok=True)

def separate_data(args):
    """
    Loads and separates the training and testing data from CSV files.
    :param args: Arguments that contain data directory and column information.
    :return: X_train, X_test, y_train, y_test
    """
    # Load training data from CSV
    df_train = pd.read_csv(os.path.join(args.train_dir, f'{args.feat}_train.csv'))
    X_train = df_train[df_train.columns[args.first_columns_to_exclude:]]
    y_train = df_train[args.target_column]

    # Load testing data from CSV
    df_test = pd.read_csv(os.path.join(args.test_dir, f'{args.feat}_test.csv'))
    X_test = df_test[df_test.columns[args.first_columns_to_exclude:]]
    y_test = df_test[args.target_column]

    return X_train, X_test, y_train, y_test


def plot_results(y_test, preds, mae, rmse, args):
    """
    Plot actual vs predicted values and save the plot to a file.
    
    :param y_test: Ground truth values.
    :param preds: Predicted values from the model.
    :param mae: Mean absolute error.
    :param rmse: Root mean square error.
    :param args: Arguments for saving paths.
    """
    # Determine the range for the plot
    min_point = min(min(y_test), min(preds))
    max_point = max(max(y_test), max(preds))
    points = np.linspace(min_point, max_point, 20)

    # Create the plot
    plt.figure(figsize=(4.5, 4.5), dpi=800)
    plt.plot(points, points, color='black', linestyle='--')  # Reference line
    sns.scatterplot(x=y_test, y=preds, palette=sns.color_palette("hls", len(y_test)))  # Scatter plot

    # Set plot labels
    plt.xlabel('Reference heat of formation (kcal/mol)', fontsize=14)
    plt.ylabel('Predicted heat of formation (kcal/mol)', fontsize=14)
    
    # Display performance metrics (MAE and RMSE) on the plot
    plt.text(30, -280, f"RMSE: {rmse:.2f} kcal/mol\nMAE: {mae:.2f} kcal/mol", fontsize=11)

    # Save the plot as an image
    plot_save_path = os.path.join(args.save_dir, 'plots', f'{args.feat}_{args.model_arch}.jpg')
    plt.savefig(plot_save_path, format="jpg", dpi=300, bbox_inches='tight')
    plt.close()
