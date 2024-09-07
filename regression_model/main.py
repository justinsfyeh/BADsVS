from utils import separate_data, makedirs
from hyperopt_run import run_hyperopt
from args import get_config
import pyfiglet

def main():
    args = get_config()

    # Get the training and testing data
    X_train, X_test, y_train, y_test = separate_data(args)

    # Display data
    print(pyfiglet.figlet_format('Training Data'))
    print(X_train, y_train)
    print(pyfiglet.figlet_format('Testing Data'))
    print(X_test, y_test)

    print(pyfiglet.figlet_format('Start Hyperopt'))
    
    # Pass the data to the hyperopt function
    run_hyperopt(args, X_train, X_test, y_train, y_test)

if __name__ == '__main__':
    main()
