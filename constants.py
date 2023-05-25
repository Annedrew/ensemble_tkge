# Constants for grid search
WEIGHT_RANGE = {
        'DE_TransE':  [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'DE_SimplE': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'DE_DistMult': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'TERO': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'ATISE': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    }

# Constants for bayesian optimization
BOUNDS = {'w1': (0.1, 0.5), 'w2': (0.5, 0.8), 'w3': (0.8, 1.0), 'w4': (0.8, 1.0), 'w5': (0.8, 1.0)}

# Constants for neural network
INPUT_SIZE = 20 # 4 * 5
OUTPUT_SIZE = 5 # 1 * 5
# TODO: The number of neurons of hidden layer can be different, this could implement as a flexible configuration
HIDDEN_SIZE = 25
EPOCH = 100#500
LEARNING_RATE = 0.01
BATCH_SIZE = 32