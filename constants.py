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
# Input: 5*5 (5 = top5 of predicted score) (5 = 5 models)
# Output: 1*5 (1 = correct score) (5 = 5 models)
# INPUT_SIZE = 25
# OUTPUT_SIZE = 5
# HIDDEN_SIZE = 25
# EPOCH = 100
# LEARNING_RATE = 0.001
# BATCH_SIZE = 64

# Input: 5*5 (5 = scores of 5 entities, e.g. the first score is the score of the first replaced entity) (5 = 5 models)
# Output: 1*5 (1 = 1) (5 = 5 entities, if the entity is correct answer, then assign 1, else assign 0)
# INPUT_SIZE = 25
# OUTPUT_SIZE = 5
# HIDDEN_SIZE = 25
# EPOCH = 100
# LEARNING_RATE = 0.001
# BATCH_SIZE = 64

# Input: 4*5 (4 = score of h, r, t, T) (5 = 5 models)
# Output: 1*5 (1 = correct score) (5 = 5 models)
# INPUT_SIZE = 20
# OUTPUT_SIZE = 5
# HIDDEN_SIZE = 20
# EPOCH = 200
# LEARNING_RATE = 0.001
# BATCH_SIZE = 64

# Input: 1*5 (1 = score of h) (5 = 5 models)
# Output: 1*5 (1 = correct score) (5 = 5 models)
# INPUT_SIZE = 5
# OUTPUT_SIZE = 5
# HIDDEN_SIZE = 5
# EPOCH = 400 #500
# LEARNING_RATE = 0.001
# BATCH_SIZE = 64

# Input: 1*5 (1 = score of simulated fact) (5 = 5 models)
# Output: 1*5 (1 = the rank of correct score) (5 = 5 models)
INPUT_SIZE = 5
OUTPUT_SIZE = 5
HIDDEN_SIZE = 5
EPOCH = 100 #500
LEARNING_RATE = 0.001
BATCH_SIZE = 256
