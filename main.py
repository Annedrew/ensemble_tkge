
from bayesian_opt import Bayesian_opt
from arguments import *
from constants import *
# from save_result import *
# from loader import Loader
# from simulated_facts import SimulatedRank
from dataset_processing.dataset_split import DatasetSplit
# from cor_generate import QueryGenerate
# import de_simple
import json
import ijson
import numpy as np
import time

from grid_search import Grid_search
from eval import Eval

def grid(args, file_path):
    # Preprocessing data
    ens_train = []
    with open(file_path, "r") as f:
        data = json.load(f)
        for query in range(len(data)):
            # Filter out TFLEX model
            rank = [int(float(item)) for item in data[query]["SIMU"].values()][:5]
            ens_train.append(rank)
        ens_train = np.array(ens_train)
        ens_train = ens_train.transpose()

    init_weight = {
                'DE_TransE': 0.2,
                'DE_SimplE': 0.2,
                'DE_DistMult': 0.2,
                'TERO': 0.2,
                'ATISE': 0.2
            }
    my_grid_search = Grid_search()
    start_time = time.time()
    best_weights = my_grid_search.grid_search_weight(ens_train, list(init_weight.values()), WEIGHT_RANGE, args.metric)
    print("best weight: ", best_weights)
    time_grid = time.time()
    run_time = round((time_grid - start_time), 2)

def grid_eval(best_weigths, file_path, model_name):
    # Preprocessing data
    ens_test = []
    with open(file_path, "r") as f:
        data = json.load(f)
        for query in range(len(data)):
            # Filter out TFLEX model
            rank = [int(float(item)) for item in data[query]["SIMU"].values()][:5]
            ens_test.append(rank)
        ens_test = np.array(ens_test)
        ens_test = ens_test.transpose()
    eval = Eval()
    eval.eval_ens(best_weigths, ens_test, model_name)
    # Save result to txt
    np.savetxt('ens_test.txt', ens_test, delimiter=',')


if __name__ == "__main__":
    model_name = ["DE_TransE", "DE_SimplE", "DE_DistMult", "TERO", "ATISE"]
    args = init_args()

    ## Only run once 
    ## Split the txt file into ens_train and ens_test
    # data_split = DatasetSplit()
    # data_split.dataset_split("dataset/icews14/test.txt")
    ## Generate the corrupted fact
    # query = QueryGenerate()
    # query.query_generate("dataset/ens_icews14/ens_train.txt")
    # query.query_generate("dataset/ens_icews14/ens_test.txt")

    # # Grid Search
    if args.method == "grid":
        grid(args, "dataset/true_ranks/query_ens_train_true_rank.json")
    
    

    


    



    #     # # eval
    #     # # quads = "results/icews14/ranked_quads.json"
    #     # quads = "questions/cor_icews14_test.json"
    #     # with open(quads, "r") as f:
    #     #     ranked_quads = json.load(f)
    #     # # quads = "results/icews14/ranked_quads.json"
    #     # for name in model_name:
    #     #     model_path = os.path.join("models", name, "icews14", "Model.model")
    #     #     loader = Loader(model_path, name)
    #     #     model = loader.load()

    #     #     ranker = SimulatedRank(ranked_quads, model, name)
    #     #     ranked_quads = ranker.sim_rank()
        
    #     # with open("eval.json", "w") as f:
    #     #     json.dump(ranked_quads, indent=4)


    #     # eval = Eval()
    #     # ens_eval = eval.eval_ens(best_weights, ens_test)
    #     # print("The evaluation results shown as follows: \n", ens_eval)

    # # Bayesian Optimization
    elif args.method == "bayes":
        start_time = time.time()
        my_bayesian_opt = Bayesian_opt()
        best_weights = my_bayesian_opt.bayesian_opt_weight()
        print(f"Best weights: {best_weights}")
        # ensemble_score = my_bayesian_opt.objective_function(best_weights[0], best_weights[1], best_weights[2], best_weights[3], best_weights[4])
        time_bayes = time.time()
        run_time = round((time_bayes - start_time), 2)
    # ens_eval = 0
    # save_file(best_weights, run_time, ens_eval, args)