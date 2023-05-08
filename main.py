from grid_search import Grid_search
from bayesian_opt import Bayesian_opt
import time
from arguments import *
from constants import *
from save_result import *
from eval import Eval
from loader import Loader
from ranker import Ranker
import json
import numpy as np
from dataset_split import DatasetSplit

if __name__ == "__main__":
    # Dataset split into 2 json files, only need to run once.
    # data_split = DatasetSplit()
    # data_split.dataset_split()

    args = init_args()
    model_name = ["DE_TransE", "DE_SimplE", "DE_DistMult", "TERO", "ATISE"]
    # Grid Search
    if args.method == "grid":
        init_weight = {
                'DE_TransE': 0.2,
                'DE_SimplE': 0.2,
                'DE_DistMult': 0.2,
                'TERO': 0.2,
                'ATISE': 0.2
            }
        my_grid_search = Grid_search()
        
        # Form the input format of function grid_search_weight()
        ens_train = []
        with open("/ens_dataset/ens_train.json", "r") as f:
            data = json.load(f)
            for query in range(len(data)):
                # Remember to filter out TFLEX model
                rank = [int(item) for item in data[query]["RANK"].values()][:5]
                ens_train.append(rank)
            ens_train = np.array(ens_train)
            ens_train = ens_train.transpose()

        start_time = time.time()
        best_weights = my_grid_search.grid_search_weight(ens_train, list(init_weight.values()), weight_ranges, args.metric)
        print("best weight: ", best_weights)
        time_grid = time.time()
        run_time = round((time_grid - start_time), 2)

        # eval = Eval()
        # eval.eval_indi(ens_test)
        # print(ens_test.shape)
        # print(ens_train.shape)
        # save the array to a text file
        # np.savetxt('ens_test.txt', ens_test, delimiter=',')



        # # eval
        # # quads = "results/icews14/ranked_quads.json"
        # quads = "questions/cor_icews14_test.json"
        # with open(quads, "r") as f:
        #     ranked_quads = json.load(f)
        # # quads = "results/icews14/ranked_quads.json"
        # for name in model_name:
        #     model_path = os.path.join("models", name, "icews14", "Model.model")
        #     loader = Loader(model_path, name)
        #     model = loader.load()

        #     ranker = Ranker(ranked_quads, model, name)
        #     ranked_quads = ranker.rank()
        
        # with open("eval.json", "w") as f:
        #     json.dump(ranked_quads, indent=4)


        # eval = Eval()
        # ens_eval = eval.eval_ens(best_weights, ens_test)
        # print("The evaluation results shown as follows: \n", ens_eval)

    # Bayesian Optimization
    elif args.method == "bayes":
        start_time = time.time()
        my_bayesian_opt = Bayesian_opt()
        best_weights = my_bayesian_opt.bayesian_opt_weight()
        print(f"Best weights: {best_weights}")
        # ensemble_score = my_bayesian_opt.objective_function(best_weights[0], best_weights[1], best_weights[2], best_weights[3], best_weights[4])
        time_bayes = time.time()
        run_time = round((time_bayes - start_time), 2)
    ens_eval = 0
    save_file(best_weights, run_time, ens_eval, args)