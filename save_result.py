import os

def save_file(best_weights, run_time, metric, args):
    if args.method == "grid":
        file_name = "grid_results.txt"
        if os.path.exists(file_name):
            with open(file_name, "a") as f:
                run_time = f"Running time: {run_time}s"
                best_weights = f"DE_TransE: {best_weights[0]}, DE_SimplE: {best_weights[1]}, DE_DistMult: {best_weights[2]}, TERO: {best_weights[3]}, ATISE: {best_weights[4]}"
                # ensemble_score = f"HITS@1: {}"
                f.write("\n" + "______RUN______" + "\n")
                f.write(run_time + "\n")
                f.write(best_weights + "\n")
                f.write(f"MRR: {metric}" + "\n")
        else:
            with open(file_name, "w") as f:
                run_time = f"Running time: {run_time}s"
                best_weights = f"DE_TransE: {best_weights[0]}, DE_SimplE: {best_weights[1]}, DE_DistMult: {best_weights[2]}, TERO: {best_weights[3]}, ATISE: {best_weights[4]}"
                # ensemble_score = f"HITS@1: {}"
                f.write("\n" + "______RUN______" + "\n")
                f.write(run_time + "\n")
                f.write(best_weights + "\n")
                f.write(f"MRR: {metric}" + "\n")
    elif args.method == "bayes:":
        file_name = "bayes_results.txt"
        if os.path.exists(file_name):
            with open(file_name, "a") as f:
                run_time = f"Running time: {run_time}s"
                best_weights = f"DE_TransE: {best_weights[0]}, DE_SimplE: {best_weights[1]}, DE_DistMult: {best_weights[2]}, TERO: {best_weights[3]}, ATISE: {best_weights[4]}"
                # ensemble_score = f"HITS@1: {}"
                f.write("\n" + "______RUN______" + "\n")
                f.write(run_time + "\n")
                f.write(best_weights + "\n")
                f.write(f"MRR: {metric}" + "\n")
        else:
            with open(file_name, "w") as f:
                run_time = f"Running time: {run_time}s"
                best_weights = f"DE_TransE: {best_weights[0]}, DE_SimplE: {best_weights[1]}, DE_DistMult: {best_weights[2]}, TERO: {best_weights[3]}, ATISE: {best_weights[4]}"
                # ensemble_score = f"HITS@1: {}"
                f.write("\n" + "______RUN______" + "\n")
                f.write(run_time + "\n")
                f.write(best_weights + "\n")
                f.write(f"MRR: {metric}" + "\n")
    else:
        pass