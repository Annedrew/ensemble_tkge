'''
Get the data from individual models
'''

from loader import Loader
from simulated_facts import SimulatedRank, SimulatedScore, Entity_Id, TrueRank
import json
import os

class DataProcess:
    def __init__(self):
         pass
    

    # Get the rank of correct answer
    def load_true_rank(self, file_path, model_name):
        with open(file_path, "r") as f:
            ranked_quads = json.load(f)
            for name in model_name:
                model_path = os.path.join("models", name, "icews14", "Model.model")
                loader = Loader(model_path, name)
                model = loader.load()
                ranker = TrueRank(ranked_quads, model, name)
                #  Get a list of all the scores, not only correct score
                rank = ranker.true_rank()
            simu_scores_path = os.path.join("dataset/true_ranks", f"{file_path.split('/')[-1].split('.')[0]}_true_rank.json")
            with open(simu_scores_path, "w") as f:
                json.dump(rank, f, indent=4)
            
            return simu_scores_path


    # Get all simulated scores
    def load_simu_score(self, file_path, model_name):
        with open(file_path, "r") as f:
            ranked_quads = json.load(f)
            for name in model_name:
                model_path = os.path.join("models", name, "icews14", "Model.model")
                loader = Loader(model_path, name)
                model = loader.load()
                ranker = SimulatedScore(ranked_quads, model, name)
                #  Get a list of all the scores, not only correct score
                rank = ranker.sim_score()
            simu_scores_path = os.path.join("dataset/scores", f"{file_path.split('/')[-1].split('.')[0]}_sim_scores.json")
            with open(simu_scores_path, "w") as f:
                json.dump(rank, f, indent=4)
            
            return simu_scores_path
    

    # Get all simulated ranks
    def load_simu_rank(self, file_path, model_name):
            with open(file_path, "r") as f:
                ranked_quads = json.load(f)
                for name in model_name:
                    model_path = os.path.join("models", name, "icews14", "Model.model")
                    loader = Loader(model_path, name)
                    model = loader.load()
                    ranker = SimulatedRank(ranked_quads, model, name)
                    #  Get a list of all the rankings, not only correct ranking
                    rank = ranker.sim_rank()
            simu_ranks_path = os.path.join("dataset/ranks", f"{file_path.split('/')[-1].split('.')[0]}_sim_ranks.json")
            with open(simu_ranks_path, "w") as f:
                json.dump(rank, f, indent=4)
            
            return simu_ranks_path


    # Get all simulated id
    def load_entity_id(self, model_name, file_path):
        with open(file_path, "r") as f:
            ranked_quads = json.load(f)
            for name in model_name:
                model_path = os.path.join("models", name, "icews14", "Model.model")
                loader = Loader(model_path, name)
                model = loader.load()
                ranker = Entity_Id(ranked_quads, model, name)
                ranker.entity_id()
                


if __name__ == "__main__":
    model_name = ["DE_TransE", "DE_SimplE", "DE_DistMult", "TERO", "ATISE"]
    dataset = DataProcess()
    # The ranks for correct answer
    # dataset.load_true_rank("dataset/queries/query_ens_train.json", model_name)
    dataset.load_true_rank("dataset/queries/query_ens_validation.json", model_name)
    # dataset.load_true_rank("dataset/queries/query_ens_test.json", model_name)

    # Simu scores for validation queries
    # dataset.load_simu_score("dataset/queries/query_ens_validation.json", model_name)

    # Simu scores for training queries
    # dataset.load_simu_score("dataset/queries/query_ens_train.json", model_name)
    # Target for training
    # dataset.load_simu_rank("dataset/queries/query_ens_train.json", model_name)

 
    # Simu scores for test queries
    # dataset.load_simu_score("dataset/queries/query_ens_test.json", model_name)
    # Target for evaluation
    # dataset.load_simu_rank("dataset/queries/query_ens_test.json", model_name)