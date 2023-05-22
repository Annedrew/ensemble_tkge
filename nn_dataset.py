# Build the training and dataset for nn
import json
# need to choose the interpreter in conda by hand.
import ijson
import csv

class NNDataset:
    def __init__(self):
        pass


    def get_top_5(self, simu_list):
        # Ascending order, save the score and corresponding ID
        top_5 = sorted(enumerate(simu_list), key=lambda x: x[1])[:5]

        return top_5
    

    def get_input(self, file_path, model_name):
        with open(file_path, "r") as f:
            # TODO: Change the key of simulated score to "SIMU", for now both scores and ranks are called "RANK"
            simu_score_dict = {}
            for name in model_name:
                simu_score_dict[name] = []
            rows = []

            # The file is too big, use ijson to parse the data incrementally
            objects = ijson.items(f, "item")
            i = 0
            for obj in objects:
                i += 1
                if i % 100 == 0:
                    print(f"Fetching the score of query: {str(i)}")
                if "DE_TransE" in obj["RANK"]:
                    value = obj["RANK"]["DE_TransE"]
                    top_5 = [top[1] for top in self.get_top_5((json.loads(value)))]
                    simu_score_dict["DE_TransE"] = top_5
                if "DE_SimplE" in obj["RANK"]:
                    value = obj["RANK"]["DE_SimplE"]
                    top_5 = [top[1] for top in self.get_top_5((json.loads(value)))]
                    simu_score_dict["DE_SimplE"] = top_5
                if "DE_DistMult" in obj["RANK"]:
                    value = obj["RANK"]["DE_DistMult"]
                    top_5 = [top[1] for top in self.get_top_5((json.loads(value)))]
                    simu_score_dict["DE_DistMult"] = top_5
                if "TERO" in obj["RANK"]:
                    value = obj["RANK"]["TERO"]
                    top_5 = [top[1] for top in self.get_top_5((json.loads(value)))]
                    simu_score_dict["TERO"] = top_5
                if "ATISE" in obj["RANK"]:
                    value = obj["RANK"]["ATISE"]
                    top_5 = [top[1] for top in self.get_top_5((json.loads(value)))]
                    simu_score_dict["ATISE"] = top_5
                    
                row = []
                for name in model_name:
                    row += simu_score_dict[name]
                rows.append(row)

        return rows


    def get_input_id(self, file_path, model_name):
        with open(file_path, "r") as f:
            simu_id_dict = {}
            for name in model_name:
                simu_id_dict[name] = []
            rows = []

            objects = ijson.items(f, "item")
            i = 0
            for obj in objects:
                i += 1
                if i % 100 == 0:
                    print(f"Fetching the id of query: {str(i)}")
                if "DE_TransE" in obj["RANK"]:
                    value = obj["RANK"]["DE_TransE"]
                    top_5 = [top[0] for top in self.get_top_5((json.loads(value)))]
                    simu_id_dict["DE_TransE"] = top_5
                if "DE_SimplE" in obj["RANK"]:
                    value = obj["RANK"]["DE_SimplE"]
                    top_5 = [top[0] for top in self.get_top_5((json.loads(value)))]
                    simu_id_dict["DE_SimplE"] = top_5
                if "DE_DistMult" in obj["RANK"]:
                    value = obj["RANK"]["DE_DistMult"]
                    top_5 = [top[0] for top in self.get_top_5((json.loads(value)))]
                    simu_id_dict["DE_DistMult"] = top_5
                if "TERO" in obj["RANK"]:
                    value = obj["RANK"]["TERO"]
                    top_5 = [top[0] for top in self.get_top_5((json.loads(value)))]
                    simu_id_dict["TERO"] = top_5
                if "ATISE" in obj["RANK"]:
                    value = obj["RANK"]["ATISE"]
                    top_5 = [top[0] for top in self.get_top_5((json.loads(value)))]
                    simu_id_dict["ATISE"] = top_5

                row = []
                for name in model_name:
                    row += simu_id_dict[name]
                rows.append(row)

        return rows

        
    def save_csv(self, file_name, rows, model_name, id_score):
        row_name = []
        for name in model_name:
            for i in range(len(model_name)):
                row_name.append(f"{name}_{i+1}")
        if id_score == "ID":
            with open(file_name, "w") as f:
                writer = csv.writer(f)
                writer.writerow(row_name)
                for row in rows:
                    writer.writerow(row)
        elif id_score == "Score":
            with open(file_name, "w") as f:
                writer = csv.writer(f)
                writer.writerow(row_name)
                for row in rows:
                    writer.writerow(row)


    # TODO: Min-Max Normalization
    def normalize(self, fila_path):
        pass
            


if __name__ == "__main__":
    model_name = ["DE_TransE", "DE_SimplE", "DE_DistMult", "TERO", "ATISE"]

    dataset = NNDataset()
    id_score = ["ID", "Score"]

    # Test code
    # simu_score = dataset.get_input("new_results/temp_sim_scores.json", model_name)
    # dataset.save_csv("new_results/temp_top_5_score.csv", simu_score, model_name, id_score[1])
    # simu_id = dataset.get_input_id("new_results/temp_sim_scores.json", model_name)
    # dataset.save_csv("new_results/temp_top_5_id.csv", simu_id, model_name, id_score[0])

    # Get the input for training 
    simu_score = dataset.get_input("new_results/query_ens_train_sim_scores.json", model_name)
    dataset.save_csv("new_results/ens_train_top_5_score.csv", simu_score, model_name, id_score[1])
    simu_id = dataset.get_input_id("new_results/query_ens_train_sim_scores.json", model_name)
    dataset.save_csv("new_results/ens_train_top_5_id.csv", simu_id, model_name, id_score[0])

    # Get the input for testing
    # simu_score = dataset.get_input("new_results/query_ens_test_sim_scores.json", model_name)
    # dataset.save_csv("new_results/ens_test_top_5_score.csv", simu_score, model_name, id_score[1])
    # simu_id = dataset.get_input_id("new_results/query_ens_test_sim_scores.json", model_name)
    # dataset.save_csv("new_results/ens_test_top_5_id.csv", simu_id, model_name, id_score[0])