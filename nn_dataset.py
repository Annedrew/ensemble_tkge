# Build the training and dataset for nn
import json
# need to choose the interpreter in conda by hand.
import ijson
import csv
import pandas as pd

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
                if model_name[0] in obj["RANK"]:
                    value = obj["RANK"][model_name[0]]
                    top_5 = [top[1] for top in self.get_top_5((json.loads(value)))]
                    simu_score_dict[model_name[0]] = top_5
                if model_name[1] in obj["RANK"]:
                    value = obj["RANK"][model_name[1]]
                    top_5 = [top[1] for top in self.get_top_5((json.loads(value)))]
                    simu_score_dict[model_name[1]] = top_5
                if model_name[2] in obj["RANK"]:
                    value = obj["RANK"][model_name[2]]
                    top_5 = [top[1] for top in self.get_top_5((json.loads(value)))]
                    simu_score_dict[model_name[2]] = top_5
                if model_name[3] in obj["RANK"]:
                    value = obj["RANK"][model_name[3]]
                    top_5 = [top[1] for top in self.get_top_5((json.loads(value)))]
                    simu_score_dict[model_name[3]] = top_5
                if model_name[4] in obj["RANK"]:
                    value = obj["RANK"][model_name[4]]
                    top_5 = [top[1] for top in self.get_top_5((json.loads(value)))]
                    simu_score_dict[model_name[4]] = top_5
                    
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
                if model_name[1] in obj["RANK"]:
                    value = obj["RANK"][model_name[1]]
                    top_5 = [top[0] for top in self.get_top_5((json.loads(value)))]
                    simu_id_dict[model_name[1]] = top_5
                if model_name[2] in obj["RANK"]:
                    value = obj["RANK"][model_name[2]]
                    top_5 = [top[0] for top in self.get_top_5((json.loads(value)))]
                    simu_id_dict[model_name[2]] = top_5
                if model_name[3] in obj["RANK"]:
                    value = obj["RANK"][model_name[3]]
                    top_5 = [top[0] for top in self.get_top_5((json.loads(value)))]
                    simu_id_dict[model_name[3]] = top_5
                if model_name[4] in obj["RANK"]:
                    value = obj["RANK"][model_name[4]]
                    top_5 = [top[0] for top in self.get_top_5((json.loads(value)))]
                    simu_id_dict[model_name[4]] = top_5

                row = []
                for name in model_name:
                    row += simu_id_dict[name]
                rows.append(row)

        return rows


    def get_target(self, file_path, model_name):
        # Get the score of correct answer from simulated score file.
        targets = {}
        for name in model_name:
            targets[name] = []
        rows = []

        with open(file_path, "r") as f:
            objects = ijson.items(f, "item")
            i = 0
            for obj in objects:
                i += 1
                if i % 100 == 0:
                    print(f"Fetching the target of query: {str(i)}")
                if model_name[0] in obj["RANK"]:
                    value = json.loads(obj["RANK"][model_name[0]])[0]
                    targets[model_name[0]] = value
                if model_name[1] in obj["RANK"]:
                    value = json.loads(obj["RANK"][model_name[1]])[0]
                    targets[model_name[1]] = value
                if model_name[2] in obj["RANK"]:
                    value = json.loads(obj["RANK"][model_name[2]])[0]
                    targets[model_name[2]] = value
                if model_name[3] in obj["RANK"]:
                    value = json.loads(obj["RANK"][model_name[3]])[0]
                    targets[model_name[3]] = value
                if model_name[4] in obj["RANK"]:
                    value = json.loads(obj["RANK"][model_name[4]])[0]
                    targets[model_name[4]] = value

                row = []
                for name in model_name:
                    row.append(targets[name])
                rows.append(row)
                
            return rows


    def save_csv(self, file_name, rows, model_name, id_score, input_target):
        row_name = []
        if input_target == "Input":
            for name in model_name:
                for i in range(int(len(rows[0])/len(model_name))):
                    row_name.append(f"{name}_{i+1}")
        if input_target == "Target":
            for name in model_name:
                for i in range(int(len(rows[0])/len(model_name))):
                    row_name.append(f"Target_{name}_{i+1}")

        if id_score == "ID":
            with open(file_name, "w") as f:
                writer = csv.writer(f)
                writer.writerow(row_name)
                for row in rows:
                    writer.writerow(row)
        if id_score == "Score":
            with open(file_name, "w") as f:
                writer = csv.writer(f)
                writer.writerow(row_name)
                for row in rows:
                    writer.writerow(row)

    def concatenate_csv(self, input_file, target_file, dataset_name):
        inputs = pd.read_csv(input_file)
        targets = pd.read_csv(target_file)
        dataset = pd.concat([inputs, targets], axis=1)
        dataset.to_csv(dataset_name, index=False)


    # TODO: Min-Max Normalization
    def normalize(self, fila_path):
        pass
            

if __name__ == "__main__":
    model_name = ["DE_TransE", "DE_SimplE", "DE_DistMult", "TERO", "ATISE"]

    dataset = NNDataset()
    id_score = ["ID", "Score"]
    input_target = ["Input", "Target"]
    # dataset.concatenate_csv("new_results/ens_train_top_5_score.csv", "new_results/ens_train_target.csv", "new_results/ens_train.csv")
    dataset.concatenate_csv("new_results/ens_test_top_5_score.csv", "new_results/ens_test_target.csv", "new_results/ens_test.csv")

    # Test code
    # target = dataset.get_target("new_results/temp_sim_scores.json", model_name)
    # dataset.save_csv("new_results/temp_target.csv", target, model_name, id_score[0], input_target[1])

    # Get the target for training
    # target = dataset.get_target("new_results/query_ens_train_sim_scores.json", model_name)
    # dataset.save_csv("new_results/ens_train_target.csv", target, model_name, id_score[0], input_target[1])

    # Get the target for testing
    # target = dataset.get_target("new_results/query_ens_test_sim_scores.json", model_name)
    # dataset.save_csv("new_results/ens_test_target.csv", target, model_name, id_score[0], input_target[1])

    # Test code
    # simu_score = dataset.get_input("new_results/temp_sim_scores.json", model_name)
    # dataset.save_csv("new_results/temp_top_5_score.csv", simu_score, model_name, id_score[1])
    # simu_id = dataset.get_input_id("new_results/temp_sim_scores.json", model_name)
    # dataset.save_csv("new_results/temp_top_5_id.csv", simu_id, model_name, id_score[0])

    # Get the input for training 
    # simu_score = dataset.get_input("new_results/query_ens_train_sim_scores.json", model_name)
    # dataset.save_csv("new_results/ens_train_top_5_score.csv", simu_score, model_name, id_score[1])
    # simu_id = dataset.get_input_id("new_results/query_ens_train_sim_scores.json", model_name)
    # dataset.save_csv("new_results/ens_train_top_5_id.csv", simu_id, model_name, id_score[0])

    # Get the input for testing
    # simu_score = dataset.get_input("new_results/query_ens_test_sim_scores.json", model_name)
    # dataset.save_csv("new_results/ens_test_top_5_score.csv", simu_score, model_name, id_score[1])
    # simu_id = dataset.get_input_id("new_results/query_ens_test_sim_scores.json", model_name)
    # dataset.save_csv("new_results/ens_test_top_5_id.csv", simu_id, model_name, id_score[0])