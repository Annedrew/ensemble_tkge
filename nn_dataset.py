# Build the training and dataset for nn
import json
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
            data = json.load(f)
            simu_score_dict = {}
            for name in model_name:
                simu_score_dict[name] = []
                
            rows = []
            for i in range(len(data)):
                for name in model_name:
                    if name == "DE_TransE":
                        top_5 = [top[1] for top in self.get_top_5(json.loads(data[i]["RANK"][name]))]
                        simu_score_dict[name] = top_5
                    elif name == "DE_SimplE":
                        top_5 = [top[1] for top in self.get_top_5(json.loads(data[i]["RANK"][name]))]
                        simu_score_dict[name] = top_5
                    elif name == "DE_DistMult":
                        top_5 = [top[1] for top in self.get_top_5(json.loads(data[i]["RANK"][name]))]
                        simu_score_dict[name] = top_5
                    elif name == "TERO":
                        top_5 = [top[1] for top in self.get_top_5(json.loads(data[i]["RANK"][name]))]
                        simu_score_dict[name] = top_5
                    elif name == "ATISE":
                        top_5 = [top[1] for top in self.get_top_5(json.loads(data[i]["RANK"][name]))]
                        simu_score_dict[name] = top_5
                row = []
                for name in model_name:
                    row += simu_score_dict[name]

                rows.append(row)

        return rows


    def get_input_id(self, file_path, model_name):
        with open(file_path, "r") as f:
            data = json.load(f)
            simu_id_dict = {}
            for name in model_name:
                simu_id_dict[name] = []
                
            rows = []
            for i in range(len(data)):
                for name in model_name:
                    if name == "DE_TransE":
                        top_5 = [top[0] for top in self.get_top_5(json.loads(data[i]["RANK"][name]))]
                        simu_id_dict[name] = top_5
                    elif name == "DE_SimplE":
                        top_5 = [top[0] for top in self.get_top_5(json.loads(data[i]["RANK"][name]))]
                        simu_id_dict[name] = top_5
                    elif name == "DE_DistMult":
                        top_5 = [top[0] for top in self.get_top_5(json.loads(data[i]["RANK"][name]))]
                        simu_id_dict[name] = top_5
                    elif name == "TERO":
                        top_5 = [top[0] for top in self.get_top_5(json.loads(data[i]["RANK"][name]))]
                        simu_id_dict[name] = top_5
                    elif name == "ATISE":
                        top_5 = [top[0] for top in self.get_top_5(json.loads(data[i]["RANK"][name]))]
                        simu_id_dict[name] = top_5
                row = []
                for name in model_name:
                    row += simu_id_dict[name]

                rows.append(row)

        return rows
        

        
    def save_csv(self, simu_score, model_name, id_score):
        row_name = []
        for name in model_name:
            for i in range(len(model_name)):
                row_name.append(f"{name}_{i+1}")
        if id_score == "ID":
            with open("top_5_id.csv", "w") as f:
                writer = csv.writer(f)
                writer.writerow(row_name)
                for row in simu_score:
                    writer.writerow(row)
        elif id_score == "Score":
            with open("top_5_score.csv", "w") as f:
                writer = csv.writer(f)
                writer.writerow(row_name)
                for row in simu_score:
                    writer.writerow(row)


    # TODO: Min-Max Normalization
    def normalize(self, fila_path):
        pass
            


if __name__ == "__main__":
    model_name = ["DE_TransE", "DE_SimplE", "DE_DistMult", "TERO", "ATISE"]

    dataset = NNDataset()

    simu_score = dataset.get_input("new_results/temp_sim_scores.json", model_name)
    simu_id = dataset.get_input_id("new_results/temp_sim_scores.json", model_name)
    id_score = ["ID", "Score"]

    # dataset.save_csv(simu_id, model_name, id_score[0])
    # dataset.save_csv(simu_score, model_name, id_score[1])