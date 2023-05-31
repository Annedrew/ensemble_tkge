# Build the training and dataset for nn
import json
import ijson # need to choose the interpreter in conda by hand.
import csv
import pandas as pd
import numpy as np

# NNDataset:
# NNDataset_relation:
# NNDataset_min_true: 
# First_10

class NNDataset:
    def __init__(self):
        pass


    def get_top_5(self, simu_list):
        # Ascending order, save the score and corresponding ID
        top_5 = sorted(enumerate(simu_list), key=lambda x: x[1])[:5]

        return top_5
    

    def get_input_score(self, file_path, model_name):
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

    # target: 5, entity: 1
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
                # 这里有问题：len(rows[0])
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


class NNDataset_relation:
    def __init__(self):
        pass


    def get_input_relation(self, file_path, model_name):
        with open(file_path, "r") as f:
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
                if obj["RELATION"] == "0":
                    if model_name[0] in obj["RANK"]:
                        value = json.loads(obj["RANK"][model_name[0]])
                        simu_score_dict[model_name[0]] = value
                    if model_name[1] in obj["RANK"]:
                        value = json.loads(obj["RANK"][model_name[1]])
                        simu_score_dict[model_name[1]] = value
                    if model_name[2] in obj["RANK"]:
                        value = json.loads(obj["RANK"][model_name[2]])
                        simu_score_dict[model_name[2]] = value
                    if model_name[3] in obj["RANK"]:
                        value = json.loads(obj["RANK"][model_name[3]])
                        simu_score_dict[model_name[3]] = value
                    if model_name[4] in obj["RANK"]:
                        value = json.loads(obj["RANK"][model_name[4]])
                        simu_score_dict[model_name[4]] = value
                    
                    row = []
                    for name in model_name:
                        row += simu_score_dict[name]
                    rows.append(row)

        return rows


    def get_target_relation(self, file_path, model_name):
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
                if obj["RELATION"] == "0":
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


    def save_csv(self, file_name, rows, model_name):
        row_name = []
        for name in model_name:
            for i in range(int(len(rows[0])/len(model_name))):
                row_name.append(f"{name}_{i+1}")

        with open(file_name, "w") as f:
            writer = csv.writer(f)
            writer.writerow(row_name)
            for row in rows:
                writer.writerow(row)


class NNDataset_min_true:
    def __init__(self):
        pass


    def get_min(self, simu_list):
        # Ascending order, save the score and corresponding ID
        min = sorted(enumerate(simu_list), key=lambda x: x[1])[:1]

        return min


    def get_input(self, file_path, model_name):
        with open(file_path, "r") as f:
            simu_score_dict = {}
            for name in model_name:
                simu_score_dict[name] = []
            rows = []

            objects = ijson.items(f, "item")
            i = 0
            for obj in objects:
                i += 1
                if i % 100 == 0:
                    print(f"Fetching the score of query: {str(i)}")
                if model_name[0] in obj["RANK"]:
                    value = obj["RANK"][model_name[0]]
                    min = [top[1] for top in self.get_min((json.loads(value)))][0]
                    simu_score_dict[model_name[0]].append(min)
                if model_name[1] in obj["RANK"]:
                    value = obj["RANK"][model_name[1]]
                    min = [top[1] for top in self.get_min((json.loads(value)))][0]
                    simu_score_dict[model_name[1]].append(min)
                if model_name[2] in obj["RANK"]:
                    value = obj["RANK"][model_name[2]]
                    min = [top[1] for top in self.get_min((json.loads(value)))][0]
                    simu_score_dict[model_name[2]].append(min)
                if model_name[3] in obj["RANK"]:
                    value = obj["RANK"][model_name[3]]
                    min = [top[1] for top in self.get_min((json.loads(value)))][0]
                    simu_score_dict[model_name[3]].append(min)
                if model_name[4] in obj["RANK"]:
                    value = obj["RANK"][model_name[4]]
                    min = [top[1] for top in self.get_min((json.loads(value)))][0]
                    simu_score_dict[model_name[4]].append(min)
                
                if i % 4 == 0:
                    row = []
                    for name in model_name:
                        row += simu_score_dict[name]
                        simu_score_dict[name] = []
                    rows.append(row)
            # len(rows) = 2689
            # len(rows[0]) = 20
            # i = 10754

        return rows


    def get_target(self, file_path, model_name):
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
                
                if i % 4 == 0:
                    row = []
                    for name in model_name:
                        row.append(targets[name])
                    rows.append(row)
                
            return rows


    def save_csv(self, file_name, rows, model_name):
        row_name = []
        elements = ["h", "r", "t", "T"]
        for name in model_name:
            for element in elements:
                row_name.append(f"{name}_{element}")

        # for name in model_name:
        #     row_name.append(f"{name}")

        with open(file_name, "w") as f:
            writer = csv.writer(f)
            writer.writerow(row_name)
            for row in rows:
                writer.writerow(row)


class First_10:
    def __init__(self):
        pass

    
    def get_input_first(self, file_path, model_name):
        with open(file_path, "r") as f:
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
                    value = json.loads(obj["RANK"][model_name[0]])[1:6]
                    simu_score_dict[model_name[0]] = value
                if model_name[1] in obj["RANK"]:
                    value = json.loads(obj["RANK"][model_name[1]])[1:6]
                    simu_score_dict[model_name[1]] = value
                if model_name[2] in obj["RANK"]:
                    value = json.loads(obj["RANK"][model_name[2]])[1:6]
                    simu_score_dict[model_name[2]] = value
                if model_name[3] in obj["RANK"]:
                    value = json.loads(obj["RANK"][model_name[3]])[1:6]
                    simu_score_dict[model_name[3]] = value
                if model_name[4] in obj["RANK"]:
                    value = json.loads(obj["RANK"][model_name[4]])[1:6]
                    simu_score_dict[model_name[4]] = value
                    
                    row = []
                    for name in model_name:
                        row += simu_score_dict[name]
                    rows.append(row)

        return rows


    def get_true_id(self, file_path):
        with open(file_path, "r") as f:
            objects = ijson.items(f, "item")
            rows_true = []
            for obj in objects:
                answer = json.loads(obj["RANK"]["DE_TransE"])[0]
                list_simu = json.loads(obj["RANK"]["DE_TransE"])[1:]
                # 答案的ID
                index_true = list_simu.index(answer) + 1
                # 所有答案的ID
                rows_true.append(index_true)
            # 如果是超过5点就不用统计
            rows_true = [0 if x > 5 else x for x in rows_true]

        return rows_true
    

    def get_target(self, file_path, model_name):
        with open(file_path, "r") as f:
            aaa = []
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
                    value = json.loads(obj["RANK"][model_name[0]])[1:6]
                    simu_score_dict[model_name[0]] = value
                if model_name[1] in obj["RANK"]:
                    value = json.loads(obj["RANK"][model_name[1]])[1:6]
                    simu_score_dict[model_name[1]] = value
                if model_name[2] in obj["RANK"]:
                    value = json.loads(obj["RANK"][model_name[2]])[1:6]
                    simu_score_dict[model_name[2]] = value
                if model_name[3] in obj["RANK"]:
                    value = json.loads(obj["RANK"][model_name[3]])[1:6]
                    simu_score_dict[model_name[3]] = value
                if model_name[4] in obj["RANK"]:
                    value = json.loads(obj["RANK"][model_name[4]])[1:6]
                    simu_score_dict[model_name[4]] = value

                # first 5
                aa = []
                for i in range(5):
                    a = sum([simu_score_dict[model_name[j]][i] for j in range(len(model_name))])/len(model_name)
                    aa.append(a)
                sorted_list = sorted(aa)
                ranks = []
                ranks += [sorted_list.index(element) + 1 for element in aa]
                aaa.append(ranks)

            return aaa
        
    def get_target_better(self, rows_true, aaa):
        # rows_true = np.array(rows_true)
        # print(f"rows_true: {rows_true.shape}")

        # aaa = np.array(aaa)
        # print(f"aaa: {aaa.shape}")
        for i in range(len(aaa)):
            if rows_true[i] != 0:
                aaa[i][aaa[i].index(1)], aaa[i][rows_true[i]-1] = aaa[i][rows_true[i]-1], 1
            else:
                pass

        return aaa

    def save_csv(self, file_name, rows):
        row_name = []

        # first 5
        for i in range(5):
            row_name.append(f"element_{i}")

        # for name in model_name:
        #     for i in range(5):
        #         row_name.append(f"{name}_{i+1}")
        with open(file_name, "w") as f:
            writer = csv.writer(f)
            writer.writerow(row_name)
            for row in rows:
                writer.writerow(row)

    def one_hot(self):
        data = pd.read_csv('new_results/ens_train_first_better.csv')
        column_data = data['element_0']
        column_data = column_data.apply(lambda x: 0 if x != 1 else x)
        data['element_0'] = column_data
        data.to_csv('modified_file.csv', index=False)

        column_data = data['element_1']
        column_data = column_data.apply(lambda x: 0 if x != 1 else x)
        data['element_1'] = column_data
        data.to_csv('modified_file.csv', index=False)

        column_data = data['element_2']
        column_data = column_data.apply(lambda x: 0 if x != 1 else x)
        data['element_2'] = column_data
        data.to_csv('modified_file.csv', index=False)

        column_data = data['element_3']
        column_data = column_data.apply(lambda x: 0 if x != 1 else x)
        data['element_3'] = column_data
        data.to_csv('modified_file.csv', index=False)

        column_data = data['element_4']
        column_data = column_data.apply(lambda x: 0 if x != 1 else x)
        data['element_4'] = column_data
        data.to_csv('modified_file.csv', index=False)

# class NormalizeInput:
#     def __init__(self):
#         pass


#     def normalize_input(self, file_name):


if __name__ == "__main__":
    # model_name = ["DE_TransE", "DE_SimplE", "DE_DistMult", "TERO", "ATISE"]
    dataset = First_10()
    dataset.one_hot()
    # rows_true = dataset.get_true_id("new_results/query_ens_train_sim_scores.json")
    # aaa = dataset.get_target("new_results/query_ens_train_sim_scores.json", model_name)
    # rows = dataset.get_target_better(rows_true, aaa)
    # dataset.save_csv("file_name.csv", rows)
                
    # def save_csv(self, file_name, rows, model_name):
    #     row_name = []

    #     # first 5
    #     for i in range(5):
    #         row_name.append(f"element_{i}")

    #     # for name in model_name:
    #     #     for i in range(5):
    #     #         row_name.append(f"{name}_{i+1}")
    #     with open(file_name, "w") as f:
    #         writer = csv.writer(f)
    #         writer.writerow(row_name)
    #         for row in rows:
    #             writer.writerow(row)

# if __name__ == "__main__":
#     model_name = ["DE_TransE", "DE_SimplE", "DE_DistMult", "TERO", "ATISE"]

    # Get the inputs and targets of training dataset
    # dataset = First_10()
    # inputs = dataset.get_input_first("new_results/query_ens_train_sim_scores.json", model_name)
    # dataset.save_csv("new_results/ens_train_input_first.csv", inputs, model_name)
    # targets = dataset.get_target("new_results/query_ens_train_sim_scores.json", model_name)
    # dataset.save_csv("new_results/ens_train_target_first.csv", targets, model_name)
    # # Get the inputs and targets of test dataset
    # inputs = dataset.get_input_first("new_results/query_ens_test_sim_scores.json", model_name)
    # dataset.save_csv("new_results/ens_test_input_first.csv", inputs, model_name)
    # targets = dataset.get_target("new_results/query_ens_test_sim_scores.json", model_name)
    # dataset.save_csv("new_results/ens_test_target_first.csv", targets, model_name)

    # # Get the inputs and targets of training dataset
    # dataset = NNDataset_min_true()
    # inputs = dataset.get_input("new_results/query_ens_train_sim_scores.json", model_name)
    # dataset.save_csv("new_results/ens_train_inputs.csv", inputs, model_name)
    # targets = dataset.get_target("new_results/query_ens_train_sim_scores.json", model_name)
    # dataset.save_csv("new_results/ens_train_targets.csv", targets, model_name)
    # # Get the inputs and targets of test dataset
    # inputs = dataset.get_input("new_results/query_ens_test_sim_scores.json", model_name)
    # dataset.save_csv("new_results/ens_test_inputs.csv", inputs, model_name)
    # targets = dataset.get_target("new_results/query_ens_test_sim_scores.json", model_name)
    # dataset.save_csv("new_results/ens_test_targets.csv", targets, model_name)
    
    # # Get the inputs and targets of training dataset
    # dataset = NNDataset_relation()
    # inputs = dataset.get_input_relation("new_results/query_ens_train_sim_scores.json", model_name)
    # dataset.save_csv("new_results/ens_train_input_relation.csv", inputs, model_name)
    # target = dataset.get_target_relation("new_results/query_ens_train_sim_scores.json", model_name)
    # dataset.save_csv("new_results/ens_train_target_relation.csv", target, model_name)
    # # Get the inputs and targets of test dataset
    # inputs = dataset.get_input_relation("new_results/query_ens_test_sim_scores.json", model_name)
    # dataset.save_csv("new_results/ens_test_input_relation.csv", inputs, model_name)
    # target = dataset.get_target_relation("new_results/query_ens_test_sim_scores.json", model_name)
    # dataset.save_csv("new_results/ens_test_target_relation.csv", target, model_name)


    # # Get the inputs(score and ID) and targets for training 
    # dataset = NNDataset()
    # id_score = ["ID", "Score"]
    # input_target = ["Input", "Target"]
    # simu_score = dataset.get_input_score("new_results/query_ens_train_sim_scores.json", model_name)
    # dataset.save_csv("new_results/ens_train_top_5_score.csv", simu_score, model_name, id_score[1])
    # simu_id = dataset.get_input_id("new_results/query_ens_train_sim_scores.json", model_name)
    # dataset.save_csv("new_results/ens_train_top_5_id.csv", simu_id, model_name, id_score[0])
    # target = dataset.get_target("new_results/query_ens_train_sim_scores.json", model_name)
    # dataset.save_csv("new_results/ens_train_target.csv", target, model_name, id_score[0], input_target[1])
    # # Get the inputs(score and ID) and targets for testing
    # simu_score = dataset.get_input_score("new_results/query_ens_test_sim_scores.json", model_name)
    # dataset.save_csv("new_results/ens_test_top_5_score.csv", simu_score, model_name, id_score[1])
    # simu_id = dataset.get_input_id("new_results/query_ens_test_sim_scores.json", model_name)
    # dataset.save_csv("new_results/ens_test_top_5_id.csv", simu_id, model_name, id_score[0])
    # target = dataset.get_target("new_results/query_ens_test_sim_scores.json", model_name)
    # dataset.save_csv("new_results/ens_test_target.csv", target, model_name, id_score[0], input_target[1])

    # # Concatenate inputs and targets
    # dataset.concatenate_csv("new_results/ens_train_top_5_score.csv", "new_results/ens_train_target.csv", "new_results/ens_train.csv")
    # dataset.concatenate_csv("new_results/ens_test_top_5_score.csv", "new_results/ens_test_target.csv", "new_results/ens_test.csv")

