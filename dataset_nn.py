'''
Data processing for nn models, get the training, validation, and test dataset.

NNDataset
NNDataset_relation
NNDataset_min_true
Entity_5
NNDataset_min_true2
NNDataset_weights

'''
import json
import ijson # need to choose the interpreter in conda by hand.
import csv
import pandas as pd
import numpy as np
import os


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


class Entity_5:
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


class NNDataset_min_true2:
    def __init__(self):
        pass


    def get_input(self, file_path, model_name):
        with open(file_path, "r") as f:
            simu_score_dict = {}
            for name in model_name:
                simu_score_dict[name] = []
            rows = []
            i = 0

            objects = ijson.items(f, "item")
            for obj in objects:
                i += 1
                if i % 100 == 0:
                    print(f"Fetching the score of query: {str(i)}")
                if obj["HEAD"] == "0":
                    if model_name[0] in obj["SIMU"]:
                        # 除去第一个正确答案
                        value = json.loads(obj["SIMU"][model_name[0]])[1:]
                        # 预测的正确答案的分数是什么
                        pred_answer = sorted(value, reverse=True)[0]
                        simu_score_dict[model_name[0]].append(pred_answer)
                    if model_name[1] in obj["SIMU"]:
                        # 除去第一个正确答案
                        value = json.loads(obj["SIMU"][model_name[1]])[1:]
                        # 预测的正确答案的分数是什么
                        pred_answer = sorted(value, reverse=True)[0]
                        simu_score_dict[model_name[1]].append(pred_answer)
                    if model_name[2] in obj["SIMU"]:
                        # 除去第一个正确答案
                        value = json.loads(obj["SIMU"][model_name[2]])[1:]
                        # 预测的正确答案的分数是什么
                        pred_answer = sorted(value, reverse=True)[0]
                        simu_score_dict[model_name[2]].append(pred_answer)
                    if model_name[3] in obj["SIMU"]:
                        # 除去第一个正确答案
                        value = json.loads(obj["SIMU"][model_name[3]])[1:]
                        # 预测的正确答案的分数是什么
                        pred_answer = sorted(value)[0]
                        simu_score_dict[model_name[3]].append(pred_answer)
                    if model_name[4] in obj["SIMU"]:
                        # 除去第一个正确答案
                        value = json.loads(obj["SIMU"][model_name[4]])[1:]
                        # 预测的正确答案的分数是什么
                        pred_answer = sorted(value)[0]
                        simu_score_dict[model_name[4]].append(pred_answer)
                    
                    row = []
                    for name in model_name:
                        row += simu_score_dict[name]
                        simu_score_dict[name] = []
                    rows.append(row)

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
                if obj["HEAD"] == "0":
                    if model_name[0] in obj["RANK"]:
                        # 正确答案
                        value = json.loads(obj["RANK"][model_name[0]])[0]
                        targets[model_name[0]].append(value)
                    if model_name[1] in obj["RANK"]:
                        # 正确答案
                        value = json.loads(obj["RANK"][model_name[1]])[0]
                        targets[model_name[1]].append(value)
                    if model_name[2] in obj["RANK"]:
                        # 正确答案
                        value = json.loads(obj["RANK"][model_name[2]])[0]
                        targets[model_name[2]].append(value)
                    if model_name[3] in obj["RANK"]:
                        # 正确答案
                        value = json.loads(obj["RANK"][model_name[3]])[0]
                        targets[model_name[3]].append(value)
                    if model_name[4] in obj["RANK"]:
                        # 正确答案
                        value = json.loads(obj["RANK"][model_name[4]])[0]
                        targets[model_name[4]].append(value)

                    row = []
                    for name in model_name:
                        row += targets[name] 
                        targets[name] = []
                    rows.append(row)

            return rows


    def save_csv(self, file_name, rows, model_name):
        row_name = []
        # elements = ["h", "r", "t", "T"]
        # for name in model_name:
        #     for element in elements:
        #         row_name.append(f"{name}_{element}")

        for name in model_name:
            row_name.append(f"{name}")

        with open(file_name, "w") as f:
            writer = csv.writer(f)
            writer.writerow(row_name)
            for row in rows:
                writer.writerow(row)


    #  将数值normalized成为normal distribution
    def normalize_z_score(self, file_path):
        normalized_data = {}
        for name in model_name:
            normalized_data[name] = []

        data = pd.read_csv(file_path)
        for column, name in zip(data.columns, model_name):
            # print(type(column))
            column_values = data[column].values
            mean = np.mean(column_values)
            std_dev = np.std(column_values)
            nor_data = (column_values - mean) / std_dev
            # normalized_data[column] = nor_data
            normalized_data[name] = nor_data
        df = pd.DataFrame(normalized_data)
        df.to_csv('output.csv', index=False)


    # 这个可以解决数值中有负值的数据，将数值缩放到0-1的范围内
    def normalize_min_max(self, file_path):
        normalized_data = {}
        for name in model_name:
            normalized_data[name] = []

        data = pd.read_csv(file_path)
        for column, name in zip(data.columns, model_name):
            column_values = data[column].values
            nor_data = (column_values - data[column].min()) / (data[column].max() - data[column].min())
            normalized_data[name] = nor_data
            # print(normalized_data)
        df = pd.DataFrame(normalized_data)
        df.to_csv('output.csv', index=False)       


class NNDataset_weights:
    def __init__(self):
        pass


    def get_input_5(self, file_path, model_name):
        with open(file_path, "r") as f:
            simu_score_dict = {}
            for name in model_name:
                simu_score_dict[name] = []
            rows = []
            i = 0

            objects = ijson.items(f, "item")
            for obj in objects:
                i += 1
                if i % 100 == 0:
                    print(f"Fetching the score of query: {str(i)}")
                if model_name[0] in obj["SIMU"]:
                    # 除去第一个正确答案
                    value = json.loads(obj["SIMU"][model_name[0]])[1:]
                    # 预测的正确答案的分数是什么
                    pred_answer = sorted(value, reverse=True)[0]
                    simu_score_dict[model_name[0]].append(pred_answer)
                if model_name[1] in obj["SIMU"]:
                    # 除去第一个正确答案
                    value = json.loads(obj["SIMU"][model_name[1]])[1:]
                    # 预测的正确答案的分数是什么
                    pred_answer = sorted(value, reverse=True)[0]
                    simu_score_dict[model_name[1]].append(pred_answer)
                if model_name[2] in obj["SIMU"]:
                    # 除去第一个正确答案
                    value = json.loads(obj["SIMU"][model_name[2]])[1:]
                    # 预测的正确答案的分数是什么
                    pred_answer = sorted(value, reverse=True)[0]
                    simu_score_dict[model_name[2]].append(pred_answer)
                if model_name[3] in obj["SIMU"]:
                    # 除去第一个正确答案
                    value = json.loads(obj["SIMU"][model_name[3]])[1:]
                    # 预测的正确答案的分数是什么
                    pred_answer = sorted(value)[0]
                    simu_score_dict[model_name[3]].append(pred_answer)
                if model_name[4] in obj["SIMU"]:
                    # 除去第一个正确答案
                    value = json.loads(obj["SIMU"][model_name[4]])[1:]
                    # 预测的正确答案的分数是什么
                    pred_answer = sorted(value)[0]
                    simu_score_dict[model_name[4]].append(pred_answer)
                
                row = []
                for name in model_name:
                    row += simu_score_dict[name]
                    simu_score_dict[name] = []
                rows.append(row)

        return rows


    def get_input_25(self, file_path, model_name):
        with open(file_path, "r") as f:
            simu_score_dict = {}
            for name in model_name:
                simu_score_dict[name] = []
            rows = []
            i = 0

            objects = ijson.items(f, "item")
            for obj in objects:
                i += 1
                if i % 100 == 0:
                    print(f"Fetching the score of query: {str(i)}")

                if model_name[0] in obj["SIMU"]:
                    # 除去第一个正确答案
                    value = json.loads(obj["SIMU"][model_name[0]])[1:]
                    # 预测的正确答案前5的分数是什么
                    pred_answer = sorted(value, reverse=True)[0:5]
                    simu_score_dict[model_name[0]] = pred_answer
                if model_name[1] in obj["SIMU"]:
                    # 除去第一个正确答案
                    value = json.loads(obj["SIMU"][model_name[1]])[1:]
                    # 预测的正确答案前5的分数是什么
                    pred_answer = sorted(value, reverse=True)[0:5]
                    simu_score_dict[model_name[1]] = pred_answer
                if model_name[2] in obj["SIMU"]:
                    # 除去第一个正确答案
                    value = json.loads(obj["SIMU"][model_name[2]])[1:]
                    # 预测的正确答案前5的分数是什么
                    pred_answer = sorted(value, reverse=True)[0:5]
                    simu_score_dict[model_name[2]] = pred_answer
                if model_name[3] in obj["SIMU"]:
                    # 除去第一个正确答案
                    value = json.loads(obj["SIMU"][model_name[3]])[1:]
                    # 预测的正确答案前5的分数是什么
                    pred_answer = sorted(value)[0:5]
                    simu_score_dict[model_name[3]] = pred_answer
                if model_name[4] in obj["SIMU"]:
                    # 除去第一个正确答案
                    value = json.loads(obj["SIMU"][model_name[4]])[1:]
                    # 预测的正确答案前5的分数是什么
                    pred_answer = sorted(value)[0:5]
                    simu_score_dict[model_name[4]] = pred_answer
                
                row = []
                for name in model_name:
                    row += simu_score_dict[name]
                    simu_score_dict[name] = []
                rows.append(row)

        return rows
    

    def get_target(self, file_path, model_name):
        with open(file_path, "r") as f:
            simu_score_dict = {}
            for name in model_name:
                simu_score_dict[name] = []
            rows = []
            i = 0

            objects = ijson.items(f, "item")
            for obj in objects:
                i += 1
                if i % 100 == 0:
                    print(f"Fetching the score of query: {str(i)}")
                if model_name[0] in obj["SIMU"]:
                    # 除去第一个正确答案
                    value = json.loads(obj["SIMU"][model_name[0]])[1:]
                    # 正确答案的分数
                    answer = json.loads(obj["SIMU"][model_name[0]])[0]
                    # 按照分数排序，排序后正确答案的排名是多少
                    corr_rank = sorted(value, reverse=True).index(answer) + 1
                    simu_score_dict[model_name[0]].append(corr_rank)
                if model_name[1] in obj["SIMU"]:
                    # 除去第一个正确答案
                    value = json.loads(obj["SIMU"][model_name[1]])[1:]
                    # 正确答案的分数
                    answer = json.loads(obj["SIMU"][model_name[1]])[0]
                    # 按照分数排序，排序后正确答案的排名是多少
                    corr_rank = sorted(value, reverse=True).index(answer) + 1
                    simu_score_dict[model_name[1]].append(corr_rank)
                if model_name[2] in obj["SIMU"]:
                    # 除去第一个正确答案
                    value = json.loads(obj["SIMU"][model_name[2]])[1:]
                    # 正确答案的分数
                    answer = json.loads(obj["SIMU"][model_name[2]])[0]
                    # 按照分数排序，排序后正确答案的排名是多少
                    corr_rank = sorted(value, reverse=True).index(answer) + 1
                    simu_score_dict[model_name[2]].append(corr_rank)
                if model_name[3] in obj["SIMU"]:
                    # 除去第一个正确答案
                    value = json.loads(obj["SIMU"][model_name[3]])[1:]
                    # 正确答案的分数
                    answer = json.loads(obj["SIMU"][model_name[3]])[0]
                    # 按照分数排序，排序后正确答案的排名是多少
                    corr_rank = sorted(value).index(answer) + 1
                    simu_score_dict[model_name[3]].append(corr_rank)
                if model_name[4] in obj["SIMU"]:
                    # 除去第一个正确答案
                    value = json.loads(obj["SIMU"][model_name[4]])[1:]
                    # 正确答案的分数
                    answer = json.loads(obj["SIMU"][model_name[4]])[0]
                    # 按照分数排序，排序后正确答案的排名是多少
                    corr_rank = sorted(value).index(answer) + 1
                    simu_score_dict[model_name[4]].append(corr_rank)
                
                row = []
                for name in model_name:
                    row += simu_score_dict[name]
                    simu_score_dict[name] = []
                rows.append(row)

        return rows


    def target_normalized(self, file_path):
        data = pd.read_csv(file_path)
        for i in range(data.shape[0]):
            ranks = data.iloc[i].to_list()
            total = sum(ranks)
            for j in range(len(ranks)):
                ranks[j] = ranks[j] / total
            ranks = pd.Series(ranks)
            data.iloc[i] = ranks
        file_path = os.path.join(os.path.dirname(file_path), f"{file_path.split('/')[-1].split('.')[0]}_normalized.csv")
        data.to_csv(file_path, index=False)

    # 这个可以解决数值中有负值的数据，将数值缩放到0-1的范围内
    def normalize_min_max(self, file_path):
        normalized_data = {}
        data = pd.read_csv(file_path)
        for column in data.columns:
            normalized_data[column] = []

        for column in data.columns:
            column_values = data[column].values
            nor_data = (column_values - data[column].min()) / (data[column].max() - data[column].min())
            normalized_data[column] = nor_data
            # print(normalized_data)
        df = pd.DataFrame(normalized_data)
        new_file_path = os.path.join(os.path.dirname(file_path), f"{file_path.split('/')[-1].split('.')[0]}_norm.csv")
        df.to_csv(new_file_path, index=False) 


    # 因为ATISE和TERO的数值是越小越好，所以需要处理一下输入的数据，这里输入的数据需要是normalized之后的数据
    def reverse_transe_ditmult(self, input_file):
        num_pred = 5
        data = pd.read_csv(input_file)
        for i in range(num_pred):
            transe = data.iloc[:, i].values
            new_transe = 1 - transe
            data.iloc[:, i] = pd.Series(new_transe)
            
        for i in range(num_pred):
            simple = data.iloc[:, i].values
            new_simple = 1 - simple
            data.iloc[:, i] = pd.Series(new_simple)
            
        for i in range(num_pred):
            distmult = data.iloc[:, i].values
            new_distmult = 1 - distmult
            data.iloc[:, i] = pd.Series(new_distmult)

        new_file_path = os.path.join(os.path.dirname(input_file), f"{input_file.split('/')[-1].split('.')[0]}_reverse.csv")
        data.to_csv(new_file_path, index=False)


    # 因为ATISE和TERO的数值是越小越好，所以需要处理一下输入的数据，这里输入的数据需要是normalized之后的数据
    def reverse_target(self, input_file):
        data = pd.read_csv(input_file)

        transe = data.iloc[:, 0].values
        simple = data.iloc[:, 1].values
        distmult = data.iloc[:, 2].values
        tero = data.iloc[:, 3].values
        atise = data.iloc[:, 4].values

        new_transe = 1 - transe
        new_simple = 1 - simple
        new_distmult = 1 - distmult
        new_tero = 1 - tero
        new_atise = 1 - atise

        data.iloc[:, 0] = pd.Series(new_transe)
        data.iloc[:, 1] = pd.Series(new_simple)
        data.iloc[:, 2] = pd.Series(new_distmult)
        data.iloc[:, 3] = pd.Series(new_tero)
        data.iloc[:, 4] = pd.Series(new_atise)

        new_file_path = os.path.join(os.path.dirname(input_file), f"{input_file.split('/')[-1].split('.')[0]}_reverse.csv")
        data.to_csv(new_file_path, index=False)

    def reverse_target_horizontal(self, input_file):
        data = pd.read_csv(input_file)
        for i in range(data.shape[0]):
            ranks = data.iloc[i].to_list()
            for j in range(len(ranks)):
                ranks[j] = 1 - ranks[j]
            data.iloc[i] = pd.Series(ranks)

        new_file_path = os.path.join(os.path.dirname(input_file), f"{input_file.split('/')[-1].split('.')[0]}_reverse_hor.csv")
        data.to_csv(new_file_path, index=False)

    def save_csv(self, file_name, rows, model_name):
        num_pred = 5
        row_name = []
        for name in model_name:
            for i in range(num_pred):
                row_name.append(f"{name}_{i+1}")

        with open(file_name, "w") as f:
            writer = csv.writer(f)
            writer.writerow(row_name)
            for row in rows:
                writer.writerow(row)


if __name__ == "__main__":
    model_name = ["DE_TransE", "DE_SimplE", "DE_DistMult", "TERO", "ATISE"]

    # 排名，先弄成小数，然后反过来，然后normalized？
    dataset = NNDataset_weights()
    # dataset.target_normalized("dataset/NN/5p_5w/intermediate file/train_target.csv")
    # dataset.reverse_target_horizontal("dataset/NN/5p_5norm_target/train_target_normalized.csv")
    # dataset.target_normalized("dataset/NN/5p_5norm_target/train_target_normalized_reverse_hor.csv")
    
    # dataset.target_normalized("dataset/NN/5p_5norm_target/test_target.csv")
    # dataset.reverse_target_horizontal("dataset/NN/5p_5norm_target/test_target_normalized.csv")
    # dataset.target_normalized("dataset/NN/5p_5norm_target/test_target_normalized_reverse_hor.csv")
    
    # dataset.target_normalized("dataset/NN/5p_5norm_target/validation_target.csv")
    # dataset.reverse_target_horizontal("dataset/NN/5p_5norm_target/validation_target_normalized.csv")
    dataset.target_normalized("dataset/NN/5p_5norm_target/validation_target_normalized_reverse_hor.csv")


    # Training Input: 25个输入，每个输入是预测的正确答案的分数。
    # rows_input_allwin = dataset.get_input_25("dataset/scores/query_ens_train_sim_scores.json", model_name)
    # dataset.save_csv("dataset/NN/25t_5w/train_input.csv", rows_input_allwin, model_name)
    # dataset.normalize_min_max("dataset/NN/25t_5w/train_input.csv")
    # dataset.reverse_transe_ditmult("dataset/NN/25t_5w/train_input_norm.csv")

    # Validation Input: 25个输入，每个输入是预测的正确答案的分数。
    # rows_input_allwin = dataset.get_input_25("dataset/scores/query_ens_validation_sim_scores.json", model_name)
    # dataset.save_csv("dataset/NN/25t_5w/validation_input.csv", rows_input_allwin, model_name)
    # dataset.normalize_min_max("dataset/NN/25t_5w/validation_input.csv")
    # dataset.reverse_transe_ditmult("dataset/NN/25t_5w/validation_input_norm.csv")

    # Test Input: 25个输入，每个输入是预测的正确答案的分数。
    # rows_input_allwin = dataset.get_input_25("dataset/scores/query_ens_test_sim_scores.json", model_name)
    # dataset.save_csv("dataset/NN/25t_5w/test_input.csv", rows_input_allwin, model_name)
    # dataset.normalize_min_max("dataset/NN/25t_5w/test_input.csv")
    # dataset.reverse_transe_ditmult("dataset/NN/25t_5w/test_input_norm.csv")



    # Training Input: 5个输入，每个输入是预测的正确答案的分数。
    # rows_input_allwin = dataset.get_input("dataset/scores/query_ens_train_sim_scores.json", model_name)
    # dataset.save_csv("dataset/NN/5a_5w/train_input.csv", rows_input_allwin, model_name)
    # dataset.normalize_min_max("dataset/NN/5a_5w/train_input.csv")
    # dataset.reverse_transe_ditmult("dataset/NN/5a_5w/train_input_norm.csv")

    # Training Output: 5个输出，每个输出是真实的正确答案的排名
    # rows_output_allwin = dataset.get_target("dataset/scores/query_ens_train_sim_scores.json", model_name)
    # dataset.save_csv("dataset/NN/5a_5w/train_target.csv", rows_output_allwin, model_name)
    # dataset.normalize_min_max("dataset/NN/5a_5w/train_target.csv")
    # dataset.reverse_target("dataset/NN/5a_5w/train_target_norm.csv")

    # Validation Input: 5个输入，每个输入是预测的正确答案的分数。
    # rows_input_allwin = dataset.get_input("dataset/scores/query_ens_validation_sim_scores.json", model_name)
    # dataset.save_csv("dataset/NN/5a_5w/validation_input.csv", rows_input_allwin, model_name)
    # dataset.normalize_min_max("dataset/NN/5a_5w/validation_input.csv")
    # dataset.reverse_transe_ditmult("dataset/NN/5a_5w/validation_input_norm.csv")

    # Validation Output: 5个输出，每个输出是真实的正确答案的排名
    # rows_output_allwin = dataset.get_target("dataset/scores/query_ens_validation_sim_scores.json", model_name)
    # dataset.save_csv("dataset/NN/5a_5w/validation_target.csv", rows_output_allwin, model_name)
    # dataset.normalize_min_max("dataset/NN/5a_5w/validation_target.csv")
    # dataset.reverse_target("dataset/NN/5a_5w/validation_target_norm.csv")

    # Test Input: 5个输入，每个输入是预测的正确答案的分数。
    # rows_input_allwin = dataset.get_input("dataset/scores/query_ens_test_sim_scores.json", model_name)
    # dataset.save_csv("dataset/NN/5a_5w/test_input.csv", rows_input_allwin, model_name)
    # dataset.normalize_min_max("dataset/NN/5a_5w/test_input.csv")
    # dataset.reverse_transe_ditmult("dataset/NN/5a_5w/test_input_norm.csv")

    # Test Output: 5个输出，每个输出是真实的正确答案的排名
    # rows_output_allwin = dataset.get_target("dataset/scores/query_ens_test_sim_scores.json", model_name)
    # dataset.save_csv("dataset/NN/5a_5w/test_target.csv", rows_output_allwin, model_name)
    # dataset.normalize_min_max("dataset/NN/5a_5w/test_target.csv")
    # dataset.reverse_target("dataset/NN/5a_5w/test_target_norm.csv")

    # dataset.normalize_min_max("new_results/prediction.csv")


    # dataset = NNDataset_min_true()
    # dataset.normaliza("ens_train_h5_h5 copy.csv")
    # dataset.normalize_min_max("h5h5target.csv")

    # rows_input = dataset.get_input("new_results/query_ens_test_sim_scores.json", model_name)
    # dataset.save_csv("h5h5input_test.csv", rows_input, model_name)
    # rows_target = dataset.get_target("new_results/query_ens_test_sim_scores.json", model_name)
    # dataset.save_csv("h5h5target_test.csv", rows_target, model_name)


    # rows = dataset.get_input("new_results/temp_sim_scores.json", model_name)
    # dataset.save_csv("h5h5input.csv", rows, model_name)
    # rows = dataset.get_target("new_results/temp_sim_scores.json", model_name)
    # dataset.save_csv("h5h5target.csv", rows, model_name)


# if __name__ == "__main__":
#     model_name = ["DE_TransE", "DE_SimplE", "DE_DistMult", "TERO", "ATISE"]

    # Get the inputs and targets of training dataset
    # dataset = Entity_5()
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

