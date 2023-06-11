# ALL CLASS:
# SumRank - Analysis the sum of ranks as metric
# CorrectRange - Analysis the concentrated range of correct answer
# Duplicates - Analysis the shared entity/relation/time in the concentrated range
# NeighborDiff - Analysis the difference of scores between each two neighbor rank

import json
import ijson
import csv
import numpy as np
import pandas as pd


# Analysis the sum of ranks as metric
class SumRank:
    def __init__(self):
        pass
    

    def sum_rank(self, file_path, model_name):
        with open(file_path, "r") as f:
        # with open("temp.json", "r") as f:
            model_name = ["DE_TransE", "DE_SimplE", "DE_DistMult", "TERO", "ATISE"]
            data = json.load(f)
            rank_values = [list(item['RANK'].values())[:5] for item in data]
            predictions = np.array(rank_values, dtype=int)
            predictions = predictions.transpose()
            predictions = {model_name[i]: predictions[i].tolist() for i in range(len(predictions))}
            for i in range(len(model_name)):
                instance = len(predictions[model_name[i]])
                ranks = sum(predictions[model_name[i]])
                print(f"Number of instance: {instance}")
                print(f"Sumup Ranks ({model_name[i]}): {ranks}")


# Analysis the concentrated range of correct answer
class CorrectRange():
    def __init__(self):
        pass


    # Get the rank for each model
    def ranks_model(self, file_path, model_name):
        de_transe = []
        de_simple = []
        de_distmult = []
        tero = []
        atise = []
        with open(file_path, "r") as f:
            data = json.load(f)
            for i in range(len(data)):
                for name in model_name:
                    if name == "DE_TransE":
                        de_transe.append(data[i]["RANK"][name])
                    elif name == "DE_SimplE":
                        de_simple.append(data[i]["RANK"][name])
                    elif name == "DE_DistMult":
                        de_distmult.append(data[i]["RANK"][name])
                    elif name == "TERO":
                        tero.append(data[i]["RANK"][name])
                    elif name == "ATISE":
                        atise.append(data[i]["RANK"][name])
            
        # Convert into numpy array
        de_transe = np.array(de_transe)
        de_simple = np.array(de_simple)
        de_distmult = np.array(de_distmult)
        tero = np.array(tero)
        atise = np.array(atise)

        return de_transe, de_simple, de_distmult, tero, atise

    
    def rank_unique(self, de_transe, de_simple, de_distmult, tero, atise):
        # Take the unique
        de_transe_rank, de_transe_counts = np.unique(de_transe, return_counts=True)
        de_simple_rank, de_simple_counts = np.unique(de_simple, return_counts=True)
        de_distmult_rank, de_distmult_counts = np.unique(de_distmult, return_counts=True)
        tero_rank, tero_counts = np.unique(tero, return_counts=True)
        atise_rank, atise_counts = np.unique(atise, return_counts=True)

        # Save csv
        with open("range_concentrate.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["DE_TransE_Rank", "DE_TransE_Counts", "DE_SimplE_Rank", "DE_SimplE_Counts", "DE_DistMult_Rank", "DE_DistMult_Counts", "TERO_Rank", "TERO_Counts", "ATISE_Rank", "ATISE_Counts"])
            for rank1, count1, rank2, count2, rank3, count3, rank4, count4, rank5, count5 in zip(de_transe_rank, de_transe_counts, de_simple_rank, de_simple_counts, de_distmult_rank, de_distmult_counts, tero_rank, tero_counts, atise_rank, atise_counts):
                writer.writerow([rank1, count1, rank2, count2, rank3, count3, rank4, count4, rank5, count5])


    def order_csv(self, file_path):
        with open(file_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)
        sorted_rows = list(sorted(rows, key=lambda row: int(row[1]), reverse=True))
        with open(f"ordered_{file_path}", "w") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow(sorted_rows)


# Analysis the shared entity/relation/time in the concentrated range
class Duplicates:
    def __init__(self):
        pass


    # Duplicate in total
    # can be used to get target too
    def detect_dupli(self, file_path):
        with open(file_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)
            duplicates = []
            for row in rows:
                row = [int(i) for i in row]
                row = np.array(row)
                id, counts = np.unique(row, return_counts=True)
                duplicate = id[counts > 1]
                if len(duplicate) == 0:
                    # -1 means no duplicates for this query
                    duplicate = [-1]
                duplicates.append(duplicate)

        return duplicates
    

    def save_csv(self, file_path, duplicates):
        with open(file_path, "w") as f:
            writer = csv.writer(f)
            for row in duplicates:
                writer.writerow(row)


# Analysis the difference of scores between each two neighbor rank
class NeighborDiff:
    def __init__(self):
        pass


    def difference(self, file_path, model_name):
        sorted_data = {}
        for name in model_name:
            sorted_data[name] = []

        with open(file_path, "r") as f:
            objects = ijson.items(f, "item")
            for obj in objects:
                # TODO: r, t and T can also be implemented
                if obj["HEAD"] == "0":
                    if model_name[0] in obj["RANK"]:
                        sorted_data[name] = sorted(json.loads(obj["RANK"][model_name[0]]))
                        differences_0 = [sorted_data[name][i+1] - sorted_data[name][i] for i in range(len(sorted_data[name])-1)]
                    if model_name[1] in obj["RANK"]:
                        sorted_data[name] = sorted(json.loads(obj["RANK"][model_name[1]]))
                        differences_1 = [sorted_data[name][i+1] - sorted_data[name][i] for i in range(len(sorted_data[name])-1)]
                    if model_name[2] in obj["RANK"]:
                        sorted_data[name] = sorted(json.loads(obj["RANK"][model_name[2]]))
                        differences_2 = [sorted_data[name][i+1] - sorted_data[name][i] for i in range(len(sorted_data[name])-1)]
                    if model_name[3] in obj["RANK"]:
                        sorted_data[name] = sorted(json.loads(obj["RANK"][model_name[3]]))
                        differences_3 = [sorted_data[name][i+1] - sorted_data[name][i] for i in range(len(sorted_data[name])-1)]
                    if model_name[4] in obj["RANK"]:
                        sorted_data[name] = sorted(json.loads(obj["RANK"][model_name[4]]))
                        differences_4 = [sorted_data[name][i+1] - sorted_data[name][i] for i in range(len(sorted_data[name])-1)]
                    filename = 'new_results/h_diff.csv'
                    with open(filename, 'w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(['Difference_0', 'Difference_1', 'Difference_2', 'Difference_3', 'Difference_4'])
                        writer.writerows([list(diff) for diff in zip(differences_0, differences_1, differences_2, differences_3, differences_4)])


# Analyse how many queries get the answer less or equal than 100
class Top100:
    def hit_100(self):
        extracted_values = []
        with open("results/icews14/ranked_quads.json", "r") as f:
            data = json.load(f)

            transe = []
            simple = []
            distmult = []
            tero = []
            atise = []
            for query in range(len(data)):
                transe.append(json.loads(data[query]['RANK']['DE_TransE']))
                simple.append(json.loads(data[query]['RANK']['DE_SimplE']))
                distmult.append(json.loads(data[query]['RANK']['DE_DistMult']))
                tero.append(json.loads(data[query]['RANK']['TERO']))
                atise.append(json.loads(data[query]['RANK']['ATISE']))
            # TransE: 35852
            # SimplE: 35852
            # DistMult: 35852
            # TeRo: 35852
            # ATiSe: 35852
            print(f"TransE: {len(transe)}")
            print(f"SimplE: {len(simple)}")
            print(f"DistMult: {len(distmult)}")
            print(f"TeRo: {len(tero)}")
            print(f"ATiSe: {len(atise)}\n")
                
            transe1 = [rank for rank in transe if rank < 100]
            simple1 = [rank for rank in simple if rank < 100]
            distmult1 = [rank for rank in distmult if rank < 100]
            tero1 = [rank for rank in tero if rank < 100]
            atise1 = [rank for rank in atise if rank < 100]

            # TransE(>100): 28828
            # SimplE(>100): 27559
            # DistMult(>100): 27117
            # TeRo(>100): 28865
            # ATiSe(>100): 28802
            print(f"TransE(>100): {len(transe1)}")
            print(f"SimplE(>100): {len(simple1)}")
            print(f"DistMult(>100): {len(distmult1)}")
            print(f"TeRo(>100): {len(tero1)}")
            print(f"ATiSe(>100): {len(atise1)}\n")

            # TransE(>100): 0.8040834542006025
            # SimplE(>100): 0.768687939306036
            # DistMult(>100): 0.7563594778533973
            # TeRo(>100): 0.8051154747294432
            # ATiSe(>100): 0.8033582505857414
            print(f"TransE(>100): {len(transe1)/len(transe)}")
            print(f"SimplE(>100): {len(simple1)/len(simple)}")
            print(f"DistMult(>100): {len(distmult1)/len(distmult)}")
            print(f"TeRo(>100): {len(tero1)/len(tero)}")
            print(f"ATiSe(>100): {len(atise1)/len(atise)}")


class Rank100:    
    # 统计排名在前100的问题有多少个，是整个test集合的
    # 前100个超过一半了
    def hit_100(self):
        with open("results/icews14/ranked_quads.json", "r") as f:
            data = json.load(f)

            transe = []
            simple = []
            distmult = []
            tero = []
            atise = []
            for query in range(len(data)):
                transe.append(json.loads(data[query]['RANK']['DE_TransE']))
                simple.append(json.loads(data[query]['RANK']['DE_SimplE']))
                distmult.append(json.loads(data[query]['RANK']['DE_DistMult']))
                tero.append(json.loads(data[query]['RANK']['TERO']))
                atise.append(json.loads(data[query]['RANK']['ATISE']))
           
            print(f"TransE: {len(transe)}") # TransE: 35852
            print(f"SimplE: {len(simple)}") # SimplE: 35852
            print(f"DistMult: {len(distmult)}") # DistMult: 35852
            print(f"TeRo: {len(tero)}") # TeRo: 35852
            print(f"ATiSe: {len(atise)}\n") # ATiSe: 35852
                
            transe1 = [rank for rank in transe if rank < 100]
            simple1 = [rank for rank in simple if rank < 100]
            distmult1 = [rank for rank in distmult if rank < 100]
            tero1 = [rank for rank in tero if rank < 100]
            atise1 = [rank for rank in atise if rank < 100]

            print(f"TransE(>100): {len(transe1)}") # TransE(>100): 28828
            print(f"SimplE(>100): {len(simple1)}") # SimplE(>100): 27559
            print(f"DistMult(>100): {len(distmult1)}") # DistMult(>100): 27117
            print(f"TeRo(>100): {len(tero1)}") # TeRo(>100): 28865
            print(f"ATiSe(>100): {len(atise1)}\n") # ATiSe(>100): 28802
            
            print(f"TransE(>100): {len(transe1)/len(transe)}") # TransE(>100): 0.8040834542006025
            print(f"SimplE(>100): {len(simple1)/len(simple)}") # SimplE(>100): 0.768687939306036
            print(f"DistMult(>100): {len(distmult1)/len(distmult)}") # DistMult(>100): 0.7563594778533973
            print(f"TeRo(>100): {len(tero1)/len(tero)}") # TeRo(>100): 0.8051154747294432
            print(f"ATiSe(>100): {len(atise1)/len(atise)}") # ATiSe(>100): 0.8033582505857414


    # 找到预测的前100个实体中，重复的实体有多少。
    def detect_dupli(self, file_path, model_nmae):
        # Find where the correct answer is 
        # Get the index of correct answer
        # Using all simulated facts file
        with open(file_path, "r") as f:
            # 这里的只能是item
            objects = ijson.items(f, "item")
            # obj 就是列表中每个单独的json
            for obj in objects:
                if model_name[0] in obj['RANK']:
                    # 越大越好
                    # 获取simu_fact的list：这里要取除了第一个之外的所有值，因为第一个值是答案，不需要排序
                    transe = json.loads(obj['RANK'][model_name[0]])[1:]
                    # 排序：这里reverse=True，是最大值排列在前面
                    transe = sorted(enumerate(transe), reverse=True, key=lambda x: x[1])[:10]
                    transe1 = []
                    for index, value in transe:
                        transe1.append(index)

                if model_name[1] in obj['RANK']:
                    # 越大越好
                    # 获取simu_fact的list：这里要取除了第一个之外的所有值，因为第一个值是答案，不需要排序
                    simple = json.loads(obj['RANK'][model_name[1]])[1:]
                    # 排序：这里reverse=True，是最大值排列在前面
                    simple = sorted(enumerate(simple), reverse=True, key=lambda x: x[1])[:10]
                    
                    simple1 = []
                    for index, value in simple:
                        simple1.append(index)

                if model_name[2] in obj['RANK']:
                    # 越大越好
                    # 获取simu_fact的list：这里要取除了第一个之外的所有值，因为第一个值是答案，不需要排序
                    distmult = json.loads(obj['RANK'][model_name[2]])[1:]
                    # 排序：这里reverse=True，是最大值排列在前面
                    distmult = sorted(enumerate(distmult), reverse=True, key=lambda x: x[1])[:10]
                    distmult1 = []
                    for index, value in distmult:
                        distmult1.append(index)
                
                if model_name[3] in obj['RANK']:
                    # 越小越好
                    # 获取simu_fact的list：这里要取除了第一个之外的所有值，因为第一个值是答案，不需要排序
                    tero = json.loads(obj['RANK'][model_name[3]])[1:]
                    # 排序：这里直接sorted，最小值排列在前面
                    tero = sorted(enumerate(tero), key=lambda x: x[1])[:10]
                    tero1 = []
                    for index, value in tero:
                        tero1.append(index)
                
                if model_name[4] in obj['RANK']:
                    # 越小越好
                    # 获取simu_fact的list：这里要取除了第一个之外的所有值，因为第一个值是答案，不需要排序
                    atise = json.loads(obj['RANK'][model_name[4]])[1:]
                    # 排序：这里直接sorted，最小值排列在前面
                    atise = sorted(enumerate(atise), key=lambda x: x[1])[:10]
                    atise1 = []
                    for index, value in atise:
                        atise1.append(index)
                print(transe1)  
                print(simple1)  
                print(distmult1)  
                print(tero1)  
                print(atise1)
                # 如何查找duplicate比较好？
        # return duplicates


# 每个模型得到的分数，正负是什么情况
model_name = ["DE_TransE", "DE_SimplE", "DE_DistMult", "TERO", "ATISE"]
with open("new_results/temp_sim_scores.json", "r") as f:
    data = json.load(f)
    for i in range(len(data)):
        for value in json.loads(data[i]['RANK']['DE_TransE']):
            if value > 0:
                print(value)
        for value in json.loads(data[i]['RANK']['DE_SimplE']):
            if value < 0:
                print(value)
        for value in json.loads(data[i]['RANK']['DE_DistMult']):
            if value < 0:
                print(value)
        for value in json.loads(data[i]['RANK']['TERO']):
            if value < 0:
                print(value)
        for value in json.loads(data[i]['RANK']['ATISE']):
            if value < 0:
                print(value)

    print(json.loads(data['RANK']))

# 每个模型得到的分数，是越大越好，还是越小越好
# with open("new_results/temp_sim_scores.json", "r") as f:
#     data = json.load(f)
    # for i in range(len(data)):
    # print(f"Correct: {json.loads(data[0]['RANK']['DE_TransE'])[0]}")
    # ordered_data = sorted(json.loads(data[0]['RANK']['DE_TransE']))
    # print(f"Minimum: {ordered_data[0]}")
    # print(f"Maximum: {ordered_data[len(ordered_data)-1]}\n")

    # print(f"Correct: {json.loads(data[0]['RANK']['DE_SimplE'])[0]}")
    # ordered_data = sorted(json.loads(data[1]['RANK']['DE_SimplE']))
    # print(f"Minimum: {ordered_data[0]}")
    # print(f"Maximum: {ordered_data[len(ordered_data)-1]}\n")

    # print(f"Correct: {json.loads(data[0]['RANK']['DE_DistMult'])[0]}")
    # ordered_data = sorted(json.loads(data[0]['RANK']['DE_DistMult']))
    # print(f"Minimum: {ordered_data[0]}")
    # print(f"Maximum: {ordered_data[len(ordered_data)-1]}\n")

    # print(f"Correct: {json.loads(data[0]['RANK']['TERO'])[0]}")
    # ordered_data = sorted(json.loads(data[0]['RANK']['TERO']))
    # print(f"Minimum: {ordered_data[0]}")
    # print(f"Maximum: {ordered_data[len(ordered_data)-1]}\n")

    # print(f"Correct: {json.loads(data[0]['RANK']['ATISE'])[0]}")
    # ordered_data = sorted(json.loads(data[0]['RANK']['ATISE']))
    # print(f"Minimum: {ordered_data[0]}")
    # print(f"Maximum: {ordered_data[len(ordered_data)-1]}\n")
        
# 每个元素，有多少个simulated facts
# with open("new_results/query_ens_test_sim_scores.json", "r") as f:
#     objects = ijson.items(f, 'item')

#     for obj in objects:
#         if "DE_TransE" in obj["RANK"]:
#             print(len(json.loads(obj['RANK']['DE_TransE'])))

# 重新弄一个temp的数据集，有两个完成的预测h,r,t,和T的。替代这个文件：new_results/temp_sim_scores.json
# with open("new_results/query_ens_train_sim_scores.json", "r") as f:
#     objects = ijson.items(f, "item")
#     i = 0
#     a = []
#     for obj in objects:
#         if i < 8:
#             a.append(obj)
#             i += 1
#         else:
#             break

# with open("temp_sim_scores.json", "w") as f:
#     json.dump(a, f, indent=4)

# 第一个是4个都重复的吗？
# 是的


if __name__ == "__main__":
    model_name = ["DE_TransE", "DE_SimplE", "DE_DistMult", "TERO", "ATISE"]
    dataset = Rank100()
    # dataset.detect_dupli("new_results/query_ens_train_sim_scores.json", model_name)
    dataset.detect_dupli("new_results/temp_sim_scores.json", model_name)

    # dataset = NeighborDiff()
    # dataset.hit_100()
    # Decide to try sum of ranks as metric
    # sumrank = SumRank()
    # sumrank.sum_rank("results/icews14/ranked_quads.json", model_name)

    # Decide the input range for each model
    # range = CorrectRange()
    # de_transe, de_simple, de_distmult, tero, atise = range.ranks_model("results/icews14/ranked_quads.json", model_name)
    # range.rank_unique(de_transe, de_simple, de_distmult, tero, atise)
    # range.order_csv("range_concentrate.csv")

    # Decide to use score or one-hot-encoding as input and output
    # duplicate = Duplicates()
    # duplicates = duplicate.detect_dupli("new_results/temp_top_5_id.csv")
    # duplicate.save_csv("new_results/duplicates_temp_top_5_id.csv", duplicates)
    # duplicates = duplicate.detect_dupli("new_results/ens_train_top_5_id.csv")
    # duplicate.save_csv("new_results/duplicates_ens_train_top_5_id.csv", duplicates)

    # Decide to use score or one-hot-encoding as output
    # diff = NeighborDiff()
    # diff.difference("new_results/temp_sim_scores.json", model_name)