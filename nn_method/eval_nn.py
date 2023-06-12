import json
import ijson
import csv

class EvalNN:
    def __init__(self):
        pass

    
    def get_true_id(self, file_path):
        with open(file_path, "r") as f:
            objects = ijson.items(f, "item")
            rows_true = []
            num_query = 0
            for obj in objects:
                num_query += 1
                answer = json.loads(obj["RANK"]["DE_TransE"])[0]
                list_simu = json.loads(obj["RANK"]["DE_TransE"])[1:]
                # 答案的ID
                index_true = list_simu.index(answer) + 1
                rows_true.append(index_true)
        
        return rows_true, num_query


    # def get_pred_id(self, file_path):
    #     with open(file_path, "r") as f:
    #         reader = csv.reader(f)
    #         rows_pred = []
    #         for row in reader:
    #             sorted_list = sorted(row)
    #             row = [sorted_list.index(element) + 1 for element in row]
    #             # Get the ID of predicted answer
    #             index_pred = row.index(1) + 1
    #             rows_pred.append(index_pred)
        
    #     return rows_pred
    
    def get_pred_id(self, file_path):
        with open(file_path, "r") as f:
            reader = csv.reader(f)
            rows_pred = []
            for row in reader:
                for element in row:
                    if float(element) > 0.5:
                        # 答案ID
                        ind = row.index(element)
                rows_pred.append(ind)
        
        return rows_pred
        

    def eval_hit(self, rows_true, num_query, rows_pred):
        b = 0
        for i in range(len(rows_pred)):
            if rows_pred[i] == rows_true[i]:
                b += 1
        # b = 0
        # for i in range(len(rows_true)):
        #     if rows_true[i] == rows_pred[i]:
        #         b += 1
        hit1_pred = b / num_query
        print("Predicted HIT@1: ", hit1_pred)

# b = 0
# for i in range(len(rows_true)):
#     if rows_true[i] == rows_pred[i]:
#         b += 1
# hit1 = b / len(rows_true)
# print(len(rows_true))
# print(len(json.loads(obj["RANK"]["DE_TransE"])[1:]))

# # hit1_true =  / len(rows_true)
# print(b)
# print("HIT@1", hit1)


if __name__ == "__main__":
    eval_nn = EvalNN()
    # 测试："new_results/temp_sim_scores.json"
    rows_true, num_query = eval_nn.get_true_id("new_results/query_ens_test_sim_scores.json")
    rows_pred = eval_nn.get_pred_id("new_results/prediction.csv")
    eval_nn.eval_hit(rows_true, num_query, rows_pred)
    