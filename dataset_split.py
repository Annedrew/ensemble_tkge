import json
from sklearn.model_selection import train_test_split
import random

class DatasetSplit():
    def dataset_split(self, file_path):
        with open(file_path, 'r') as file:
            dataset = file.readlines()
            random.shuffle(dataset)

            train_ratio = 0.7
            train_size = int(len(dataset) * train_ratio)

            ens_train = dataset[:train_size]
            ens_test = dataset[train_size:]
            with open("dataset/ens_icews14/ens_train.txt","w") as f:
                f.writelines(ens_train)
            with open("dataset/ens_icews14/ens_test.txt","w") as f:
                f.writelines(ens_test)


        # with open("results/icews14/ranked_quads.json", "r") as f:
        #     data = json.load(f)
        #     query_id = []
        #     for query in range(len(data)):
        #         query_id.append(data[query]["FACT_ID"])
        #     # random_state=42 ensures get the same split every time
        #     ens_train_id, ens_test_id = train_test_split(query_id, test_size=0.3, random_state=42)
        #     # 25096
        #     # print(len(ens_train_id))
        #     # 10756
        #     # print(len(ens_test_id))
        #     ens_train = []
        #     ens_test = []
        #     for train_id in range(len(ens_train_id)):
        #         for query in range(len(data)):
        #             if data[query]["FACT_ID"] == ens_train_id[train_id]:
        #                 ens_train.append(data[query])

        #     for test_id in range(len(ens_test_id)):
        #         for query in range(len(data)):
        #             if data[query]["FACT_ID"] == ens_test_id[test_id]:
        #                 ens_test.append(data[query])

        # with open("/ens_dataset/ens_train.json", "w") as f:
        #     # 25096
        #     json.dump(ens_train, f, indent=4)
        # with open("/ens_dataset/ens_test.json", "w") as f:
        #     # 10756
        #     json.dump(ens_test, f, indent=4)