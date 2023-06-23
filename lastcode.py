import pandas as pd
import csv
import os
import json

# Normalized the predictions, such that the output will be the weights
def norm(file_path):
    data = pd.read_csv(file_path, header=None)

    for index, row in data.iterrows():
        new_row = row / sum(row)
        data.loc[index] = new_row
    data.to_csv('nn_method/5input/result/prediction_naive_norm.csv', header=False, index=False)
            
# split prediction into 4 files corresponding to 4 elements
def one_element(file_path, elements):
    new_file = []
    for i in range(len(elements)):
        new_file.append(os.path.join(os.path.dirname(file_path), f"{file_path.split('/')[-1].split('.')[0]}_{elements[i]}.csv"))

    with open(file_path, "r") as f:
        with open(new_file[3], "w",  newline='') as f2:
            reader = csv.reader(f)
            writer = csv.writer(f2)
            for i, row in enumerate(reader):
                if i % 4 == 3:  # 每隔4行
                    writer.writerow(row)


def ens_element(file_path):
    element = ['head','relation', 'tail', 'time']

    # Create the file name for each element
    new_file = []
    for i in range(len(element)):
        new_file.append(os.path.join(os.path.dirname(file_path), f"ensemble_{element[i]}.csv"))

    with open(file_path, "r") as f:
        data = json.load(f)
        new_data = data[3::4]
    with open(new_file[3], 'w') as outfile:
        json.dump(new_data, outfile, indent=4)


if __name__ == "__main__":
    elements = ["head", "relation", "tail", "time"]
    file_path = "nn_method/25input/result/prediction_naive.csv"
    # get the csv file of weights
    one_element(file_path, elements)

    # file_path = "dataset/ranks/ensemble_query_ens_test_sim_ranks.json"
    # # get the ranks after ensemble with elementwise 
    # ens_element(file_path)