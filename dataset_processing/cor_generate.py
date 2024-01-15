import numpy as np
import pandas as pd
import csv
import re
import json
import os

class QueryGenerate():
    def __init__(self):
        pass


    # Convert txt into csv
    # file_path: test.txt
    def txt_to_csv(self, file_path):
        csv_file = os.path.join("dataset/intermediate_file/", file_path.split('/')[-1].split('.')[0] + '.csv')
        with open(csv_file, 'w', encoding='utf-8') as f1:
            writer = csv.writer(f1, delimiter='\t')
            with open(file_path, encoding='utf-8') as f2:
                for line in f2:
                    line = re.split("[\t\n]", line)
                    writer.writerow(line)
        print(f"'{file_path}' has been converted to '{csv_file}'.")
        
        return csv_file


    # Generate corrupted quadruple and save it as csv file
    # file_path: test.csv
    # cor_csv_file: corrupted_quadruple_test.csv
    def generate_corrupted_quadruple(self, file_path):
        with open(file_path, encoding='utf-8') as f:
            cor_csv_file = os.path.join("dataset/intermediate_file/", 'query_' + file_path.split('/')[-1])
            with open(cor_csv_file, 'w', encoding='utf-8') as f2:
                f2.write("HEAD\tRELATION\tTAIL\tTIME\tANSWER" + '\n')
                for line in f:
                    list = line.strip('\n').split('\t')
                    if list[-1] == '':
                        list = list[0:-1]
                    else:
                        pass
                    a = np.array(list)
                    for i in range(4):
                        if i == 0:
                            co_qu = np.array([0, a[1], a[2], a[3], a[0]])
                            f2.write('\t'.join(co_qu) + '\n')
                        elif i == 1:
                            co_qu = np.array([a[0], 0, a[2], a[3], a[1]])
                            f2.write('\t'.join(co_qu) + '\n')
                        elif i == 2:
                            co_qu = np.array([a[0], a[1], 0, a[3], a[2]])
                            f2.write('\t'.join(co_qu) + '\n')
                        elif i == 3:
                            co_qu = np.array([a[0], a[1], a[2], 0, a[3]])
                            f2.write('\t'.join(co_qu) + '\n')
        print(f"Corrupted quadruples are generated, saved in {cor_csv_file}.")
        
        return cor_csv_file


    # Ddd ID for each fact
    def add_fact_id(self, file_path):
        data = pd.read_csv(file_path, sep='\t', encoding='utf-8')
        fact_column = []
        num1 = 0
        num2 = 0
        for index, row in data.iterrows():
            if index % 4 == 0:
                num1 += 1
                num2 = 0
            else:
                pass
            fact_column.append(f'FACT_{num1}_{num2}')
            num2 += 1
        data['FACT_ID'] = fact_column
        data = data[['FACT_ID', 'HEAD', 'RELATION', 'TAIL', 'TIME', 'ANSWER']]
        data.to_csv(file_path, index=False)
        print("FACT_ID has been added.")


    # Convert csv into json: [{key: value}, {key: value}...]
    def csv_to_json(self, file_path):
        json_file = os.path.join("dataset/queries/", file_path.split('/')[-1].split('.')[0] + '.json')
        with open(json_file, 'w', encoding='utf-8') as js_f:
            js_f.write('[')
            with open(file_path, encoding='utf-8') as f2:
                records = csv.DictReader(f2)
                first = True
                for row in records:
                    if not first:
                        js_f.write(',')
                    first=False
                    json.dump(row, js_f, indent=4, ensure_ascii=False)
            js_f.write(']')
        print(f"'{file_path}' has been converted to '{json_file}'.")


    def query_generate(self, file_path):
        csv = self.txt_to_csv(file_path)
        query_csv = self.generate_corrupted_quadruple(csv)
        self.add_fact_id(query_csv)
        self.csv_to_json(query_csv)


if __name__ == "__main__":
    a = QueryGenerate()
    a.query_generate("dataset/ens_dataset/ens_test.txt")
    a.query_generate("dataset/ens_dataset/ens_validation.txt")