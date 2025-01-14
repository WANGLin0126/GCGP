""" 
collect the txt files in ./outputs_generalization/ and generate a csv file (resukts_generalization.csv) with the following columns:
["Dataset","Model", "Cond_ratio", "Learn_A", "k","Epoch", "Mean_Acc", "Std.", "Best_Acc"]
and 
output the best results for each dataset, model, and cond_ratio
"""
import csv
import pandas as pd
import re

# line_numbers = [222]  
header = ["Dataset","Model", "Cond_ratio", "Learn_A", "k","Epoch", "Mean_Acc", "Std.", "Best_Acc"]

with open('results_generalization.csv', 'w', newline='', encoding='utf-8') as cvs_file:
    writer = csv.writer(cvs_file)
    writer.writerow(header)

    for dataset in ['Cora', 'Citeseer','Pubmed', 'Photo', 'Computers','CS', 'Physics']:
        for model in ['GCN','SGC', 'GAT', 'SAGE', 'APPNP', 'Cheby', 'MLP']:
            for cond_ratio in [0.25, 0.50, 1.0]:
                for k in [1, 2, 3, 4, 5]:
                    for learn_A in [0,1]:
                        path = 'outputs_generalization/' + dataset +'_SGNK_'+ model + '_size_' + str(cond_ratio) + '_k_' + str(k) + '_learnA_' + str(learn_A) + '.txt'
                        

                        with open(path, 'r') as file:
                            lines = file.readlines()
                            selected_lines = lines[-1]
                            best_acc = lines[-3].strip()
                            extracted_data = re.findall(r'\d+\.\d+|\d+', selected_lines)
                            finall_line = [dataset, model, cond_ratio, learn_A, k] + extracted_data + [best_acc]

                            writer.writerow(finall_line)


df = pd.read_csv('results_generalization.csv')
print(df.columns.tolist())
rows = []
for dataset in ['Cora','Citeseer', 'Pubmed', 'Photo', 'Computers', 'CS']:  
    for cond_ratio in [0.25, 0.50, 1]:
        for model in ['GCN', 'SGC', 'GAT', 'SAGE', 'APPNP', 'Cheby', 'MLP']:

            dataset_rows = df[(df.iloc[:, 0] == dataset) & (df['Cond_ratio'] == cond_ratio) & (df['Model'] == model)]
            max_row = dataset_rows.loc[dataset_rows.iloc[:, 8].idxmax()].to_frame().T
            rows.append(max_row.index.values.tolist()[0])

print("-------------------------best resulst------------------------")

pd.set_option('display.max_rows', None)
print(df.iloc[rows])