"""
process the txt files in /outputs/ and generate the results.csv file with the following columns:
["Dataset", "Cond_ratio", "Ridge", "k", "Learn_A", "Epoch", "Acc", "Std.", "Time"]
and 
output the best results for each dataset, cond_ratio, and learn_A
"""

import csv
import pandas as pd
import re

# line_numbers = [223]  
# header = ["Dataset", "Cond_ratio", "Ridge", "k", "Learn_A", "Epoch", "Acc", "Std.", "Time"]


# with open('results.csv', 'w', newline='', encoding='utf-8') as cvs_file:
#     writer = csv.writer(cvs_file)
#     writer.writerow(header)

#     for dataset in ['Cora', 'Citeseer','Pubmed', 'Photo', 'Computers']:  
#         for cond_ratio in [0.25, 0.50, 1]:
#             for ridge in ['1e-3', '1e-2', '1e-1', '5e-1', '1e0', '5', '1e1']:
#                 for k in [1, 2, 3, 4, 5]:
#                     for learn_A in [0,1]:
#                         for kernel in ['SGNK']:
#                             path = 'outputs/' + dataset + '_kernel_' + kernel + '_size_' + str(cond_ratio) + '_ridge_' + str(ridge) + '_k_' + str(k) + '_learn_A_' + str(learn_A) + '.txt'
                            
#                             # if learn_A :
#                             #     line_numbers = [224]
#                             # else:
#                             #     line_numbers = [223]

#                             with open(path, 'r') as file:
#                                 lines = file.readlines()
#                                 selected_lines = [lines[i] for i in line_numbers if i < len(lines)]
                                
#                                 extracted_data = re.findall(r'\d+\.\d+|\d+', selected_lines[0])
#                                 # extracted_data = selected_lines[0].split()[1:]
#                                 finall_line = [dataset, cond_ratio, ridge, k, learn_A] + extracted_data

#                                 writer.writerow(finall_line)


df = pd.read_csv('results.csv')
print(df.columns.tolist())
rows = []
for dataset in ['Cora','Citeseer', 'Pubmed', 'Photo', 'Computers']:  
    for cond_ratio in [0.25, 0.50, 1]:
        for leanr_A in [0, 1]:

            dataset_rows = df[(df.iloc[:, 0] == dataset) & (df['Cond_ratio'] == cond_ratio) & (df['Learn_A'] == leanr_A)]
            max_row = dataset_rows.loc[dataset_rows.iloc[:, 6].idxmax()].to_frame().T
            rows.append(max_row.index.values.tolist()[0])

print("-------------------------best resulst------------------------")
print(df.iloc[rows])