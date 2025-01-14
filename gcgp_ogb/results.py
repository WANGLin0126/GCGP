# process the txt results file and generate the final results
import csv
import pandas as pd
import re

line_numbers = [623]  
header = ["Dataset", "Cond_Size", "Ridge", "k", "Learn_A", "Norm", "Epoch", "Acc", "Std."]


with open('results.csv', 'w', newline='', encoding='utf-8') as cvs_file:
    writer = csv.writer(cvs_file)
    writer.writerow(header)

    for dataset in ['ogbn-arxiv']:  
        for cond_size in [90, 454, 909]:
            for ridge in ['1e-3', '1e-2', '1e-1', '5e-1', '1e0', '5', '1e1']:
                for k in [1, 2, 3, 4]:
                    for learn_A in [0, 1]:
                        for norm in [0, 1]:
                            path = 'outputs/' + dataset + '_size_' + str(cond_size) + '_ridge_' + str(ridge) + '_k_' + str(k) + '_learn_A_' + str(learn_A) + '_norm_' + str(norm) + '.txt'
                            
                            with open(path, 'r') as file:
                                lines = file.readlines()
                                # selected_lines = [lines[i] for i in line_numbers if i < len(lines)]
                                selected_lines = lines[-2]
                                extracted_data = re.findall(r'\d+\.\d+|\d+', selected_lines)
                                # extracted_data = selected_lines[0].split()[1:]
                                finall_line = [dataset, cond_size, ridge, k, learn_A, norm] + extracted_data

                                writer.writerow(finall_line)

df = pd.read_csv('results.csv')
print(df.columns.tolist())
rows = []
for dataset in ['ogbn-arxiv']: 
    for cond_size in [90, 454, 909]:
        for leanr_A in [0, 1]:
            dataset_rows = df[(df.iloc[:, 0] == dataset) & (df['Cond_Size'] == cond_size) & (df['Learn_A'] == leanr_A)]
            max_row = dataset_rows.loc[dataset_rows.iloc[:, 7].idxmax()].to_frame().T
            rows.append(max_row.index.values.tolist()[0])

print("-------------------------best resulst------------------------")
print(df.iloc[rows])