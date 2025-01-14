# process the txt results file and generate the final results
import csv
import pandas as pd
import re

line_numbers = [624]  
header = ["Dataset", "Cond_Size", "Ridge", "k", "Learn_A", "Norm", "Epoch", "Acc", "Std."]

# 创建并写入表头到 CSV 文件
with open('results.csv', 'w', newline='', encoding='utf-8') as cvs_file:
    writer = csv.writer(cvs_file)
    writer.writerow(header)

    for dataset in ['Flickr', 'Reddit']: 
        if dataset == 'Flickr':
            cond_sizes = [44, 223, 446]
            line_numbers = [624]  
        elif dataset == 'Reddit':
            cond_sizes = [77, 153, 307]
            line_numbers = [841] 
        for cond_size in cond_sizes:

            if dataset == 'Flickr':
                ridges = [ '1e-3', '1e-2', '1e-1','5e-1', '1e0', '5', '1e1']
            elif dataset == 'Reddit':
                ridges = [ '1e-3', '1e-2', '1e-1', '1e0', '1e1']

            for ridge in ridges:
                if dataset == 'Flickr':
                    K = [1, 2, 3, 4, 5]
                elif dataset == 'Reddit':
                    K = [1, 2, 3, 4]
                for k in K:
                    for learn_A in [0,1]:
                        for norm in [0, 1]:
                            path = 'outputs/' + dataset + '_size_' + str(cond_size) + '_ridge_' + str(ridge) + '_k_' + str(k) + '_learn_A_' + str(learn_A) + '_norm_' + str(norm) + '.txt'
                            
                            # if learn_A :
                            #     line_numbers = [223]
                            # else:
                            #     line_numbers = [222]

                            with open(path, 'r') as file:
                                lines = file.readlines()
                                selected_lines = [lines[i] for i in line_numbers if i < len(lines)]
                                
                                extracted_data = re.findall(r'\d+\.\d+|\d+', selected_lines[0])
                                # extracted_data = selected_lines[0].split()[1:]
                                finall_line = [dataset, cond_size, ridge, k, learn_A, norm] + extracted_data

                                writer.writerow(finall_line)


df = pd.read_csv('results.csv')
print(df.columns.tolist())
rows = []
for dataset in ['Flickr', 'Reddit']:  #'CS', 'Physics'
    if dataset == 'Flickr':
        cond_sizes = [44, 223, 446]
    elif dataset == 'Reddit':
        cond_sizes = [77, 153, 307]
    for cond_size in cond_sizes:
        for leanr_A in [0, 1]:
            dataset_rows = df[(df.iloc[:, 0] == dataset) & (df['Cond_Size'] == cond_size) & (df['Learn_A'] == leanr_A)]
            max_row = dataset_rows.loc[dataset_rows.iloc[:, 7].idxmax()].to_frame().T
            rows.append(max_row.index.values.tolist()[0])

print("-------------------------best resulst------------------------")
print(df.iloc[rows])