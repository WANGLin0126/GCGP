"""
plot the time-accuracy curve for each dataset in the outputs_time folder
"""

import os
import numpy as np
import pandas as pd
# from openpyxl import load_workbook
import matplotlib.pyplot as plt
from matplotlib import pyplot
import matplotlib.ticker as ticker
import glob


PATH = '/Users/wanglin/Documents/Projects/LargeGC/'

# 定义一个函数用于读取并处理数据
def read_data(file_name, dataset):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    data = []
    if dataset == 'ogbn-arxiv':
        select_lines = lines[620:820]
    elif dataset == 'Flickr':
        select_lines = lines[473:623]
    else:
        select_lines = lines[22:222] 

    for line in select_lines:    # for sgntk_flickr/outputs_time/ results
        if contains_number(line):
            # 去除首尾的方括号和空白字符
            line = line.strip().strip('[').strip(']')
            # 跳过空行
            if not line:
                continue
            # 用空格或逗号分隔数据
            elements = line.replace(',', '').split()
            # 将字符串转换为浮点数
            row = [float(elem) for elem in elements]
            data.append(row)
    return np.array(data)

def contains_number(s):
    return any(char.isdigit() for char in s)



# ['planetoid', 'ogbn-arxiv', 'Flickr', 'Reddit']:
for dataset in ['planetoid']:

    if dataset == 'ogbn-arxiv':
        folder_path = PATH + 'sgntk_ogb/outputs_time/'
    elif dataset == 'Flickr':
        folder_path = PATH + 'sgntk_flickr/outputs_time/'
    elif dataset == 'planetoid':
        folder_path = PATH + 'sgntk_planetoid/outputs_time/'




    # 列出文件夹中的所有文件和目录
    items = sorted(os.listdir(folder_path))
    # 去除第一个文件
    items = items[1:]
    # items 
    font_size = 16
    figure_size = (4, 2.5)
    palette = pyplot.get_cmap('Set1')
    i = 0

    for dataset in ['Cora', 'Citeseer', 'Pubmed','Photo', 'Computers']:
        sizes = [140, 120, 60, 160, 200]
        for cond_ratio in [0.25,0.5,1]:

            item_sgnk = dataset + '_kernel_SGNK_size_'+ str(cond_ratio)
            item_sntk = dataset + '_kernel_SNTK_size_'+ str(cond_ratio)
            # 构建完整的路径
            item_sgnk_path = os.path.join(folder_path, f'*{item_sgnk}*')
            item_sntk_path = os.path.join(folder_path, f'*{item_sntk}*')
            
            # 查找匹配的文件
            file_sgnk = glob.glob(item_sgnk_path)
            file_sntk = glob.glob(item_sntk_path)


            # # 判断是否为文件
            # if os.path.isfile(file_sgnk[0]):
            #     print(f"文件：{file_sgnk[0]}")

            # if os.path.isfile(file_sntk[0]):
            #     print(f"文件：{file_sntk[0]}")

            data_sgnk = read_data(file_sgnk[0], dataset = dataset)
            data_sntk = read_data(file_sntk[0], dataset = dataset)
            
            zero_row = np.zeros((1, data_sntk.shape[1]))
            data_sgnk = np.vstack((zero_row, data_sgnk))
            data_sntk = np.vstack((zero_row, data_sntk))

            ind_sgnk = np.argmax(data_sgnk[:,1])
            ind_sntk = np.argmax(data_sntk[:,1])

            # 将每组数据画成一个曲线图并保存
            plt.figure(i,figsize=figure_size)
            plt.plot(data_sntk[:,3][0:ind_sntk+1], data_sntk[:,1][0:ind_sntk+1], color=palette(1), label='GC-SNTK')
            plt.plot(data_sgnk[:,3][0:ind_sgnk+1], data_sgnk[:,1][0:ind_sgnk+1], color=palette(0), label='GCGP(Our)')

            print(f"{dataset} {cond_ratio} {data_sntk[:,3][ind_sntk+1]:.4f}, {data_sgnk[:,3][ind_sgnk+1]:.4f}, {data_sntk[:,3][ind_sntk+1]/data_sgnk[:,3][ind_sgnk+1]:.1f}")

            plt.xlabel('Training Time (s)', fontsize=font_size)
            plt.ylabel('Test Acc(%)', fontsize=font_size)
            plt.grid()

            plt.legend(fontsize=font_size, loc='lower right')
            plt.xticks(fontsize=font_size)
            plt.yticks(fontsize=font_size)
            plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
            plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True))
            # plt.title('Pubmed (30)', fontsize=font_size)
            plt.title(dataset + ' (' + str(int(cond_ratio*sizes[i])) + ')', fontsize=font_size)
            plt.savefig( PATH + 'figures/time_acc2/'+ dataset +'_'+ str(cond_ratio)+'.pdf', bbox_inches='tight')

                # plt.savefig( PATH + 'figures/time_acc/'+ item.split('_')[0]+'_'+ item.split('_')[2]+'_'+item.split('_')[4][0]+'.pdf', bbox_inches='tight')
            # else:
            #     plt.savefig( PATH + 'figures/time_acc/'+ item.split('_')[0]+'_'+ item.split('_')[2]+'_'+item.split('_')[9]+'.pdf', bbox_inches='tight')
            plt.close()
        i += 1


