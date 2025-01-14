'''
plot time-acc comparison curves
'''

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot
import pandas as pd
import matplotlib.ticker as ticker
from matplotlib.ticker import LogLocator

palette = pyplot.get_cmap('Set1')
colors = ['#3E4348','#5A9367','#C65146','#AF4034']

PATH = '/Users/wanglin/Documents/Projects/LargeGC/'     # 本地路径


labels = ['GC-SNTK', 'One-Step', 'GCGP(Our)']
datasets = ['Cora', 'Citeseer','Pubmed', 'Ogbn-arxiv', 'Reddit']
sizes = [70, 60, 30, 90, 77]

# 读取 Excel 文件的指定区域
file_path = PATH + 'results_LargeGC.xlsx'
sheet_name = 'Time_SGNK'
data = pd.read_excel(file_path, sheet_name=sheet_name, usecols='B:AE',skiprows=3, nrows=500)

# 将上述数据画为折线图
font_size = 10
figure_size = (5, 2)

for i in range(len(datasets)):
    dataset = datasets[i]
    size = sizes[i]
    plt.figure(i+1,figsize=figure_size)

    time1  = list(data.iloc[:, i*6 + 0])
    time2  = list(data.iloc[:, i*6 + 2])
    time3  = list(data.iloc[:, i*6 + 4])

    acc1  = list(data.iloc[:, i*6 + 1])
    acc2  = list(data.iloc[:, i*6 + 3])
    acc3  = list(data.iloc[:, i*6 + 5])

    idx1 = acc1.index(max(acc1))
    idx2 = acc2.index(max(acc2))
    idx3 = acc3.index(max(acc3))

    plt.plot(time1[:idx1], acc1[:idx1], label=labels[0], color=palette(1))
    plt.plot(time2[:idx2], acc2[:idx2], label=labels[1], color=palette(2))
    plt.plot(time3[:idx3], acc3[:idx3], label=labels[2], color=palette(0))

    plt.xlabel('Training Time(s)', fontsize=font_size)
    plt.ylabel('Test Acc(%)', fontsize=font_size)
    plt.grid()
    # plt.xscale('log')

    plt.legend(fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    plt.title(dataset + '(' + str(size) + ')', fontsize=font_size)
    plt.savefig(PATH + 'figures/time_acc1/' + dataset + '_time.pdf', bbox_inches='tight')
# plt.close()



font_size = 14
figure_size = (4, 2.8)
marker_size = 180
larger = 1.3

for i in range(len(datasets)):
    dataset = datasets[i]
    size = sizes[i]
    
    time1  = list(data.iloc[:, i*6 + 0])
    time2  = list(data.iloc[:, i*6 + 2])
    time3  = list(data.iloc[:, i*6 + 4])

    acc1  = list(data.iloc[:, i*6 + 1])
    acc2  = list(data.iloc[:, i*6 + 3])
    acc3  = list(data.iloc[:, i*6 + 5])

    idx1 = acc1.index(max(acc1))
    idx2 = acc2.index(max(acc2))
    idx3 = acc3.index(max(acc3))

    time1 = time1[idx1]
    time2 = time2[idx2]
    time3 = time3[idx3]
    
    values = [time1, time2, time3]  # 对应的值
    
    plt.figure(i+6,figsize=figure_size)

    plt.scatter(time1,acc1[idx1],label=labels[0],marker='*',s=marker_size*larger, color = '#519D9E')
    plt.scatter(time2,acc2[idx2],label=labels[1],marker='s',s=marker_size, color = colors[1])
    plt.scatter(time3,acc3[idx3],label=labels[2],marker='h',s=marker_size, color = colors[2])


    if dataset == 'Pubmed':
        plt.text(time1,acc1[idx1] - 3, f'{time2/time1:.1f}x', fontsize=font_size, ha='center')  # 标签稍微偏上
        plt.text(time2,acc2[idx2] - 3, f'{time2/time2:.1f}x', fontsize=font_size, ha='center')  # 标签稍微偏上
        plt.text(time3,acc3[idx3] + 1.5, f'{time2/time3:.1f}x', fontsize=font_size, ha='center')  # 标签稍微偏上
    elif dataset == 'Cora':
        plt.text(time1,acc1[idx1] - 3, f'{time2/time1:.1f}x', fontsize=font_size, ha='center')  # 标签稍微偏上
        plt.text(time2,acc2[idx2] - 3, f'{time2/time2:.1f}x', fontsize=font_size, ha='center')  # 标签稍微偏上
        plt.text(time3,acc3[idx3] + 1.5, f'{time2/time3:.1f}x', fontsize=font_size, ha='center')  # 标签稍微偏上
    else:
        plt.text(time1,acc1[idx1] + 1.5, f'{time2/time1:.1f}x', fontsize=font_size, ha='center')  # 标签稍微偏上
        plt.text(time2,acc2[idx2] + 1.5, f'{time2/time2:.1f}x', fontsize=font_size, ha='center')  # 标签稍微偏上
        plt.text(time3,acc3[idx3] + 1.5, f'{time2/time3:.1f}x', fontsize=font_size, ha='center')  # 标签稍微偏上

    plt.xlabel('Time cost(s)', fontsize=font_size)
    plt.ylabel('Acc(%)', fontsize=font_size)
    plt.xscale('log')
    log_locator = LogLocator(base=10.0, numticks=5)  # 设置最多显示 5 个主刻度
    plt.grid()
    plt.legend(fontsize = font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    ax = plt.gca()
    # formatter = ticker.ScalarFormatter(useMathText=True)
    # formatter.set_scientific(True)
    # formatter.set_powerlimits((-2, 3))     # 设置指数的显示范围
    locator = ticker.MaxNLocator(nbins=5)  # 设置刻度数量
    ax.yaxis.set_major_locator(locator)

    # 显示网格线
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    padding_factor = 0.5
    plt.xlim(time3 * (1 - padding_factor), time2 * (1 + padding_factor))
    min_acc = min(acc1[idx1],acc2[idx2],acc3[idx3])
    max_acc = max(acc1[idx1],acc2[idx2],acc3[idx3])
    padding_factor_y = 0.1
    plt.ylim(min_acc * (1 - padding_factor_y), max_acc * (1 + padding_factor_y))
    
    # plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=5)) # 设置 x 轴刻度数量
    plt.title(dataset + '(' + str(size) + ')', fontsize=font_size)
    plt.savefig(PATH + 'figures/time_acc1/' + dataset + '_star_time.pdf', bbox_inches='tight')

plt.show()