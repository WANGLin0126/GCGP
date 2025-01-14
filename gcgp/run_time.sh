#!/bin/bash

# 读取配置文件并逐行执行
while IFS= read -r line
do
  dataset=$(echo "$line" | awk '{print $1}')
  cond_ratio=$(echo "$line" | awk '{print $2}')
  ridge=$(echo "$line" | awk '{print $3}')
  k=$(echo "$line" | awk '{print $4}')
  learn_A=$(echo "$line" | awk '{print $5}')
  kernel=$(echo "$line" | awk '{print $6}')

  echo "$dataset $kernel $cond_ratio  $ridge $k $learn_A $epoch"
  python main.py --dataset $dataset --cond_ratio $cond_ratio --ridge $ridge --k $k --kernel $kernel --epochs 200 --learn_A $learn_A > outputs_time/${dataset}_kernel_${kernel}_size_${cond_ratio}_ridge_${ridge}_k_${k}_learn_A_${learn_A}.txt
done < config.txt









