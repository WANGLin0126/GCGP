#!/bin/bash

# 读取配置文件并逐行执行
# while IFS= read -r line
# do
#   dataset=$(echo "$line" | awk '{print $1}')
#   cond_size=$(echo "$line" | awk '{print $2}')
#   ridge=$(echo "$line" | awk '{print $3}')
#   k=$(echo "$line" | awk '{print $4}')
#   learn_A=$(echo "$line" | awk '{print $5}')
#   norm=$(echo "$line" | awk '{print $6}')
#   epoch=$(echo "$line" | awk '{print $7}')

#   echo " $dataset $cond_size  $ridge $k $learn_A  $norm"
#   python main.py --dataset $dataset --cond_size $cond_size --ridge $ridge --k $k --kernel SGNK --epochs 200 --learn_A $learn_A --norm $norm > outputs_time/${dataset}_size_${cond_size}_ridge_${ridge}_k_${k}_learn_A_${learn_A}_norm_${norm}.txt
#   # python main.py --dataset $dataset --cond_size $cond_size --ridge $ridge --k $k --kernel SGNK --epochs 200 --learn_A $learn_A --norm $norm --save 1 --iter 1
# done < config.txt

for kernel in SNTK
do
for cond_size in 90 454 909
do
for learn_A in 0
do
for norm in 0
do
for dataset in  ogbn-arxiv
do
for ridge in 1e-3
do
for k in 2
do
  echo " $dataset $cond_size  $ridge $k $learn_A  $norm"
  python main.py --dataset ${dataset} --cond_size ${cond_size} --ridge ${ridge} --K ${k} --kernel ${kernel} --epochs 200 > outputs_time/${dataset}_size_${cond_size}_ridge_${ridge}_k_${k}_learn_A_${learn_A}_morm_${norm}_kernel_${kernel}.txt
done
done
done
done
done
done
done