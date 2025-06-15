#!/bin/bash


while IFS= read -r line
do
  dataset=$(echo "$line" | awk '{print $1}')
  cond_size=$(echo "$line" | awk '{print $2}')
  ridge=$(echo "$line" | awk '{print $3}')
  k=$(echo "$line" | awk '{print $4}')
  learn_A=$(echo "$line" | awk '{print $5}')
  norm=$(echo "$line" | awk '{print $6}')
  epoch=$(echo "$line" | awk '{print $7}')

  echo " $dataset $cond_size  $ridge $k $learn_A  $norm"
  # python main.py --dataset $dataset --cond_size $cond_size --ridge $ridge --k $k --kernel SGTK --epochs $epoch --learn_A $learn_A --norm $norm --iter 1 --save True
  python main.py --dataset $dataset --cond_size $cond_size --ridge $ridge --k $k --kernel SGNK --epochs 270 --learn_A $learn_A --norm $norm > save/${dataset}_size_${cond_size}_ridge_${ridge}_k_${k}_learn_A_${learn_A}_norm_${norm}.txt
done < config.txt

