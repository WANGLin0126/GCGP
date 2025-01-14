for dataset in Cora Citeseer Pubmed Photo Computers; do
  for model in GCN SGC APPNP GAT SAGE Cheby MLP; do
    for cond_ratio in 0.25 0.5 1.0; do
      for learnA in 0 1; do
        for k in 1 2 3 4 5; do
          for kernel in SNTK; do
            echo "$dataset $model $cond_ratio $learnA $k"
            python generalization.py --model $model --dataset $dataset --cond_ratio $cond_ratio --k $k --epochs 400 --learnA $learnA > outputs_generalization/${dataset}_${kernel}_${model}_size_${cond_ratio}_k_${k}_learnA_${learnA}.txt
          done
        done
      done
    done
  done
done
