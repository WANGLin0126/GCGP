for dataset in  ogbn-arxiv; do  
    for cond_size in 90 454 909; do
        for ridge in  1e-3 1e-2 1e-1 5e-1 1e0 5 1e1; do 
            for k in 1 2 3 4; do 
                for learn_A in 0 1; do
                    for norm in 0 1; do
                        echo ${dataset}_size_${cond_size}_ridge_${ridge}_k_${k}_learn_A_${learn_A}_norm_${norm}
                        python main.py --dataset $dataset --cond_size $cond_size --ridge $ridge --k $k --kernel SGNK --epochs 200 --learn_A $learn_A --norm $norm > outputs/${dataset}_size_${cond_size}_ridge_${ridge}_k_${k}_learn_A_${learn_A}_norm_${norm}.txt
                    done
                done
            done
        done
    done 
