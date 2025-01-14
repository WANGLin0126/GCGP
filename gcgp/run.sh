

for dataset in  Cora Citeseer Pubmed Computers Photo; do 
    for cond_ratio in 0.25 0.5 1; do
        for ridge in 1e-3 1e-2 1e-1 5e-1 1e0 5 1e1; do 
            for k in 1 2 3 4 5; do
                for learn_A in 0 1; do 
                    for kernel in SGNK; do
                        echo ${dataset}_kernel_${kernel}_size_${cond_ratio}_ridge_${ridge}_k_${k}_learn_A_${learn_A}
                        python main.py --dataset $dataset --cond_ratio $cond_ratio --ridge $ridge --k $k --kernel $kernel --epochs 200 --learn_A $learn_A > outputs/${dataset}_kernel_${kernel}_size_${cond_ratio}_ridge_${ridge}_k_${k}_learn_A_${learn_A}.txt
                    done
                done
            done
        done
    done
done