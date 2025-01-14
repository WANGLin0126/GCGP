



for dataset in  Reddit; do 
    for cond_size in 77 153 307; do
        for ridge in 1e-3 1e-2 1e-1 1e0 1e1 ; do 
            for k in 1 2 3 4; do 
                for learn_A in 0 1; do
                    for norm in 0 1; do
                        for kernel in SGNK; do
                            echo "---------------------------"
                            echo ${dataset}_kernel_${kernel}_size_${cond_size}_ridge_${ridge}_k_${k}_learn_A_${learn_A}_norm_${norm}
                            echo "---------------------------"
                            python main.py --dataset $dataset --cond_size $cond_size --ridge $ridge --k $k --kernel $kernel --epochs 270 --learn_A $learn_A --norm $norm --batch_size 8000 --num_hops 0 \
                            > outputs/${dataset}_size_${cond_size}_ridge_${ridge}_k_${k}_learn_A_${learn_A}_norm_${norm}.txt
                        done
                    done
                done
            done
        done
    done
done

