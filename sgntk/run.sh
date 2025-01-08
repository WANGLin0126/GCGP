

# for dataset in  Cora Citeseer Pubmed Computers Photo; do  # Cora Citeseer Pubmed Computers Photo CS Physics
#     for cond_ratio in 0.25 0.5 1; do # 0.5 1
#         for ridge in 1e-3 1e-2 1e-1 5e-1 1e0 5 1e1; do #1e-2 1e-1 3e-1 6e-1 1e0 3 6 1e1
#             for k in 1 2 3 4 5; do # 4 5
#                 for learn_A in 1; do  # 0 1
#                     for kernel in SGNK; do
#                         # 打印输出
#                         echo ${dataset}_kernel_${kernel}_size_${cond_ratio}_ridge_${ridge}_k_${k}_learn_A_${learn_A}
#                         python main.py --dataset $dataset --cond_ratio $cond_ratio --ridge $ridge --k $k --kernel $kernel --epochs 200 --learn_A $learn_A > outputs/${dataset}_kernel_${kernel}_size_${cond_ratio}_ridge_${ridge}_k_${k}_learn_A_${learn_A}.txt
#                     done
#                 done
#             done
#         done
#     done
# done


# for dataset in  Cora ; do  # Cora Citeseer Pubmed Computers Photo CS Physics
#     for cond_ratio in 0.25 ; do # 0.5 1
#         for ridge in 1e-3 ; do #1e-2 1e-1 3e-1 6e-1 1e0 3 6 1e1
#             for k in 5; do # 4 5
#                 for learn_A in 0; do  # 0 1
#                     for kernel in SGNK; do
#                         # 打印输出
#                         echo ${dataset}_kernel_${kernel}_size_${cond_ratio}_ridge_${ridge}_k_${k}_learn_A_${learn_A}
#                         python main.py --dataset $dataset --cond_ratio $cond_ratio --ridge $ridge --k $k --kernel $kernel --epochs 200 --learn_A $learn_A > outputs/${dataset}_kernel_${kernel}_size_${cond_ratio}_ridge_${ridge}_k_${k}_learn_A_${learn_A}.txt
#                     done
#                 done
#             done
#         done
#     done
# done


for dataset in  Cora Citeseer Pubmed Computers Photo; do  # Cora Citeseer Pubmed Computers Photo CS Physics
    for cond_ratio in 0.25 0.5 1; do # 0.5 1
        for ridge in 1e-3 1e-2 1e-1 5e-1 1e0 5 1e1; do #1e-2 1e-1 3e-1 6e-1 1e0 3 6 1e1
            for k in 1 2 3 4 5; do # 4 5
                for learn_A in 0; do  # 0 1
                    for kernel in SNTK; do
                        # 打印输出
                        echo ${dataset}_kernel_${kernel}_size_${cond_ratio}_ridge_${ridge}_k_${k}_learn_A_${learn_A}
                        python main.py --dataset $dataset --cond_ratio $cond_ratio --ridge $ridge --k $k --kernel $kernel --epochs 200 --learn_A $learn_A > outputs/${dataset}_kernel_${kernel}_size_${cond_ratio}_ridge_${ridge}_k_${k}_learn_A_${learn_A}.txt
                    done
                done
            done
        done
    done
done