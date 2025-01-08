# parser.add_argument('--dataset', type=str, default="ogbn-arxiv", help='name of dataset (default: ogbn-arxiv)')
# parser.add_argument('--cond_size', type=int, default=90, help='condensed ratio of the training set (default: 0.5, the condened set is 0.5*training set)')
# parser.add_argument('--ridge', type=float, default=1e-5, help='ridge parameter of KRR (default: 1e-3)')
# parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 100)')
# parser.add_argument('--lr_X', type=float, default=1e-2, help='learning rate (default: 0.005)')
# parser.add_argument('--lr_A', type=float, default=1e-2, help='learning rate (default: 0.005)')
# parser.add_argument('--k', type=int, default=3, help='the convolutiona times of the dataset (default: 2)')
# parser.add_argument('--K', type=int, default=3, help='number of aggr in SGTK (default: 2)')
# parser.add_argument('--L', type=int, default=1, help='the number of layers after each aggr (default: 2)')
# parser.add_argument('--learn_A', type=bool, default=False, help='whether to learn the adjacency matrix')
# parser.add_argument('--norm', type=bool, default=False, help='whether to normalize the features')
# parser.add_argument('--set_seed', type=bool, default=False, help='whether to set seed')
# parser.add_argument('--seed', type=int, default=5, help='setup the random seed (default: 5)')
# parser.add_argument('--iter', type=int, default=5, help='iteration times (default: 3)')
# parser.add_argument('--kernel', type=str, default='SGTK', help='kernel method in KRR [SGTK, SGNK] (default: SGTK)')
# parser.add_argument('--split_method', type=str, default='kmeans', help='split method of the test set [kmeans,none] (default: kmeans)')
# parser.add_argument('--save', type=bool, default=False, help='whether to save the results')


# for dataset in  ogbn-arxiv; do  # Cora Citeseer Pubmed Computers Photo
#     for cond_size in 90 454 909; do
#         for ridge in  1e-4 1e-3 1e-2 1e-1; do #1e-2 1e-1 3e-1 6e-1 1e0 3 6 1e1
#             for k in 1 2 3 4 5; do # 4 5
#                 for learn_A in 0 1; do
#                     for norm in 0 1; do
#                         # 打印输出
#                         # echo ${dataset}_size_${cond_ratio}_ridge_${ridge}_k_${k}_learn_A_${learn_A}
#                         python main.py --dataset $dataset --cond_size $cond_size --ridge $ridge --k $k --kernel SGTK --epochs 200 --learn_A $learn_A --norm $norm > outputs/${dataset}_size_${cond_size}_ridge_${ridge}_k_${k}_learn_A_${learn_A}_norm_${norm}.txt
#                     done
#                 done
#             done
#         done
#     done
# done



for dataset in  ogbn-arxiv ; do  # Cora Citeseer Pubmed Computers Photo
    for cond_size in 90 454 909; do
        for ridge in 1e-3 1e-2 1e-1 5e-1 1e0 5 1e1; do
            for k in 1 2 3 4 5; do
                for learn_A in 0 1; do
                    for norm in 0 1; do
                        # 打印输出
                        echo ${dataset}_size_${cond_ratio}_ridge_${ridge}_k_${k}_learn_A_${learn_A}
                        python main.py --dataset $dataset --cond_size $cond_size --ridge $ridge --k $k --kernel SGNK --epochs 200 --train_batch_size 2000 --learn_A $learn_A --norm $norm > outputs/${dataset}_size_${cond_size}_ridge_${ridge}_k_${k}_learn_A_${learn_A}_norm_${norm}.txt
                    done
                done
            done
        done
    done
done



# for dataset in  ogbn-arxiv; do  # Cora Citeseer Pubmed Computers Photo
#     for cond_size in 90; do
#         for ridge in 1e-1; do #1e-2 1e-1 3e-1 6e-1 1e0 3 6 1e1
#             for k in 2 3 4 5; do
#                 for learn_A in 0; do
#                     for norm in 0 1; do
#                         # 打印输出
#                         # echo ${dataset}_size_${cond_ratio}_ridge_${ridge}_k_${k}_learn_A_${learn_A}
#                         python main.py --dataset $dataset --cond_size $cond_size --ridge $ridge --k $k --kernel SGTK --epochs 200 --learn_A $learn_A --norm $norm > outputs/${dataset}_size_${cond_size}_ridge_${ridge}_k_${k}_learn_A_${learn_A}_norm_${norm}.txt
#                     done
#                 done
#             done
#         done
#     done
# done
