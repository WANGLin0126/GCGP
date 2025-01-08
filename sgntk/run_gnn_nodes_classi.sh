

for dataset in Computers; do
for model in GCN GAT APPNP GIN SAGE SGC; do
python gnn_nodes_classification.py  --dataset ${dataset} --model ${model}
done
done