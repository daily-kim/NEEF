# NEEF(A Node Embedding Enhancement Framework to Mitigate Cold Start Problem in GNN)

# OGB(Open Graph Benchmark)

- Dataset will downloaded automatically.

# Docker Container

- Docker container use cgmc project directory as volume
- File change will be apply directly to file in docker container

# Requirements

- The requirements are installed during the Docker build process.
- Tested combination: Python 3.8.5 + PyTorch 1.13.1 + DGL 1.1.0+cu116 + OGB 1.3.6

# Usages

---

## Setting

`make up` : build docker image and start docker container

## Train and evaluation[GCN,SAGE]

For ogbl-ddi

```
python main.py --dataset ogbl-ddi  --model {SAGE,GCN} --hidden_channels 256 --batch_size 65536 --lr 0.005 --epochs 200 --eval_steps 5 --runs 5 --use_warmer --warmup_core 5 --warmup_epochs 1000 --warmup_channels 256 --warmup_model GCN
```

For ogbl-collab

```
python main.py --dataset ogbl-collab  --model {SAGE,GCN} --use_valedges_as_input --num_layers 3 --hidden_channels 256--batch_size 65536 --lr 0.001 --epochs 400 --eval_steps 5 --runs 5 --use_warmer --warmup_core 5 --warmup_epochs 1000 --warmup_type 2 --warmup_channels 256 --warmup_model GCN
```

```
python seal.py --dataset ogbl-ddi --ratio_per_hop 0.2 --use_edge_weight --eval_steps 1 --dynamic_val --dynamic_test --use_feature --use_warmup --warmup_core 5 --eval_hits_K {20,50,100} --runs 5
```

\* 'warmer' and 'warmup' are conceptually equivalent to the NEEF framework. \* If node feature exist, '--use_pretrain' can pass NEEF traning step.

---

## Train and evaluation[SEAL]

For ogbl-ddi

```
python seal.py --dataset ogbl-ddi --ratio_per_hop 0.2 --use_edge_weight --eval_steps 1 --dynamic_val --dynamic_test --use_feature --use_warmup --warmup_core 5 --eval_hits_K {20,50,100} --runs 5
```

For ogbl-collab

```
python seal.py --dataset ogbl-collab --ratio_per_hop 0.2 --use_edge_weight --eval_steps 1 --dynamic_val --dynamic_test --use_feature --use_warmup --warmup_core 5 --eval_hits_K {20,50,100} --runs 5
```

## \* For SEAL, the creation of enhanced node feture should be done first.

## Get enhanced node feature only

\* Save NEEF node embedding to the each dataset path. \* Using for SEAL model

For ogbl-ddi

```
python get_feat_only.py --dataset ogbl-ddi --model {SAGE,GCN} --hidden_channels 256 --batch_size 65536 --lr 0.005 --epochs 200 --eval_steps 5 --runs 5 --use_warmer --warmup_core 5 --warmup_epochs 1000 --warmup_channels 256 --warmup_model GCN
```

For ogbl-collab

```
python get_feat_only.py --dataset ogbl-collab --model {SAGE,GCN} --use_valedges_as_input --num_layers 3 --hidden_channels 256--batch_size 65536 --lr 0.001 --epochs 400 --eval_steps 5 --runs 5 --use_warmer --warmup_core 5 --warmup_epochs 1000 --warmup_type 2 --warmup_channels 256 --warmup_model GCN
```

## Evaluation seperatly by edge type

- For check performance difference by edge types, add '--eval_seperate' argument
