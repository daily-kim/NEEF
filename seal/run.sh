python seal.py --dataset ogbl-ddi --ratio_per_hop 0.2 --use_edge_weight --eval_steps 1 --dynamic_val --dynamic_test --use_feature --use_warmup --warmup_core 5 --eval_hits_K 50 --runs 3
python seal.py --dataset ogbl-ddi --ratio_per_hop 0.2 --use_edge_weight --eval_steps 1 --dynamic_val --dynamic_test --use_feature --use_warmup --warmup_core 5 --eval_hits_K 100 --runs 3
python seal.py --dataset ogbl-ddi --ratio_per_hop 0.2 --use_edge_weight --eval_steps 1 --dynamic_val --dynamic_test --use_feature --use_warmup --warmup_core 5 --eval_hits_K 20 --runs 3

