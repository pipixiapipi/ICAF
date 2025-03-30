#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

dataset='tpo'
method='ICAF'
exp='r101'
split='2460'

config=configs/${dataset}.yaml
labeled_id_path=splits/$dataset/$split/labeled_group.txt
unlabeled_id_path=splits/$dataset/$split/unlabeled_group.txt
save_path=exp/$dataset/$method/$exp/$split

mkdir -p $save_path

python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    --use_env \
    $method.py \
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path --port $2 2>&1 | tee $save_path/$now.log
