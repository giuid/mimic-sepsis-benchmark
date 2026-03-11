#!/bin/bash
export OMP_NUM_THREADS=4

run_eval_on_gpu() {
    local gpu=$1
    local model=$2
    local setup=$3
    local data_dir=$4
    local ckpt=$5
    
    export CUDA_VISIBLE_DEVICES=$gpu
    echo "Starting $model on $setup using GPU $gpu"
    
    cmd="python scripts/evaluate_downstream.py --model_name $model --setup $setup --data_dir $data_dir --device cuda:0 --batch_size 256"
    if [ ! -z "$ckpt" ]; then
        cmd="$cmd --checkpoint '$ckpt'"
    fi
    
    eval $cmd || echo "Failed $model on $setup"
}

run_eval_on_gpu 4 mean sota data/sota  & 
run_eval_on_gpu 5 locf sota data/sota  & 
run_eval_on_gpu 6 linear_interp sota data/sota  & 
run_eval_on_gpu 7 saits_vanilla sota data/sota 'outputs/sota/saits/random/D17_2026-02-19_15-11-55/checkpoints/best-epoch=48-val/loss=0.2752.ckpt' & 
wait
echo 'Batch finished, continuing...'

run_eval_on_gpu 4 saits_sapbert sota data/sota 'outputs/sota/saits/random/D17_2026-02-19_16-09-02/checkpoints/best-epoch=99-val/loss=0.3087.ckpt' & 
run_eval_on_gpu 5 brits sota data/sota 'outputs/sota/brits/random/D17_2026-02-19_15-22-24/checkpoints/best-epoch=49-val/loss=0.1612.ckpt' & 
run_eval_on_gpu 6 gpvae sota data/sota 'outputs/sota/gpvae/random/D17_2026-02-19_15-17-02/checkpoints/best-epoch=38-val/loss=39343.2227.ckpt' & 
run_eval_on_gpu 7 mrnn sota data/sota 'outputs/sota/mrnn/random/D17_2026-02-19_15-22-28/checkpoints/best-epoch=49-val/loss=0.0999.ckpt' & 
wait
echo 'Batch finished, continuing...'

run_eval_on_gpu 4 sssd sota data/sota 'outputs/sssd/random/2026-02-18_12-20-48/checkpoints/best-epoch=49-val/loss=0.1457.ckpt' & 
run_eval_on_gpu 5 timesfm sota data/sota 'outputs/sota/timesfm/random/D17_2026-02-23_12-57-43/checkpoints/best-epoch=09-val/loss=0.5012.ckpt' & 
run_eval_on_gpu 6 timesfm_sapbert sota data/sota 'outputs/sota/timesfm_sapbert/random/D17_2026-02-23_13-04-27/checkpoints/best-epoch=09-val/loss=2.0519.ckpt' & 
run_eval_on_gpu 7 mean handpicked data/handpicked  & 
wait
echo 'Batch finished, continuing...'

run_eval_on_gpu 4 locf handpicked data/handpicked  & 
run_eval_on_gpu 5 linear_interp handpicked data/handpicked  & 
run_eval_on_gpu 6 saits_vanilla handpicked data/handpicked 'outputs/handpicked/saits/random/D17_2026-02-18_15-40-17/checkpoints/best-epoch=95-val/loss=0.3362.ckpt' & 
run_eval_on_gpu 7 saits_sapbert handpicked data/handpicked 'outputs/handpicked/saits/random/D17_2026-02-18_14-16-23/checkpoints/best-epoch=96-val/loss=0.3244.ckpt' & 
wait
echo 'Batch finished, continuing...'

run_eval_on_gpu 4 brits handpicked data/handpicked 'outputs/handpicked/brits/random/D17_2026-02-19_15-22-22/checkpoints/best-epoch=49-val/loss=0.1717.ckpt' & 
run_eval_on_gpu 5 gpvae handpicked data/handpicked 'outputs/handpicked/gpvae/random/D17_2026-02-19_15-17-00/checkpoints/best-epoch=37-val/loss=39336.9219.ckpt' & 
run_eval_on_gpu 6 mrnn handpicked data/handpicked 'outputs/handpicked/mrnn/random/D17_2026-02-19_15-22-26/checkpoints/best-epoch=49-val/loss=0.1066.ckpt' & 
run_eval_on_gpu 7 sssd handpicked data/handpicked 'outputs/sssd/random/2026-02-18_12-20-48/checkpoints/best-epoch=49-val/loss=0.1457.ckpt' & 
wait
echo 'Batch finished, continuing...'

wait
echo 'All evaluations complete!'
