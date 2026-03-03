#!/bin/bash
# Recovery script for interrupted training runs - FIXED HYDRA SYNTAX

# GPU 7: SAITS Semantic GNN
tmux new-window -t training_resume:1 -n "SAITS_SEM_GNN" "CUDA_VISIBLE_DEVICES=7 python train.py data.processed_dir=data/sota model=saits model.embedding_type=vanilla +model.use_kgi=True +model.use_graph_layer=True +model.kgi_embedding_file=medbert_relation_embeddings_semantic.pkl +checkpoint=outputs/mimic4/saits/random/2026-02-25_15-55-32/checkpoints/vanilla_KGI/last.ckpt"

# GPU 6: SAITS Semantic NoGNN
tmux new-window -t training_resume:2 -n "SAITS_SEM_NOGNN" "CUDA_VISIBLE_DEVICES=6 python train.py data.processed_dir=data/sota model=saits model.embedding_type=vanilla +model.use_kgi=True +model.use_graph_layer=False +model.kgi_embedding_file=medbert_relation_embeddings_semantic.pkl +checkpoint=outputs/mimic4/saits/random/2026-02-25_15-55-26/checkpoints/vanilla_KGI/last.ckpt"

# GPU 5: SAITS Generic GNN
tmux new-window -t training_resume:3 -n "SAITS_GEN_GNN" "CUDA_VISIBLE_DEVICES=5 python train.py data.processed_dir=data/sota model=saits model.embedding_type=vanilla +model.use_kgi=True +model.use_graph_layer=True +model.kgi_embedding_file=medbert_relation_embeddings_generic.pkl +checkpoint=outputs/mimic4/saits/random/2026-02-25_15-55-22/checkpoints/vanilla_KGI/last.ckpt"

# GPU 4: BRITS KGI
tmux new-window -t training_resume:4 -n "BRITS_KGI" "CUDA_VISIBLE_DEVICES=4 python train.py data.processed_dir=data/sota model=brits +model.use_kgi=True +checkpoint=outputs/mimic4/brits/random/2026-02-25_16-41-06/checkpoints/default_KGI/last.ckpt"

# GPU 3: MRNN KGI
tmux new-window -t training_resume:5 -n "MRNN_KGI" "CUDA_VISIBLE_DEVICES=3 python train.py data.processed_dir=data/sota model=mrnn +model.use_kgi=True +checkpoint=outputs/mimic4/mrnn/random/2026-02-25_16-41-11/checkpoints/default_KGI/last.ckpt"

echo "All 5 training runs resumed in tmux session 'training_resume' with corrected Hydra syntax."
