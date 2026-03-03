#!/bin/bash
# ============================================================
#  DynamicKGI Training Run — 2026-02-27
#  Trains SAITS, MRNN, BRITS with the new DynamicKnowledgeInjector
#  GPUs: 4, 5, 6, 7
# ============================================================

GPUS=(4 5 6 7)

# 4 jobs — one per GPU
# GPU 4: SAITS + DynamicKGI
# GPU 5: MRNN  + DynamicKGI
# GPU 6: BRITS + DynamicKGI
# GPU 7: SAITS Vanilla (control + new alpha exploration)
CONFIGS=(
    "model=saits ++model.embedding_type=sapbert ++model.use_kgi=True ++model.kgi_embedding_file=medbert_relation_embeddings_generic.pkl ++model.alpha_joint=0.01"
    "model=mrnn  ++model.use_kgi=True ++model.kgi_embedding_file=medbert_relation_embeddings_generic.pkl ++model.alpha_joint=0.01"
    "model=brits ++model.use_kgi=True ++model.kgi_embedding_file=medbert_relation_embeddings_generic.pkl ++model.alpha_joint=0.01"
    "model=saits ++model.embedding_type=vanilla ++model.use_kgi=False ++model.alpha_joint=0.01"
)

NAMES=(
    "dkgi_saits_kgi"
    "dkgi_mrnn_kgi"
    "dkgi_brits_kgi"
    "dkgi_saits_vanilla"
)

echo "============================================================"
echo "  DynamicKnowledgeInjector Joint Training"
echo "  Models: SAITS-KGI | MRNN-KGI | BRITS-KGI | SAITS-Vanilla"
echo "  GPUs:   4          | 5        | 6         | 7"
echo "============================================================"

for i in "${!CONFIGS[@]}"; do
    CFG="${CONFIGS[$i]}"
    NAME="${NAMES[$i]}"
    GPU="${GPUS[$i]}"

    echo "Launching [$NAME] on GPU $GPU..."

    # Kill any old session with the same name
    tmux kill-session -t "$NAME" 2>/dev/null

    # Launch detached tmux session
    tmux new-session -d -s "$NAME" \
        "cd /home/guido/Code/charite/baselines && \
         CUDA_VISIBLE_DEVICES=$GPU python scripts/training/train_joint.py \
         $CFG \
         data.processed_dir=data/sota \
         trainer.devices=1 \
         trainer.accelerator=gpu \
         trainer.max_epochs=40; \
         echo ''; echo '=== TRAINING COMPLETE: $NAME ==='; \
         echo 'Press Enter to close...'; read"
done

echo ""
echo "All 4 DynamicKGI training jobs dispatched!"
echo ""
echo "Monitor with:"
echo "-------------------------------------------"
for name in "${NAMES[@]}"; do
    echo "  tmux a -t $name"
done
echo "-------------------------------------------"
echo "(Ctrl+b, then d to detach from a session)"
echo ""
echo "GPU status: watch -n5 nvidia-smi"
