#!/bin/bash
TARGET_DIR="/home/guido/Code/charite/baselines/outputs/mimic4/saits/random/2026-02-24_14-26-09/checkpoints"
KGI_LOG="/home/guido/Code/charite/baselines/logs/saits_kgi_sota.log"
VANILLA_LOG="/home/guido/Code/charite/baselines/logs/saits_vanilla_baseline_sota.log"

echo "Smart Monitoring KGI vs Vanilla in: $TARGET_DIR"

while true; do
  find "$TARGET_DIR" -maxdepth 1 -name "*.ckpt" ! -name "*_KGI.ckpt" ! -name "*_Vanilla.ckpt" | while read file; do
    
    # Estrae il valore decimale della loss dal nome file (es. 0.4619)
    loss_val=$(echo "$file" | grep -o 'loss=[0-9]\+\.[0-9]\+' | cut -d'=' -f2)
    
    if [ -z "$loss_val" ]; then
        continue
    fi

    # Cerca quel preciso valore di loss all'interno dei file di log di WaNDB/PyTorch
    is_kgi=$(grep -o "$loss_val" "$KGI_LOG" | head -n 1)
    is_vanilla=$(grep -o "$loss_val" "$VANILLA_LOG" | head -n 1)
    
    # Pulisce eventuali suffissi temporanei del demone precedente
    clean_name=$(echo "$file" | sed 's/_disambiguated_[0-9]*//g')
    base_clean="${clean_name%.ckpt}"
    
    if [ -n "$is_kgi" ]; then
        mv "$file" "${base_clean}_KGI.ckpt"
        echo "Mapped successfully -> KGI: $(basename "${base_clean}_KGI.ckpt")"
    elif [ -n "$is_vanilla" ]; then
        mv "$file" "${base_clean}_Vanilla.ckpt"
        echo "Mapped successfully -> Vanilla: $(basename "${base_clean}_Vanilla.ckpt")"
    fi
  done
  sleep 10
done
