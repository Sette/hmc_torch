#!/bin/bash

# Definição das variáveis fixas
DATASET="seq_FUN"
DATASET_PATH="/home/bruno/storage/data/datasets"
BATCH_SIZE=4
LR=1e-4
DROPOUT=0.7
HIDDEN_DIM=2000
NUM_LAYERS=3
WEIGHT_DECAY=1e-5
NON_LIN="relu"
DEVICE="cuda"
NUM_EPOCHS=13
OUTPUT_PATH="/home/bruno/storage/models/gofun"
METHOD="global"

# Loop para executar com diferentes seeds
for SEED in 0 1 2 3 4 5 6 7 8 9
do
    python -m hmc.model.train \
        --datasets "$DATASET" \
        --dataset_path "$DATASET_PATH" \
        --batch_size "$BATCH_SIZE" \
        --lr "$LR" \
        --dropout "$DROPOUT" \
        --hidden_dim "$HIDDEN_DIM" \
        --num_layers "$NUM_LAYERS" \
        --weight_decay "$WEIGHT_DECAY" \
        --non_lin "$NON_LIN" \
        --device "$DEVICE" \
        --num_epochs "$NUM_EPOCHS" \
        --seed "$SEED" \
        --output_path "$OUTPUT_PATH" \
        --method "$METHOD" &
done

# Aguarda todos os processos terminarem antes de finalizar o script
wait
