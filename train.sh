#!/bin/bash

# Definição de valores padrão para os parâmetros
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

# Função para exibir ajuda
usage() {
    echo "Uso: $0 [opções]"
    echo ""
    echo "Opções disponíveis:"
    echo "  --dataset <nome>          Nome do dataset (default: $DATASET)"
    echo "  --dataset_path <caminho>  Caminho do dataset (default: $DATASET_PATH)"
    echo "  --batch_size <num>        Tamanho do batch (default: $BATCH_SIZE)"
    echo "  --lr <valor>              Taxa de aprendizado (default: $LR)"
    echo "  --dropout <valor>         Taxa de dropout (default: $DROPOUT)"
    echo "  --hidden_dim <num>        Dimensão oculta (default: $HIDDEN_DIM)"
    echo "  --num_layers <num>        Número de camadas (default: $NUM_LAYERS)"
    echo "  --weight_decay <valor>    Decaimento de peso (default: $WEIGHT_DECAY)"
    echo "  --non_lin <função>        Função de ativação (default: $NON_LIN)"
    echo "  --device <tipo>           Dispositivo (cuda/cpu) (default: $DEVICE)"
    echo "  --num_epochs <num>        Número de épocas (default: $NUM_EPOCHS)"
    echo "  --output_path <caminho>   Caminho de saída dos modelos (default: $OUTPUT_PATH)"
    echo "  --method <metodo>         Método de treinamento (default: $METHOD)"
    echo "  --help                    Exibe esta mensagem e sai"
    exit 0
}

# Processamento dos argumentos
while [ "$#" -gt 0 ]; do
    case $1 in
        --dataset) DATASET="$2"; shift ;;
        --dataset_path) DATASET_PATH="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        --lr) LR="$2"; shift ;;
        --dropout) DROPOUT="$2"; shift ;;
        --hidden_dim) HIDDEN_DIM="$2"; shift ;;
        --num_layers) NUM_LAYERS="$2"; shift ;;
        --weight_decay) WEIGHT_DECAY="$2"; shift ;;
        --non_lin) NON_LIN="$2"; shift ;;
        --device) DEVICE="$2"; shift ;;
        --num_epochs) NUM_EPOCHS="$2"; shift ;;
        --output_path) OUTPUT_PATH="$2"; shift ;;
        --method) METHOD="$2"; shift ;;
        --help) usage ;;
        *) echo "Opção inválida: $1"; usage ;;
    esac
    shift
done

# Loop para executar com diferentes seeds
for SEED in 0
do
    python -m hmc.train.main \
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
