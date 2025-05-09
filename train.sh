#!/bin/bash


# Lista de datasets
datasets=('cellcycle_GO' 'derisi_GO' 'eisen_GO' 'expr_GO' 'gasch1_GO'
          'gasch2_GO' 'seq_GO' 'spo_GO' 'cellcycle_FUN' 'derisi_FUN'
          'eisen_FUN' 'expr_FUN' 'gasch1_FUN' 'gasch2_FUN' 'seq_FUN' 'spo_FUN')

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
EPOCHS=400
OUTPUT_PATH="/home/bruno/storage/models/gofun"
METHOD="local"
HPO="false"

# Function to display help
usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Available options:"
    echo "  --dataset <name>          Dataset name (default: $DATASET)"
    echo "  --dataset_path <path>     Dataset path (default: $DATASET_PATH)"
    echo "  --batch_size <num>        Batch size (default: $BATCH_SIZE)"
    echo "  --lr <value>              Learning rate (default: $LR)"
    echo "  --dropout <value>         Dropout rate (default: $DROPOUT)"
    echo "  --hidden_dim <num>        Hidden dimension (default: $HIDDEN_DIM)"
    echo "  --num_layers <num>        Number of layers (default: $NUM_LAYERS)"
    echo "  --weight_decay <value>    Weight decay (default: $WEIGHT_DECAY)"
    echo "  --non_lin <function>      Activation function (default: $NON_LIN)"
    echo "  --device <type>           Device (cuda/cpu) (default: $DEVICE)"
    echo "  --epochs <num>        Number of epochs (default: $EPOCHS)"
    echo "  --output_path <path>      Output path for models (default: $OUTPUT_PATH)"
    echo "  --method <method>         Training method (default: $METHOD)"
    echo "  --hpo <true/false>       Hyperparameter optimization (default: $HPO)"
    echo "  --help                    Display this message and exit"
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
        --epochs) EPOCHS="$2"; shift ;;
        --output_path) OUTPUT_PATH="$2"; shift ;;
        --method) METHOD="$2"; shift ;;
        --hpo) HPO="$2"; shift ;;
        --help) usage ;;
        *) echo "Invalid option: $1"; usage ;;
    esac
    shift
done

if [ "$DATASET" = "all" ]; then
    # Loop sobre os datasets
    # Número máximo de processos em paralelo
    MAX_JOBS=6
    current_jobs=0
    for dataset in "${datasets[@]}"; do
        # Loop para executar com diferentes seeds
        for SEED in 0
        do
            echo "Running with arguments: --datasets $DATASET --dataset_path $DATASET_PATH --batch_size $BATCH_SIZE --lr $LR --dropout $DROPOUT --hidden_dim $HIDDEN_DIM --num_layers $NUM_LAYERS --weight_decay $WEIGHT_DECAY --non_lin $NON_LIN --device $DEVICE --epochs $EPOCHS --seed $SEED --output_path $OUTPUT_PATH --method $METHOD"
            # Controle de processos simultâneos
            PYTHONPATH=src poetry run python -m hmc.train.main \
                --datasets "$dataset" \
                --dataset_path "$DATASET_PATH" \
                --batch_size "$BATCH_SIZE" \
                --lr "$LR" \
                --dropout "$DROPOUT" \
                --hidden_dim "$HIDDEN_DIM" \
                --num_layers "$NUM_LAYERS" \
                --weight_decay "$WEIGHT_DECAY" \
                --non_lin "$NON_LIN" \
                --device "$DEVICE" \
                --epochs "$EPOCHS" \
                --seed "$SEED" \
                --output_path "$OUTPUT_PATH" \
                --method "$METHOD" \
                --hpo "$HPO" &  # Executa em segundo plano


            ((current_jobs++))

            # Se o número de processos em execução atingir MAX_JOBS, esperar antes de continuar
            if (( current_jobs >= MAX_JOBS )); then
                wait -n  # Aguarda pelo menos um processo terminar antes de continuar
                ((current_jobs--))  # Reduz o contador de processos em execução
            fi

        done
    done
else
    for SEED in 0
        do
            echo "Running with arguments: --datasets $DATASET --dataset_path $DATASET_PATH --batch_size $BATCH_SIZE --lr $LR --dropout $DROPOUT --hidden_dim $HIDDEN_DIM --num_layers $NUM_LAYERS --weight_decay $WEIGHT_DECAY --non_lin $NON_LIN --device $DEVICE --num_epochs $EPOCHS --seed $SEED --output_path $OUTPUT_PATH --method $METHOD"
            # Controle de processos simultâneos
            PYTHONPATH=src poetry run python -m hmc.train.main \
                --datasets "$DATASET" \
                --dataset_path "$DATASET_PATH" \
                --batch_size $BATCH_SIZE \
                --lr $LR \
                --dropout $DROPOUT \
                --hidden_dim $HIDDEN_DIM \
                --num_layers $NUM_LAYERS \
                --weight_decay $WEIGHT_DECAY \
                --non_lin "$NON_LIN" \
                --device "$DEVICE" \
                --epochs $EPOCHS \
                --seed $SEED \
                --output_path "$OUTPUT_PATH" \
                --method "$METHOD" \
                --hpo "$HPO" &  # Executa em segundo plano
    done
fi
TRAIN_PID=$!
# Aguarda a execução de todos os processos restantes
trap "kill $TRAIN_PID" SIGINT SIGTERM
wait

echo "All experiments completed!"
