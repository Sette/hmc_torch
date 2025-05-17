#!/bin/bash


# Lista de datasets
datasets=('cellcycle_GO' 'derisi_GO' 'eisen_GO' 'expr_GO' 'gasch1_GO'
          'gasch2_GO' 'seq_GO' 'spo_GO' 'cellcycle_FUN' 'derisi_FUN'
          'eisen_FUN' 'expr_FUN' 'gasch1_FUN' 'gasch2_FUN' 'seq_FUN' 'spo_FUN')

# Definição de valores padrão para os parâmetros
DATASET="seq_FUN"
DATASET_PATH="/home/bruno/storage/data/datasets"
BATCH_SIZE=4
NON_LIN="relu"
DEVICE="cuda"
EPOCHS=400
OUTPUT_PATH="/home/bruno/storage/models/gofun"
METHOD="local"
HPO="false"
HIDDEN_DIMS="64 64 128 128 256 128"
LR_VALUES="2.089343433609757e-06 1.2001828778028627e-06 2.177246665700762e-06 6.3578025240712225e-06 3.0989798384334672e-06 1.5453850689526738e-05"
DROPOUT_VALUES="0.740386914297677 0.5500644363368916 0.753285175114529 0.4901667669873963 0.4134921804354519 0.4432743726722958"
NUM_LAYERS_VALUES="2 2 1 1 1 2"
WEIGHT_DECAY_VALUES="3.157085231104779e-05 1.621087221226643e-06 0.004736849157519908 4.523000925403428e-05 1.7509182070548764e-06 0.0018886373877888352"
export PYTHONPATH=src

# Function to display help
usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Available options:"
    echo "  --dataset <name>          Dataset name (default: $DATASET)"
    echo "  --dataset_path <path>     Dataset path (default: $DATASET_PATH)"
    echo "  --batch_size <num>        Batch size (default: $BATCH_SIZE)"
    echo "  --dropout_values <values> Dropout rates"
    echo "  --hidden_dims <values>    Hidden dimensions"
    echo "  --num_layers_values <values> Number of layers"
    echo "  --weight_decay_values <values> Weight decay"
    echo "  --non_lin <function>      Activation function (default: $NON_LIN)"
    echo "  --device <type>           Device (cuda/cpu) (default: $DEVICE)"
    echo "  --epochs <num>            Number of epochs (default: $EPOCHS)"
    echo "  --output_path <path>      Output path for models (default: $OUTPUT_PATH)"
    echo "  --method <method>         Training method (default: $METHOD)"
    echo "  --hpo <true/false>        Hyperparameter optimization (default: $HPO)"
    echo "  --help                    Display this message and exit"
    exit 0
}

# Processamento dos argumentos
while [ "$#" -gt 0 ]; do
    case $1 in
        --dataset) DATASET="$2"; shift ;;
        --dataset_path) DATASET_PATH="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        --lr_values) LR_VALUES=($2); shift ;;
        --dropout_values) DROPOUT_VALUES=($2); shift ;;
        --hidden_dims) HIDDEN_DIMS=($2); shift ;;
        --num_layers_values) NUM_LAYERS_VALUES=($2); shift ;;
        --weight_decay_values) WEIGHT_DECAY_VALUES=($2); shift ;;
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
            echo "Running with arguments: --dataset $DATASET --dataset_path $DATASET_PATH --batch_size $BATCH_SIZE --lr_values $LR_VALUES --dropout $DROPOUT_VALUES --hidden_dims $HIDDEN_DIMS --num_layers_values $NUM_LAYERS_VALUES --weight_decay_values $WEIGHT_DECAY_VALUES --non_lin $NON_LIN --device $DEVICE --epochs $EPOCHS --seed $SEED --output_path $OUTPUT_PATH --method $METHOD"
            # Controle de processos simultâneos
            PYTHONPATH=src poetry run python -m hmc.train.main \
                --datasets "$dataset" \
                --dataset_path "$DATASET_PATH" \
                --batch_size "$BATCH_SIZE" \
                --lr_values "${LR_VALUES[@]}" \
                --dropout_values "${DROPOUT_VALUES[@]}" \
                --hidden_dims "${HIDDEN_DIMS[@]}" \
                --num_layers_values "${NUM_LAYERS_VALUES[@]}" \
                --weight_decay_values "${WEIGHT_DECAY_VALUES[@]}" \
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
            echo "Running with arguments: --dataset $DATASET --dataset_path $DATASET_PATH --batch_size $BATCH_SIZE --lr $LR --dropout_values $DROPOUT_VALUES --hidden_dims $HIDDEN_DIMS --num_layers_values $NUM_LAYERS_VALUES --weight_decay_values $WEIGHT_DECAY_VALUES --non_lin $NON_LIN --device $DEVICE --epochs $EPOCHS --seed $SEED --output_path $OUTPUT_PATH --method $METHOD"
            # Controle de processos simultâneos
            cmd="poetry run python -m hmc.train.main \
                --datasets $DATASET \
                --dataset_path $DATASET_PATH \
                --batch_size $BATCH_SIZE \
                --non_lin $NON_LIN \
                --device $DEVICE \
                --epochs $EPOCHS \
                --seed $SEED \
                --output_path $OUTPUT_PATH \
                --method $METHOD \
                --hpo $HPO"
            if [ "$HPO" = "false" ]; then
                cmd+=" --lr_values ${LR_VALUES[@]} \
                        --dropout_values ${DROPOUT_VALUES[@]} \
                        --hidden_dims ${HIDDEN_DIMS[@]} \
                        --num_layers_values ${NUM_LAYERS_VALUES[@]} \
                        --weight_decay_values ${WEIGHT_DECAY_VALUES[@]}"
            fi

            $cmd &  # Executa em segundo plano
    done
fi
TRAIN_PID=$!
# Aguarda a execução de todos os processos restantes
trap "kill $TRAIN_PID" SIGINT SIGTERM
wait

echo "All experiments completed!"
