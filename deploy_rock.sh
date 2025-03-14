#!/bin/bash

# Definições do servidor remoto
REMOTE_USER="bruno"
REMOTE_HOST="rock.dc.ufscar.br"
REMOTE_PORT="2002"  # Altere para a porta correta do seu servidor
REMOTE_PATH="/home/bruno/git/hmc_torch"
SCRIPT_TO_RUN="train.sh"  # Nome do script Python a ser executado remotamente
TMUX_SESSION="train_session"  # Nome da sessão tmux

# Pergunta ao usuário como deseja atualizar o código
echo "Escolha o modo de atualização:"
echo "1 - Sincronizar apenas os arquivos modificados"
echo "2 - Executar git pull no servidor"
read -rp "Digite o número da opção desejada (1 ou 2): " UPDATE_MODE

if [[ "$UPDATE_MODE" == "1" ]]; then
    # Obtém a lista de arquivos modificados no Git
    MODIFIED_FILES=$(git status --porcelain | awk '{print $2}')

    if [ -z "$MODIFIED_FILES" ]; then
        echo "Nenhum arquivo modificado para enviar."
    else
        echo "Arquivos modificados:"
        echo "$MODIFIED_FILES"

        # Pergunta ao usuário se deseja sincronizar os arquivos modificados
        read -rp "Deseja sincronizar os arquivos modificados? (s/n): " SYNC_CHOICE
        SYNC_CHOICE=${SYNC_CHOICE,,}  # Converte para minúsculas

        if [[ "$SYNC_CHOICE" == "s" ]]; then
            echo "Enviando arquivos modificados..."
            for FILE in $MODIFIED_FILES; do
                rsync -avz --progress -e "ssh -p $REMOTE_PORT" "$FILE" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/$FILE"
            done
        else
            echo "Pulando sincronização de arquivos."
        fi
    fi

    # Garante que train.sh seja enviado
    echo "Enviando train.sh para o servidor..."
    rsync -avz --progress -e "ssh -p $REMOTE_PORT" "train.sh" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/train.sh"

elif [[ "$UPDATE_MODE" == "2" ]]; then
    echo "Executando git pull no servidor..."
    ssh -p "$REMOTE_PORT" "$REMOTE_USER@$REMOTE_HOST" "cd $REMOTE_PATH && git pull"
else
    echo "Opção inválida! Saindo..."
    exit 1
fi

# Verifica se train.sh existe no servidor antes de executar
ssh -p "$REMOTE_PORT" "$REMOTE_USER@$REMOTE_HOST" "test -f $REMOTE_PATH/$SCRIPT_TO_RUN"
if [ $? -ne 0 ]; then
    echo "Erro: O arquivo $SCRIPT_TO_RUN não foi encontrado no servidor!"
    exit 1
fi

# Pergunta se deseja rodar com tmux ou não
read -rp "Deseja executar com tmux? (s/n): " TMUX_CHOICE
TMUX_CHOICE=${TMUX_CHOICE,,}

if [[ "$TMUX_CHOICE" == "s" ]]; then
    echo "Executando o script de treinamento dentro do tmux..."
    ssh -p "$REMOTE_PORT" "$REMOTE_USER@$REMOTE_HOST" "
        tmux has-session -t $TMUX_SESSION 2>/dev/null || tmux new-session -d -s $TMUX_SESSION
        tmux send-keys -t $TMUX_SESSION 'cd $REMOTE_PATH && chmod +x $SCRIPT_TO_RUN && ./train.sh' C-m
    "
    echo "Treinamento iniciado dentro do tmux! Sessão: '$TMUX_SESSION'."
else
    echo "Executando o script de treinamento diretamente..."
    ssh -p "$REMOTE_PORT" "$REMOTE_USER@$REMOTE_HOST" "
        cd $REMOTE_PATH && chmod +x $SCRIPT_TO_RUN && ./train.sh
    "
    echo "Treinamento iniciado sem tmux."
fi
