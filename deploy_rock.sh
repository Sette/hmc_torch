#!/bin/bash

# Definições do servidor remoto
REMOTE_USER="bruno"
REMOTE_HOST="rock"
REMOTE_PORT="2002"
REMOTE_PATH="/home/bruno/git/hmc_torch"
SCRIPT_TO_RUN="train.sh"  # Nome do script a ser executado remotamente
TMUX_SESSION="train_session"  # Nome da sessão tmux

# Pergunta ao usuário se deseja executar git pull antes de rodar o treinamento
echo -n "Deseja executar git pull antes de iniciar o treinamento? (s/n): "
read GIT_PULL_CHOICE
GIT_PULL_CHOICE=$(echo "$GIT_PULL_CHOICE" | tr '[:upper:]' '[:lower:]')  # Converte para minúsculas

if [ "$GIT_PULL_CHOICE" = "s" ]; then
    echo "Executando git pull no servidor..."
    ssh -t -p "$REMOTE_PORT" "$REMOTE_USER@$REMOTE_HOST" "
        export GIT_SSH_COMMAND='ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null'
        cd $REMOTE_PATH &&
        git pull
    "
else
    echo "Pulando git pull."
fi

# Pergunta ao usuário se deseja executar com tmux ou sem tmux
echo -n "Deseja executar com tmux? (s/n): "
read TMUX_CHOICE
TMUX_CHOICE=$(echo "$TMUX_CHOICE" | tr '[:upper:]' '[:lower:]')  # Converte para minúsculas

if [ "$TMUX_CHOICE" = "s" ]; then
    echo "Executando o script de treinamento dentro do tmux..."
    ssh -p "$REMOTE_PORT" "$REMOTE_USER@$REMOTE_HOST" "
        tmux has-session -t $TMUX_SESSION 2>/dev/null || tmux new-session -d -s $TMUX_SESSION
        tmux send-keys -t $TMUX_SESSION 'cd $REMOTE_PATH && chmod +x $SCRIPT_TO_RUN && ./train.sh' C-m
    "
    echo "Treinamento iniciado dentro do tmux! Sessão: '$TMUX_SESSION'."
else
    echo "Executando o script de treinamento diretamente..."
    ssh -p "$REMOTE_PORT" "$REMOTE_USER@$REMOTE_HOST" "
        source ~/.zshrc &&
        cd $REMOTE_PATH &&
        python3 -m venv .venv &&
        source .venv/bin/activate &&
        poetry install --with torch-cuda --no-root &&
        chmod +x $SCRIPT_TO_RUN && ./train.sh --device cuda
    "
    echo "Treinamento iniciado sem tmux."
fi
