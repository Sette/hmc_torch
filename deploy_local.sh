#!/bin/zsh

# Verifica se foi passado um argumento (cuda ou cpu)
if [ "$#" -ne 1 ]; then
    echo "Uso: $0 [cuda | cpu]"
    exit 1
fi

PYTORCH_MODE=$1

if [ "$PYTORCH_MODE" != "cuda" ] && [ "$PYTORCH_MODE" != "cpu" ]; then
    echo "Erro: O argumento deve ser 'cuda' ou 'cpu'."
    exit 1
fi

# Pergunta ao usuário se deseja instalar os pacotes com poetry
echo -n "Do you want to install dependencies with Poetry? (y/n): "
read INSTALL_CHOICE
INSTALL_CHOICE=$(echo "$INSTALL_CHOICE" | tr '[:upper:]' '[:lower:]')  # Convert to lowercase


# Pergunta ao usuário se deseja executar com tmux ou sem tmux
echo -n "Deseja executar com tmux? (s/n): "
read TMUX_CHOICE
TMUX_CHOICE=$(echo "$TMUX_CHOICE" | tr '[:upper:]' '[:lower:]')  # Converte para minúsculas


if [ "$TMUX_CHOICE" = "s" ]; then
    echo "Executando o script de treinamento dentro do tmux..."
    source ~/.zshrc &&
    tmux has-session -t $TMUX_SESSION 2>/dev/null || tmux new-session -d -s $TMUX_SESSION

    if [ "$PYTORCH_MODE" = "cuda" ]; then
        tmux send-keys -t $TMUX_SESSION 'source ~/.zshrc &&
        python3 -m venv .venv &&
        source .venv/bin/activate &&
        poetry source add pytorch-gpu https://download.pytorch.org/whl/cu118 --priority=explicit &&
        poetry source remove pytorch-cpu || true &&
        poetry install --no-root &&
        chmod +x train.sh && ./train.sh --device cuda'
    else
        tmux send-keys -t $TMUX_SESSION 'source ~/.zshrc &&
        python3 -m venv .venv &&
        source .venv/bin/activate &&
        poetry source add pytorch-cpu https://download.pytorch.org/whl/cpu --priority=explicit &&
        poetry source remove pytorch-gpu || true &&
        poetry install --no-root &&
        chmod +x train.sh && ./train.sh --device cpu'
    fi
    echo "Treinamento iniciado dentro do tmux! Sessão: '$TMUX_SESSION'."
else
    echo "Executando o script de treinamento diretamente..."
    source ~/.zshrc
    if [ "$INSTALL_CHOICE" = "y" ] && [ "$PYTORCH_MODE" = "cuda" ]; then
        python3 -m venv .venv &&
        source .venv/bin/activate &&
        poetry source add pytorch-gpu https://download.pytorch.org/whl/cu118 --priority=explicit &&
        poetry source remove pytorch-cpu || true &&
        poetry install --no-root &&
        chmod +x train.sh
    fi

    if [ "$INSTALL_CHOICE" = "y" ] && [ "$PYTORCH_MODE" = "cuda" ]; then
        python3 -m venv .venv &&
        source .venv/bin/activate &&
        poetry source add pytorch-cpu https://download.pytorch.org/whl/cpu --priority=explicit &&
        poetry source remove pytorch-gpu || true &&
        poetry install --no-root &&
        chmod +x train.sh
    fi

    if [ "$PYTORCH_MODE" = "cuda" ]; then
        echo "Usando PyTorch com CUDA..."
        ./train.sh --device cuda
    else
        echo "Usando PyTorch com CPU..."
        ./train.sh --device cpu
    fi

    echo "Treinamento terminado sem tmux."
fi
