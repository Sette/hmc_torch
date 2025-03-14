#!/bin/bash

# Definições do servidor remoto
REMOTE_USER="bruno"
REMOTE_HOST="rock.dc.ufscar.br"
REMOTE_PORT="2002"  # Altere para a porta correta do seu servidor
REMOTE_PATH="/home/bruno/git/hmc_torch"
SCRIPT_TO_RUN="train.sh"  # Nome do script Python a ser executado remotamente

# Obtém a lista de arquivos modificados no Git
MODIFIED_FILES=$(git status --porcelain | awk '{print $2}')

# Verifica se há arquivos modificados
if [ -z "$MODIFIED_FILES" ]; then
    echo "Nenhum arquivo modificado para enviar."
    exit 0
fi

# Envia apenas os arquivos modificados para o servidor
echo "Enviando arquivos modificados..."
for FILE in $MODIFIED_FILES; do
    rsync -avz --progress -e "ssh -p $REMOTE_PORT" "$FILE" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/$FILE"
done

# Conecta-se ao servidor, inicia ou anexa a uma sessão tmux e executa o script Python
echo "Executando o script Python remotamente dentro do tmux..."
ssh -p "$REMOTE_PORT" "$REMOTE_USER@$REMOTE_HOST" "
    tmux has-session -t $TMUX_SESSION 2>/dev/null || tmux new-session -d -s $TMUX_SESSION
    tmux send-keys -t $TMUX_SESSION 'cd $REMOTE_PATH && python3 $SCRIPT_TO_RUN' C-m
"

echo "Deploy concluído! O script está rodando dentro do tmux na sessão '$TMUX_SESSION'."