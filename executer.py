# %%
import os
import pandas as pd
import sys
from datetime import UTC
from datetime import datetime as dt
from hmc.utils.dir import create_dir, create_job_id
from hmc.model.train import train
from hmc.model import train_constraint

# %%
base_path = "/home/bruno/storage/data/fma/trains"
id = "hierarchical_tworoots"


# %%
train_path = os.path.join(base_path, id)
torch_path = os.path.join(train_path,'torch')
metadata_path = os.path.join(train_path,"metadata.json")
labels_path = os.path.join(train_path,"labels.json")


models_path = os.path.join(train_path, "models")

hmc_path = os.path.join(train_path, 'hmc_torch_effnet')

job_id = create_job_id()
print(f"Job ID: {job_id}")

model_path = os.path.join(hmc_path, job_id)
create_dir(model_path)

'''
sys.argv = [
'script.py',
'--input_path', '/home/bruno/storage/data/fma/trains/rock_electronic',
'--output_path', '/home/bruno/storage/data/fma/trains/rock_electronic',
'--batch_size', '64',
'--epochs', '100',
'--thresholds', '0.5', '0.5', '0.5', '0.5',
'--dropouts', '0.3', '0.3', '0.3', '0.3',
'--patience', '2'
]
'''

sys.argv = [
'script.py',
'--dataset', 'seq_GO', 'derisi_GO', 'gasch1_GO', 'cellcycle_FUN',
'--dataset_path', '/home/bruno/storage/data/go_fun',
'--batch_size', '32',
'--lr', '1e-5',
'--dropout', '0.7',
'--hidden_dim', '50',
'--num_layers', '2',
'--weight_decay', '1e-5',
'--non_lin', 'relu',
'--device', '0',
'--num_epochs', '2000',
'--seed', '0',
'--output_path', '/home/bruno/storage/models/gofun'
]

# %%
# Salvar os argumentos atuais
time_start = dt.now(UTC)
print("Experiment started at {}".format(time_start.strftime("%H:%M:%S")))
print(".......................................")
df_predict = train_constraint()
time_end = dt.now(UTC)
time_elapsed = time_end - time_start
print(".......................................")
print("Experiment finished at {} / elapsed time {}s".format(time_end.strftime("%H:%M:%S"), time_elapsed.total_seconds()))

