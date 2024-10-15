# %%
import os
import pandas as pd
import sys
from datetime import UTC
from datetime import datetime as dt
from hmc.utils.dir import create_dir, create_job_id
from hmc.model.train import run

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

sys.argv = [
'script.py',
'--input_path', '/home/bruno/storage/data/fma/trains/rock_electronic',
'--output_path', '/home/bruno/storage/data/fma/trains/rock_electronic',
'--batch_size', '64',
'--epochs', '10',
'--thresholds', '0.5', '0.5', '0.5', '0.5',
'--lrs', '0.001', '0.001', '0.001', '0.001',
'--dropouts', '0.3', '0.3', '0.3', '0.1',
'--patience', '2'
]




# %%
# Salvar os argumentos atuais
time_start = dt.now(UTC)
print("[{}] Experiment started at {}".format(id, time_start.strftime("%H:%M:%S")))
print(".......................................")
df_predict = run()
time_end = dt.now(UTC)
time_elapsed = time_end - time_start
print(".......................................")
print("[{}] Experiment finished at {} / elapsed time {}s".format(id, time_end.strftime("%H:%M:%S"), time_elapsed.total_seconds()))

