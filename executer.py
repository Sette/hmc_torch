# %%
import os
import pandas as pd
from datetime import datetime as dt
from hmc.utils.dir import create_dir, create_job_id
from hmc.model.train import run

# %%
base_path = "/mnt/disks/data/fma/trains"
id = "hierarchical_tworoots"


# %%
train_path = os.path.join(base_path,id)
torch_path = os.path.join(train_path,'torch')
metadata_path = os.path.join(train_path,"metadata.json")
labels_path = os.path.join(train_path,"labels.json")


models_path = os.path.join(train_path, "models")

hmc_path = os.path.join(train_path, 'hmc_torch_effnet')

job_id = create_job_id()
print(f"Job ID: {job_id}")

model_path = os.path.join(hmc_path, job_id)
create_dir(model_path)


args = pd.Series({
    "batch_size":64,
    "epochs":15,
    "dropout":0.1,
    'patience':5,
    'shuffle':True,
    'max_queue_size':64,
    "labels_path": labels_path,
    "metadata_path": metadata_path,
    "model_path": model_path,
    "train_path": os.path.join(torch_path,'train'),
    "test_path": os.path.join(torch_path,'test'),
    "val_path": os.path.join(torch_path,'val')
})


# %%

time_start = dt.utcnow()
print("[{}] Experiment started at {}".format(id, time_start.strftime("%H:%M:%S")))
print(".......................................")
print(args)
run(args)
time_end = dt.utcnow()
time_elapsed = time_end - time_start
print(".......................................")
print("[{}] Experiment finished at {} / elapsed time {}s".format(id, time_end.strftime("%H:%M:%S"), time_elapsed.total_seconds()))

