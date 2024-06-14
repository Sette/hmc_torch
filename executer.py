# %%
import os
import pandas as pd
from datetime import datetime as dt
from hmc.utils.dir import create_dir
from hmc.model.train import run

# %%
base_path = "/mnt/disks/data/fma/trains"
id = "hierarchical_tworoots_dev"


# %%
train_path = os.path.join(base_path,id)
torch_path =os.path.join(train_path,'torch')
metadata_path = os.path.join(train_path,"metadata.json")
labels_path = os.path.join(train_path,"labels.json")


args = pd.Series({
    "batch_size":64,
    "epochs":10,
    "dropout":0.5,
    'patience':1,
    'shuffle':True,
    'max_queue_size':64,
    "labels_path": labels_path,
    "metadata_path": metadata_path,
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

