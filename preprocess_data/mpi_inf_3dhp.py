import shutil
import numpy as np
import os

data = np.load("datasets/mpi_inf_3dhp/mpi_inf_3dhp_train.npz")

for imgname in data["imgname"]:
    dest_fpath = f"datasets/mpi_inf_3dhp/mpi_inf_3dhp_train/{imgname}"
    os.makedirs(os.path.dirname(dest_fpath), exist_ok=True)
    shutil.copyfile(
        f"/media/fast/LaCie/data/video_datasets/mpi_inf_3dhp/{imgname}",
        dest_fpath,
    )
