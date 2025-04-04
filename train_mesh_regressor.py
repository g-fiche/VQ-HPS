from vq_hps import (
    MeshRegressor,
    MeshRegressor_Train,
    MixedDataset,
    DatasetHMR,
    set_seed,
)
from mesh_vq_vae import FullyConvAE, MeshVQVAE
import hydra
from omegaconf import DictConfig
import os
import numpy as np
import torch

import argparse

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("-p", "--path", type=str, help="Path to data", default="datasets")

args = parser.parse_args()
path = args.path


@hydra.main(
    config_path="configs",
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())

    set_seed()

    ref_bm_path = "body_models/smplh/neutral/model.npz"
    ref_bm = np.load(ref_bm_path)

    """Data"""
    training_data = MixedDataset(
        f"configs/{cfg.training_data.file}",
        augment=True,
        flip=True,
        proportion=1.0,
    )
    validation_data = DatasetHMR(
        dataset_file=f"{path}/{cfg.validation_data.file}",
        augment=False,
        proportion=0.1,
    )

    """ ConvMesh VQVAE model """
    convmesh_model = FullyConvAE(cfg.modelconv, test_mode=True)
    mesh_vqvae = MeshVQVAE(convmesh_model, **cfg.vqvaemesh)
    mesh_vqvae.load(path_model="checkpoint/MESH_VQVAE/mesh_vqvae_54")
    convmesh_model.init_test_mode()
    pytorch_total_params = sum(
        p.numel() for p in mesh_vqvae.parameters() if p.requires_grad
    )
    print(f"VQVAE-Mesh: {pytorch_total_params}")

    """ MeshRegressor model """
    mesh_regressor = MeshRegressor(
        **cfg.regressor,
    )
    pytorch_total_params = sum(
        p.numel() for p in mesh_regressor.parameters() if p.requires_grad
    )
    print(f"Regressor: {pytorch_total_params}")

    """Joint regressor"""
    J_regressor = torch.from_numpy(np.load("body_models/J_regressor_h36m.npy")).float()

    J_regressor_24 = torch.from_numpy(np.load("body_models/J_regressor_24.npy")).float()

    """ Training """
    pretrain_mesh_regressor = MeshRegressor_Train(
        mesh_regressor,
        training_data,
        validation_data,
        cfg.train,
        faces=torch.from_numpy(ref_bm["f"].astype(np.int32)),
        vqvae_mesh=mesh_vqvae,
        joints_regressor=J_regressor,
        joints_regressor_smpl=J_regressor_24,
    )
    pretrain_mesh_regressor.fit()


if __name__ == "__main__":
    main()
