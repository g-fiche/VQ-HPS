from torch.utils.data import DataLoader, Dataset
import torch
import torch.optim as optim
from tqdm import tqdm
from ...base import Train
import matplotlib.pyplot as plt

plt.switch_backend("agg")
from matplotlib.gridspec import GridSpec
from .follow_up_regressor import Follow
from mesh_vq_vae import MeshVQVAE
from ...model import MeshRegressor
from ...utils.mesh_render import renderer
from ...utils.img_renderer import visualize_reconstruction_pyrender, PyRender_Renderer
from ...utils.eval import *
from statistics import mean
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    rotation_6d_to_matrix,
)
import numpy as np

from .idr_torch import IDR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from loguru import logger
import os
import math

import random


class MeshRegressor_Train(Train):
    def __init__(
        self,
        model: MeshRegressor,
        training_data: Dataset,
        validation_data: Dataset,
        config_training: dict = None,
        vqvae_mesh: MeshVQVAE = None,
        faces=None,
        joints_regressor=None,
        joints_regressor_smpl=None,
        multigpu_bool=False,
    ):
        if multigpu_bool:
            self.idr = IDR()
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                world_size=self.idr.size,
                rank=self.idr.rank,
            )
            torch.cuda.set_device(self.idr.local_rank)

        self.device = torch.device("cuda")

        """Model"""
        self.model = model
        self.model.to(self.device)
        self.vqvae_mesh = vqvae_mesh
        self.vqvae_mesh.to(self.device)

        tpose = np.load("body_models/tpose.npy")
        self.tpose = torch.from_numpy(tpose).float()

        if multigpu_bool:
            self.model = DDP(
                self.model,
                device_ids=[self.idr.local_rank],
                find_unused_parameters=True,
            )
            self.vqvae_mesh = DDP(
                self.vqvae_mesh,
                device_ids=[self.idr.local_rank],
                find_unused_parameters=True,
            )

        """ Dataloader """
        if multigpu_bool:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                training_data,
                num_replicas=self.idr.size,
                rank=self.idr.rank,
                shuffle=True,
            )
            self.training_loader = torch.utils.data.DataLoader(
                dataset=training_data,
                batch_size=config_training["batch"] // self.idr.size,
                shuffle=False,
                num_workers=config_training["workers"],
                pin_memory=True,
                drop_last=True,
                sampler=train_sampler,
            )
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                validation_data,
                num_replicas=self.idr.size,
                rank=self.idr.rank,
                shuffle=True,
            )
            self.validation_loader = torch.utils.data.DataLoader(
                dataset=validation_data,
                batch_size=config_training["batch"] // self.idr.size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                sampler=val_sampler,
                drop_last=True,
                prefetch_factor=2,
            )
        else:
            self.training_loader = DataLoader(
                training_data,
                batch_size=config_training["batch"],
                shuffle=True,
                num_workers=config_training["workers"],
                drop_last=True,
            )
            self.validation_loader = DataLoader(
                validation_data,
                batch_size=config_training["batch"],
                shuffle=True,
                num_workers=config_training["workers"],
                pin_memory=True,
                drop_last=True,
            )

        self.lr = config_training["lr"]

        """ Optimizer """

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=config_training["weight_decay"],
        )
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=100
        )

        """ Loss """
        self.loss = torch.nn.CrossEntropyLoss(reduction="mean")
        self.mse = torch.nn.MSELoss(reduction="mean")

        """ Config """
        self.config_training = config_training
        self.train_res_recon_error = []
        self.train_v2v_error = []
        self.train_pampjpe_error = []
        self.train_rot_error = []
        self.train_acc = []
        self.train_2d = []
        self.val_res_recon_error = []
        self.val_rot_err = []
        self.val_v2v_error = []
        self.val_pampjpe_error = []
        self.val_acc = []
        self.val_2d = []
        self.load_epoch = 0
        self.parameters = dict()

        self.multigpu_bool = multigpu_bool

        self.epochs = config_training["total_epoch"]

        self.f = faces

        self.joints_reg = joints_regressor
        self.joints_reg_smpl = joints_regressor_smpl

        """Follow"""
        self.follow = Follow(
            "mesh_regressor",
            dir_save="checkpoint",
            multigpu_bool=multigpu_bool,
        )

    def one_epoch(self):
        pass

    def fit(self):
        logger.add(
            os.path.join(self.follow.path, "train.log"),
            level="INFO",
            colorize=False,
        )

        self.model.train()
        t = self.load_epoch
        for i in range(t, self.epochs):
            if self.multigpu_bool:
                self.training_loader.sampler.set_epoch(i)
                self.validation_loader.sampler.set_epoch(i)
            logger.info(f"Train {i}")
            for data in tqdm(self.training_loader):
                self.optimizer.zero_grad()
                with torch.no_grad():
                    data["mesh_idx"] = self.vqvae_mesh.get_codebook_indices(
                        data["local_mesh"].to(self.device),
                    ).cpu()

                pred_v = self.tpose.repeat(data["img"].shape[0], 1, 1)
                J_regressor_batch = (
                    self.joints_reg[None, :].expand(pred_v.shape[0], -1, -1).to(pred_v)
                )
                pred_j = torch.matmul(J_regressor_batch, pred_v).reshape(-1, 51)

                out = self.model(
                    data["img"].to(self.device),
                    pose_init=pred_j.to(self.device),
                    mesh_decoder=self.vqvae_mesh,
                    joints_reg=self.joints_reg,
                )

                loss = 0
                loss = self.loss(
                    torch.flatten(out["prob_indices"], 0, 1),
                    torch.flatten(data["mesh_idx"].type(torch.LongTensor)),
                )
                rot_loss = self.mse(
                    rotation_6d_to_matrix(out["pred_rot"]),
                    axis_angle_to_matrix(data["rotation"]),
                ).mean()
                self.train_rot_error.append(rot_loss.item())
                loss += rot_loss

                is_3dpw = data["is_3dpw"] == True
                not_3dpw = data["is_3dpw"] == False
                reproj_loss = 0
                if is_3dpw.any():
                    reproj_loss += reprojection_loss(
                        data["j2d"][is_3dpw][:, :, :2],
                        out["pred_mesh"][is_3dpw],
                        out["pred_cam"][is_3dpw],
                        self.joints_reg_smpl,
                    )
                if not_3dpw.any():
                    reproj_loss += reprojection_loss_conf(
                        data["j2d"][not_3dpw],
                        out["pred_mesh"][not_3dpw],
                        out["pred_cam"][not_3dpw],
                        self.joints_reg,
                    )
                loss += reproj_loss
                self.train_2d.append(reproj_loss.item())

                loss.backward(retain_graph=True)
                self.train_res_recon_error.append(loss.item())
                self.optimizer.step()

                accuracy = torch.sum(
                    data["mesh_idx"] == out["pred_indices"]
                ) / torch.numel(data["mesh_idx"])
                self.train_acc.append(100 * accuracy.item())

                recon_error = pa_mpjpe(data["mesh"], out["pred_mesh"], self.joints_reg)

                glob_recon_error = v2v(
                    data["mesh"],
                    out["pred_mesh"],
                )

                self.train_pampjpe_error.append(1000 * recon_error.item())
                self.train_v2v_error.append(1000 * glob_recon_error.item())

            logger.info(
                f"Training loss: {mean(self.train_res_recon_error[-len(self.training_loader):])} || Accuracy: {mean(self.train_acc[-len(self.training_loader):])} || PA-MPJPE: {mean(self.train_pampjpe_error[-len(self.training_loader):])} || V2V: {mean(self.train_v2v_error[-len(self.training_loader):])}"
            )

            self.plot_meshes_(
                out["pred_mesh"][:4],
                self.f,
                show=False,
                save=f"{self.follow.path_samples_train}/{i}-reconstruction.png",
            )
            self.plot_meshes_(
                data["mesh"].to(self.device)[:4],
                self.f,
                show=False,
                save=f"{self.follow.path_samples_train}/{i}-real.png",
            )

            pred_v = out["pred_mesh"].detach()
            cam = out["pred_cam"]
            raw_img = data["raw_img"].cpu().numpy().transpose(0, 2, 3, 1)
            self.plot_reproj_(
                raw_img[:4],
                pred_v[:4],
                cam[:4],
                show=False,
                save=f"{self.follow.path_samples_train}/{i}_reprojection.png",
            )

            with torch.no_grad():
                for data in tqdm(self.validation_loader):
                    data["mesh_idx"] = self.vqvae_mesh.get_codebook_indices(
                        data["local_mesh"].to(self.device),
                    ).cpu()
                    pred_v = self.tpose.repeat(data["img"].shape[0], 1, 1)
                    J_regressor_batch = (
                        self.joints_reg[None, :]
                        .expand(pred_v.shape[0], -1, -1)
                        .to(pred_v)
                    )
                    pred_j = torch.matmul(J_regressor_batch, pred_v).reshape(-1, 51)

                    out = self.model(
                        data["img"].to(self.device),
                        pose_init=pred_j.to(self.device),
                        mesh_decoder=self.vqvae_mesh,
                        joints_reg=self.joints_reg,
                    )

                    loss = 0

                    loss = self.loss(
                        torch.flatten(out["prob_indices"], 0, 1),
                        torch.flatten(data["mesh_idx"].type(torch.LongTensor)),
                    )

                    rotation = self.mse(
                        rotation_6d_to_matrix(out["pred_rot"]),
                        axis_angle_to_matrix(data["rotation"]),
                    ).mean()
                    loss += rotation
                    self.val_rot_err.append(rotation.item())

                    is_3dpw = data["is_3dpw"] == True
                    not_3dpw = data["is_3dpw"] == False
                    reproj_loss = 0
                    if is_3dpw.any():
                        reproj_loss += reprojection_loss(
                            data["j2d"][is_3dpw][:, :, :2],
                            out["pred_mesh"][is_3dpw],
                            out["pred_cam"][is_3dpw],
                            self.joints_reg_smpl,
                        )
                    if not_3dpw.any():
                        reproj_loss += reprojection_loss_conf(
                            data["j2d"][not_3dpw],
                            out["pred_mesh"][not_3dpw],
                            out["pred_cam"][not_3dpw],
                            self.joints_reg,
                        )
                    loss += reproj_loss
                    self.val_2d.append(reproj_loss.item())

                    self.val_res_recon_error.append(loss.item())
                    recon_error = pa_mpjpe(
                        data["mesh"], out["pred_mesh"], self.joints_reg
                    )
                    self.val_pampjpe_error.append(1000 * recon_error.item())

                    global_error = v2v(data["mesh"], out["pred_mesh"])
                    self.val_v2v_error.append(1000 * global_error.item())
                    accuracy = torch.sum(
                        data["mesh_idx"] == out["pred_indices"]
                    ) / torch.numel(data["mesh_idx"])
                    self.val_acc.append(100 * accuracy.item())

                logger.info(
                    f"Validation V2V: Accuracy: {mean(self.val_acc[-len(self.validation_loader):])} || PA-MPJPE: {mean(self.val_pampjpe_error[-len(self.validation_loader):])} || V2V: {mean(self.val_v2v_error[-len(self.validation_loader):])}"
                )

                self.plot_meshes_(
                    out["pred_mesh"][:4].to(self.device),
                    self.f,
                    show=False,
                    save=f"{self.follow.path_samples}/{i}-reconstruction.png",
                )
                self.plot_meshes_(
                    data["mesh"].to(self.device)[:4],
                    self.f,
                    show=False,
                    save=f"{self.follow.path_samples}/{i}-real.png",
                )

                pred_v = out["pred_mesh"]
                cam = out["pred_cam"]
                raw_img = data["raw_img"].cpu().numpy().transpose(0, 2, 3, 1)
                self.plot_reproj_(
                    raw_img[:4],
                    pred_v[:4],
                    cam[:4],
                    show=False,
                    save=f"{self.follow.path_samples}/{i}_reprojection.png",
                )

            self.parameters = dict(
                model=self.model.state_dict(),
                optimizer=self.optimizer.state_dict(),
                epoch=i,
                loss=mean(self.train_res_recon_error[-len(self.training_loader) :]),
                pampjpe=mean(self.val_pampjpe_error[-len(self.validation_loader) :]),
                v2v=mean(self.val_v2v_error[-len(self.validation_loader) :]),
            )
            self.follow(
                epoch=i,
                loss_train=mean(
                    self.train_res_recon_error[-len(self.training_loader) :]
                ),
                loss_validation=mean(
                    self.val_res_recon_error[-len(self.validation_loader) :]
                ),
                loss_rot_train=mean(self.train_rot_error[-len(self.training_loader) :]),
                loss_rot_validation=mean(
                    self.val_rot_err[-len(self.validation_loader) :]
                ),
                loss_2d_train=mean(self.train_2d[-len(self.training_loader) :]),
                loss_2d_validation=mean(self.val_2d[-len(self.validation_loader) :]),
                v2v_train=mean(self.train_v2v_error[-len(self.training_loader) :]),
                v2v_validation=mean(self.val_v2v_error[-len(self.validation_loader) :]),
                pampjpe_train=mean(
                    self.train_pampjpe_error[-len(self.training_loader) :]
                ),
                pampjpe_validation=mean(
                    self.val_pampjpe_error[-len(self.validation_loader) :]
                ),
                accuracy_train=mean(self.train_acc[-len(self.training_loader) :]),
                accuracy_validation=mean(self.val_acc[-len(self.validation_loader) :]),
                parameters=self.parameters,
            )
            self.lr_scheduler.step()

    def plot_meshes_(self, meshes, faces, show: bool = True, save: str = None):
        images = renderer(meshes, faces, self.device)
        fig = plt.figure(figsize=(10, 10))
        if len(meshes) == 16:
            nrows = 4
            ncols = 4
        elif len(meshes) == 4:
            nrows = 2
            ncols = 2
        else:
            ncols = len(meshes)
            nrows = 1

        gs = GridSpec(ncols=ncols, nrows=nrows)
        i = 0
        for line in range(nrows):
            for col in range(ncols):
                ax = fig.add_subplot(gs[line, col])
                if images[i].shape[0] == 1:
                    ax.imshow(images[i][0, :, :].cpu().detach().numpy())
                else:
                    ax.imshow(images[i].cpu().detach().numpy())
                plt.axis("off")
                i = i + 1
        if show:
            plt.show()
        if save is not None:
            plt.savefig(save)
            plt.close()

    def plot_reproj_(
        self, images, meshes, cameras, show: bool = True, save: str = None
    ):
        rendered_img = []
        render_reproj = PyRender_Renderer(faces=self.f)
        for img, vertices, camera in zip(images, meshes, cameras):
            rendered_img.append(
                visualize_reconstruction_pyrender(img, vertices, camera, render_reproj)
            )
        fig = plt.figure(figsize=(10, 10))
        if len(meshes) == 16:
            nrows = 4
            ncols = 4
        elif len(meshes) == 4:
            nrows = 2
            ncols = 2
        else:
            ncols = len(meshes)
            nrows = 1

        gs = GridSpec(ncols=ncols, nrows=nrows)
        i = 0
        for line in range(nrows):
            for col in range(ncols):
                ax = fig.add_subplot(gs[line, col])
                if rendered_img[i].shape[0] == 1:
                    ax.imshow(rendered_img[i][0, :, :])
                else:
                    ax.imshow(rendered_img[i])
                plt.axis("off")
                i = i + 1
        if show:
            plt.show()
        if save is not None:
            plt.savefig(save)
            plt.close()

    def plot_img_(self, images, show: bool = True, save: str = None):
        rendered_img = []
        for img in images:
            img = (img * 255).astype(np.uint8)
            rendered_img.append(img)
        fig = plt.figure(figsize=(10, 10))
        if len(images) == 16:
            nrows = 4
            ncols = 4
        elif len(images) == 4:
            nrows = 2
            ncols = 2
        else:
            ncols = len(images)
            nrows = 1

        gs = GridSpec(ncols=ncols, nrows=nrows)
        i = 0
        for line in range(nrows):
            for col in range(ncols):
                ax = fig.add_subplot(gs[line, col])
                if rendered_img[i].shape[0] == 1:
                    ax.imshow(rendered_img[i][0, :, :])
                else:
                    ax.imshow(rendered_img[i])
                plt.axis("off")
                i = i + 1
        if show:
            plt.show()
        if save is not None:
            plt.savefig(save)
            plt.close()

    def plot_error_(self, v2v_list, mpjpe_list, pampjpe_list):
        plt.figure(figsize=(10, 10))
        plt.boxplot(v2v_list, notch=True)
        plt.savefig(f"{self.follow.path}/v2v.png")
        plt.legend()
        plt.close()

        plt.figure(figsize=(10, 10))
        plt.boxplot(mpjpe_list, notch=True)
        plt.savefig(f"{self.follow.path}/mpjpe.png")
        plt.legend()
        plt.close()

        plt.figure(figsize=(10, 10))
        plt.boxplot(pampjpe_list, notch=True)
        plt.savefig(f"{self.follow.path}/pampjpe.png")
        plt.legend()
        plt.close()

    def eval(self):
        pass

    def test(self, visualize=False, detailed=False, recover_smpl=False):
        """
        If visualize, batch size must be 4, 2 or 1.
        If detailed, batch size must be 1.
        """
        logger.add(
            os.path.join(self.follow.path, "train.log"),
            level="INFO",
            colorize=False,
        )
        reduction = not detailed
        self.model.eval()
        pampjpe_list = []
        pav2v_list = []
        mpjpe_list = []
        v2v_list = []

        pampjpe_agg = []
        mpjpe_agg = []
        v2v_agg = []
        imgname_agg = []
        imgname_viz = []
        with torch.no_grad():
            for data in tqdm(self.validation_loader):

                pred_v = self.tpose.repeat(data["img"].shape[0], 1, 1)
                J_regressor_batch = (
                    self.joints_reg[None, :].expand(pred_v.shape[0], -1, -1).to(pred_v)
                )
                pred_j = torch.matmul(J_regressor_batch, pred_v).reshape(-1, 51)
                out = self.model(
                    data["img"].to(self.device),
                    pose_init=pred_j.to(self.device),
                    mesh_decoder=self.vqvae_mesh,
                    joints_reg=self.joints_reg,
                )

                imgname_agg.extend(data["imgname"])

                pampjpe_err = pa_mpjpe(
                    data["mesh"], out["pred_mesh"], self.joints_reg, reduction=reduction
                )
                pampjpe_list.append(1000 * pampjpe_err.mean().item())
                pampjpe_agg.extend((1000 * pampjpe_err).tolist())

                pav2v_err = pa_v2v(data["mesh"], out["pred_mesh"], reduction=reduction)
                pav2v_list.append(1000 * pav2v_err.mean().item())

                mpjpe_err = mpjpe(
                    data["mesh"], out["pred_mesh"], self.joints_reg, reduction=reduction
                )
                mpjpe_list.append(1000 * mpjpe_err.mean().item())
                mpjpe_agg.extend((1000 * mpjpe_err).tolist())

                v2v_err = v2v(data["mesh"], out["pred_mesh"], reduction=reduction)
                v2v_list.append(1000 * v2v_err.mean().item())
                v2v_agg.extend((1000 * v2v_err).tolist())

                if visualize and random.random() < 1:
                    imgname_list = data["imgname"][0].split("/")
                    imgname = "+".join(imgname_list)
                    imgname_viz.append(imgname)
                    self.plot_meshes_(
                        out["pred_mesh"],
                        self.f,
                        show=False,
                        save=f"{self.follow.path_samples}/{imgname}_reconstruction.png",
                    )
                    self.plot_meshes_(
                        data["mesh"].to(self.device),
                        self.f,
                        show=False,
                        save=f"{self.follow.path_samples}/{imgname}_real.png",
                    )
                    self.plot_img_(
                        data["raw_img"].cpu().numpy().transpose(0, 2, 3, 1),
                        show=False,
                        save=f"{self.follow.path_samples}/{imgname}_img.png",
                    )
                    cam = out["pred_cam"]
                    raw_img = data["raw_img"].cpu().numpy().transpose(0, 2, 3, 1)
                    self.plot_reproj_(
                        raw_img,
                        out["pred_mesh"],
                        cam,
                        show=False,
                        save=f"{self.follow.path_samples}/{imgname}_reprojection.png",
                    )

            logger.info(
                f"PA-MPJPE: {mean(pampjpe_list)} || PA-V2V: {mean(pav2v_list)} || MPJPE: {mean(mpjpe_list)} || V2V: {mean(v2v_list)}"
            )

            if detailed:
                import pandas as pd

                dict_results = {
                    "imgname": imgname_agg,
                    "v2v": v2v_agg,
                    "mpjpe": mpjpe_agg,
                    "pa_mpjpe": pampjpe_agg,
                }
                df = pd.DataFrame(dict_results)
                df.to_csv(f"{self.follow.path}/results.csv", index=False)

            self.plot_meshes_(
                out["pred_mesh"][:4],
                self.f,
                show=False,
                save=f"{self.follow.path_samples}/reconstruction.png",
            )
            self.plot_meshes_(
                data["mesh"].to(self.device)[:4],
                self.f,
                show=False,
                save=f"{self.follow.path_samples}/real.png",
            )
            cam = out["pred_cam"]
            raw_img = data["raw_img"].cpu().numpy().transpose(0, 2, 3, 1)
            self.plot_reproj_(
                raw_img[:4],
                out["pred_mesh"][:4],
                cam[:4],
                show=False,
                save=f"{self.follow.path_samples}/reprojection.png",
            )
            self.plot_img_(
                data["raw_img"][:4].cpu().numpy().transpose(0, 2, 3, 1),
                show=False,
                save=f"{self.follow.path_samples}/img.png",
            )

            if detailed:
                self.plot_error_(v2v_agg, mpjpe_agg, pampjpe_agg)

    def load(self, path: str = "", optimizer: bool = True):
        print("LOAD [", end="")
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model"])
        if optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.load_epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]
        print(
            f"model: ok  | optimizer:{optimizer}  |  loss: {loss}  |  epoch: {self.load_epoch}]"
        )
