"""Implements VQ-HPS"""

from __future__ import absolute_import, division, print_function
import torch
from torch import nn
from torch import nn
import torchvision.models as models
from .transformer import build_transformer
from .positional_encoding import build_position_encoding
import json
import math
from ..backbone.hrnet import hrnet_w48
from pytorch3d.transforms import rotation_6d_to_matrix
from collections import OrderedDict


class MeshRegressor(nn.Module):
    """FastMETRO for 3D human pose and mesh reconstruction from a single RGB image"""

    def __init__(
        self,
        cfg,
        backbone="vqvae",
        num_vertex=54,
        seq_len=49,
        autoreg=True,
    ):
        """
        Parameters:
            - args: Arguments
            - backbone: CNN Backbone used to extract image features from the given image
            - mesh_sampler: Mesh Sampler used in the coarse-to-fine mesh upsampling
            - num_joints: The number of joint tokens used in the transformer decoder
            - num_vertices: The number of vertex tokens used in the transformer decoder
        """
        super().__init__()

        self.autoreg = autoreg

        # configurations for the transformer
        with open(cfg) as f:
            data = f.read()
        self.transformer_config = json.loads(data)

        # build transformer
        self.transformer = build_transformer(
            self.transformer_config, autoreg=self.autoreg
        )
        self.model_dim = self.transformer_config["model_dim"]
        self.seq_len = seq_len

        # token embeddings
        self.mesh_token_embed = nn.Embedding(num_vertex, self.model_dim)  # verts

        # backbone
        self.backbone = backbone
        if backbone == "resnet50":
            resnet_checkpoints = models.ResNet50_Weights.DEFAULT
            resnet_model1 = models.resnet50(weights=resnet_checkpoints)
            # remove the last fc layer
            self.backbone_pose = torch.nn.Sequential(
                *list(resnet_model1.children())[:-2]
            )

            resnet_model2 = models.resnet50(weights=resnet_checkpoints)
            self.backbone_rot = torch.nn.Sequential(
                *list(resnet_model2.children())[:-1]
            )
            self.img_dim = 2048

        elif backbone == "hrnet_w48":
            pretrained_ckpt_path = "body_models/pose_hrnet_w48.pth"
            self.backbone_pose = hrnet_w48(
                pretrained_ckpt_path=pretrained_ckpt_path,
                downsample=True,
                use_conv=True,
            )
            self.backbone_rot = hrnet_w48(
                pretrained_ckpt_path=pretrained_ckpt_path,
                downsample=True,
                use_conv=True,
            )
            self.img_dim = 720

        self.conv_1x1 = nn.Conv2d(
            self.img_dim, self.transformer_config["model_dim"], kernel_size=1
        )

        # positional encodings
        self.position_encoding = build_position_encoding(
            pos_type="sine",
            hidden_dim=self.model_dim,
        )

        if not autoreg:
            self.indices_regressor = nn.Linear(self.model_dim + 9, 512)

        self.pose_dim = 17 * 3
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(self.img_dim + self.pose_dim, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.rot_predictor = nn.Linear(1024, 6)
        self.cam_predictor = nn.Linear(1024, 3)

    def forward(
        self,
        img,
        pose_init=None,
        mesh_decoder=None,
        joints_reg=None,
    ):
        device = img.device
        batch_size = img.size(0)

        # Rotation and camera prediction

        img_features_rot = self.backbone_rot(img)
        img_enc = self.avgpool(img_features_rot)
        img_enc = img_enc.view(img_enc.size(0), -1)

        xc = torch.cat([img_enc, pose_init], 1)
        xc = self.fc1(xc)
        xc = self.drop1(xc)
        xc = self.fc2(xc)
        xc = self.drop2(xc)

        pred_rot = self.rot_predictor(xc)
        pred_cam = self.cam_predictor(xc).view(-1, 3).cpu()

        # Mesh logit prediction
        mesh_tokens = self.mesh_token_embed.weight.unsqueeze(1).repeat(
            (1, batch_size, 1)
        )
        raw_img_features = self.backbone_pose(img)
        _, _, h, w = raw_img_features.shape
        img_features_pose = self.conv_1x1(raw_img_features).flatten(2).permute(2, 0, 1)

        pos_enc = (
            self.position_encoding(
                batch_size,
                int(math.sqrt(self.seq_len)),
                int(math.sqrt(self.seq_len)),
                device,
            )
            .flatten(2)
            .permute(2, 0, 1)
        )

        _, mesh_features = self.transformer(img_features_pose, mesh_tokens, pos_enc)

        if self.autoreg:
            mesh_logits = mesh_features.transpose(0, 1)
        else:
            mesh_features = mesh_features.transpose(0, 1)

            pred_rot_features = pred_rot.unsqueeze(1).repeat(1, 54, 1).to(mesh_features)
            pred_cam_features = pred_cam.unsqueeze(1).repeat(1, 54, 1).to(mesh_features)
            concat_features = torch.cat(
                [mesh_features, pred_rot_features, pred_cam_features], dim=-1
            )

            mesh_logits = self.indices_regressor(
                concat_features
            )  # batch_size X num_vertices X 1

        mesh_indices = torch.argmax(mesh_logits, dim=-1)
        with torch.no_grad():
            mesh_canonical = mesh_decoder.decode(mesh_indices)
        rotmat = rotation_6d_to_matrix(pred_rot)
        pred_v = (rotmat @ mesh_canonical.transpose(2, 1)).transpose(2, 1)
        J_regressor_batch = (
            joints_reg[None, :].expand(pred_v.shape[0], -1, -1).to(pred_v)
        )
        pose_init = torch.matmul(J_regressor_batch, pred_v).reshape(-1, 51)

        out = {}
        out["prob_indices"] = mesh_logits.cpu()
        out["pred_indices"] = mesh_indices.cpu()
        out["pred_mesh_canonical"] = mesh_canonical.cpu()
        out["pred_mesh"] = pred_v.cpu()
        out["pred_rot"] = pred_rot.cpu()
        out["pred_cam"] = pred_cam.cpu()

        return out

    def load(self, path_model: str):
        checkpoint = torch.load(path_model)
        new_state_dict = OrderedDict()
        for k, v in checkpoint["model"].items():
            if "module" in k:
                name = k[7:]  # remove `module.`
            else:
                name = k
            if "rotcam_feat" not in k:
                new_state_dict[name] = v
        self.load_state_dict(new_state_dict)
        loss = checkpoint["loss"]
        print(f"\t [Mesh regressor is loaded successfully with loss = {loss}]")
