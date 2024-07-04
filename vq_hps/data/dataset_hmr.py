"""
This file implements the dataloader for datasets with images and SMPL annotations.

Part of this code is inspired from the dataloaders of BEDLAM (https://github.com/pixelite1201/BEDLAM/blob/master/train/dataset/dataset_smpl.py) 
and FastMETRO (https://github.com/postech-ami/FastMETRO/blob/main/src/datasets/human_mesh_tsv.py)
"""

import sys

sys.path.append("./")

from torch.utils.data import Dataset
from torchvision.transforms import Normalize
import cv2
import numpy as np
from vq_hps.utils.augmentations import (
    crop,
    rot_aa,
    flip_img,
    flip_pose,
    transform,
    flip_kp,
    flip_17,
)
import albumentations as A
import torch

from vq_hps.utils.smpl_utils import get_body_model_sequence
import random

import cv2

J24_TO_J17 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 14, 16, 17]


class DatasetHMR(Dataset):
    def __init__(
        self,
        dataset_file: str,
        augment: bool = True,
        flip: bool = True,
        proportion: float = 1.0,
    ):
        """Initializes the dataset.

        Args:
            dataset_file (str): Path to the dataset file (should be a .npz file)
            augment (bool, optional): If augment is True, augmentations will be performed during training. Defaults to True.
            flip (bool, optional): If True, images will be randomly flipped during training. Defaults to True.
            proportion (float, optional): Proportion of the dataset used (to reduce the validation for instance). Defaults to 1.0.
        """
        super().__init__()

        self.dataset_file = dataset_file.split("/")[-1]

        self.data = np.load(dataset_file, allow_pickle=True)

        self.imgname = self.data["imgname"]

        full_len = len(self.imgname)

        new_len = full_len
        sampled_indices = [x for x in range(full_len)]
        if proportion != 1:
            new_len = int(proportion * full_len)
            sampled_indices = random.sample(range(full_len), new_len)
            self.imgname = self.data["imgname"][sampled_indices]

        self.scale = self.data["scale"][sampled_indices]
        self.center = self.data["center"][sampled_indices]

        if "pose_cam" in self.data:
            self.full_pose = self.data["pose_cam"][sampled_indices]
        elif "pose" in self.data:
            self.full_pose = self.data["pose"][sampled_indices]
        else:
            root_orient = self.data["root_orient"][sampled_indices]
            pose_body = self.data["pose_body"][sampled_indices]
            self.full_pose = np.concatenate([root_orient, pose_body], axis=-1)

        if "gender" in self.data:
            self.gender = self.data["gender"][sampled_indices]
        else:
            self.gender = ["neutral" for _ in range(new_len)]

        if "j2d" in self.data:
            self.j2d = self.data["j2d"][sampled_indices]
        else:
            self.j2d = self.data["part"][sampled_indices]

        if "betas" in self.data:
            self.betas = self.data["betas"][sampled_indices]
        else:
            self.betas = self.data["shape"][sampled_indices]

        self.is_train = augment

        print(f"{len(self.imgname)} training samples found")

        self.rot_factor = 30
        self.noise_factor = 0.4
        self.scale_factor = 0.25
        self.flip = flip

        self.normalize_vqvae = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.normalize_resnet = Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.imgname)

    def augm_params(self):
        """Get augmentation parameters."""
        rot = 0  # rotation
        sc = 1  # scaling
        flip = 0
        if self.is_train:
            if np.random.uniform() <= 0.5 and self.flip:
                flip = 1

            # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
            rot = min(
                2 * self.rot_factor,
                max(-2 * self.rot_factor, np.random.randn() * self.rot_factor),
            )

            # The scale is multiplied with a number
            # in the area [1-scaleFactor,1+scaleFactor]
            sc = min(
                1 + self.scale_factor,
                max(1 - self.scale_factor, np.random.randn() * self.scale_factor + 1),
            )
            # but it is zero with probability 3/5
            if np.random.uniform() <= 0.6:
                rot = 0

        return flip, rot, sc

    def rgb_processing(self, rgb_img, center, scale, rot, flip):
        """Process the image"""
        if self.is_train:
            aug_comp = [
                A.Downscale(0.5, 0.9, interpolation=0, p=0.1),
                A.ImageCompression(20, 100, p=0.1),
                A.RandomRain(blur_value=4, p=0.1),
                A.MotionBlur(blur_limit=(3, 15), p=0.2),
                A.Blur(blur_limit=(3, 10), p=0.1),
                A.RandomSnow(
                    brightness_coeff=1.5, snow_point_lower=0.2, snow_point_upper=0.4
                ),
            ]
            aug_mod = [
                A.CLAHE((1, 11), (10, 10), p=0.2),
                A.ToGray(p=0.2),
                A.RandomBrightnessContrast(p=0.2),
                A.MultiplicativeNoise(
                    multiplier=[0.5, 1.5], elementwise=False, per_channel=True, p=0.2
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    always_apply=False,
                    p=0.2,
                ),
                A.Posterize(p=0.1),
                A.RandomGamma(gamma_limit=(80, 200), p=0.1),
                A.Equalize(mode="cv", p=0.1),
            ]
            albumentation_aug = A.Compose(
                [A.OneOf(aug_comp, p=0.3), A.OneOf(aug_mod, p=0.3)]
            )
            rgb_img = albumentation_aug(image=rgb_img)["image"]

        rgb_img = crop(rgb_img, center, scale, [224, 224], rot=rot)

        if flip:
            rgb_img = flip_img(rgb_img)

        rgb_img = np.transpose(rgb_img.astype("float32"), (2, 0, 1)) / 255.0
        return rgb_img

    def pose_processing(self, pose, r, f):
        """Process SMPL theta parameters  and apply all augmentation transforms."""
        # rotation or the pose parameters
        pose = pose.astype("float32")
        pose[:3] = rot_aa(pose[:3], r)
        # flip the pose parameters
        if f:
            pose = flip_pose(pose)
        # (72),float
        pose = pose.astype("float32")
        return pose

    def j2d_processing(self, kp, center, scale, r, f):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i, 0:2] = transform(kp[i, 0:2] + 1, center, scale, [224, 224], rot=r)
        # convert to normalized coordinates
        kp[:, :-1] = 2.0 * kp[:, :-1] / 224 - 1.0
        # flip the x coordinates
        if f:
            if kp.shape[0] == 24:
                kp = flip_kp(kp)
            elif kp.shape[0] == 17:
                kp = flip_17(kp)
            else:
                print(
                    "Error, the skeleton to be flipped must be in SMPL or COCO25 format"
                )
        kp = kp.astype("float32")
        return kp

    def __getitem__(self, index):
        imgname = self.imgname[index]
        is_3dpw = "3DPW" in imgname or "EMDB" in imgname
        if not is_3dpw:
            img_path = f"datasets/{self.dataset_file[:-4]}/{imgname}"
        else:
            img_path = imgname
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if "closeup" in img_path:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        center = self.center[index]
        if not is_3dpw:
            scale = self.scale[index]
        else:
            scale = 1 / self.scale[index]

        flip, rot, sc = self.augm_params()
        img = self.rgb_processing(img, center, sc * scale, rot, flip)

        img = torch.from_numpy(img).float()
        resnet_img = self.normalize_resnet(img)

        j2d = self.j2d[index][:24]
        if not is_3dpw:
            j2d = j2d[J24_TO_J17]
            j2d = self.j2d_processing(j2d, center, sc * scale, rot, flip)
            j2d_full = np.zeros((24, 3))
            j2d_full[:17] = j2d
        else:
            j2d_full = self.j2d_processing(j2d, center, sc * scale, rot, flip)

        betas = self.betas[index][:10]
        gender = str(self.gender[index])
        full_pose = self.full_pose[index][:66]
        pose = self.pose_processing(full_pose, rot, flip)

        root_orient = pose[:3]
        pose_body = pose[3:]

        root_orient = np.expand_dims(root_orient, 0)
        pose_body = np.expand_dims(pose_body, 0)

        local_mesh = get_body_model_sequence(
            gender=gender,
            num_frames=1,
            betas=betas,
            pose_body=pose_body,
            num_betas=10,
            normalize=True,
            cpu=True,
        )

        global_mesh = get_body_model_sequence(
            gender=gender,
            num_frames=1,
            betas=betas,
            pose_body=pose_body,
            root_orient=root_orient,
            num_betas=10,
            normalize=True,
            cpu=True,
        )

        item = {}
        item["img"] = resnet_img
        item["imgname"] = img_path
        item["raw_img"] = img
        item["j2d"] = j2d
        item["gender"] = gender
        item["mesh"] = torch.from_numpy(global_mesh[0]).float()
        item["local_mesh"] = torch.from_numpy(local_mesh[0]).float()
        item["rotation"] = torch.from_numpy(root_orient[0]).float()
        item["is_3dpw"] = is_3dpw

        return item
