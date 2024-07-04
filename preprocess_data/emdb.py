import pickle as pkl
import numpy as np
import os
from pathlib import Path
import cv2
import torch
import tqdm
import sys

sys.path.append(".")
from vq_hps import batch_rodrigues, rotation_matrix_to_angle_axis

import random


def pkl_to_npz(dataset_path):

    list_pkl = [p for p in Path(f"{dataset_path}").glob(f"**/*.pkl")]

    save_path = os.path.join(dataset_path, "test.npz")
    imgname_list = []
    gender_list = []
    root_orient_list = []
    pose_list = []
    betas_list = []
    center_list = []
    scale_list = []
    j2d_list = []
    for data_file in tqdm.tqdm(list_pkl):
        dir_path = os.path.dirname(data_file)
        data = pkl.load(open(data_file, "rb"), encoding="latin1")
        if data["emdb1"]:
            img_path = os.path.join(dir_path, "images")
            gender = data["gender"]
            betas = data["smpl"]["betas"]
            for i in range(data["n_frames"]):
                if data["good_frames_mask"][i]:
                    if i not in data["bboxes"]["invalid_idxs"]:
                        imgname = os.path.join(img_path, "{:05d}.jpg".format(i))
                        imgname_list.append(imgname)
                        gender_list.append(gender)

                        root_orient = data["smpl"]["poses_root"][i]
                        rotmat = batch_rodrigues(
                            torch.from_numpy(np.array(root_orient)).unsqueeze(0)
                        ).reshape(1, 3, 3)
                        cam_ext = data["camera"]["extrinsics"][i]
                        cam_rot = torch.from_numpy(cam_ext[:3, :3]).unsqueeze(0)
                        rotmat = torch.bmm(cam_rot, rotmat)
                        root_orient_list.append(
                            rotation_matrix_to_angle_axis(rotmat)
                            .squeeze(0)
                            .cpu()
                            .numpy()
                        )

                        pose_list.append(data["smpl"]["poses_body"][i])
                        betas_list.append(betas)

                        j2d = np.ones((24, 3))
                        j2d[:, :2] = data["kp2d"][i]
                        j2d_list.append(j2d)

                        bbox = data["bboxes"]["bboxes"][i]
                        center = [(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2]
                        center_list.append(center)
                        scale = 1.2 * max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200
                        scale_list.append(1 / scale)

        np.savez(
            save_path,
            betas=betas_list,
            center=center_list,
            gender=gender_list,
            imgname=imgname_list,
            j2d=j2d_list,
            pose_body=pose_list,
            root_orient=root_orient_list,
            scale=scale_list,
        )


pkl_to_npz("datasets/EMDB")
