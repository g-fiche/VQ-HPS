import sys

sys.path.append(".")

import os
import torch
import numpy as np
import pickle as pkl
import os.path as osp
from tqdm import tqdm

from vq_hps import get_smooth_bbox_params
from vq_hps import batch_rodrigues, rotation_matrix_to_angle_axis


VIS_THRESH = 0.3


def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / cam_coord[:, 2] * f[0] + c[0]
    y = cam_coord[:, 1] / cam_coord[:, 2] * f[1] + c[1]
    z = cam_coord[:, 2]
    return np.stack((x, y, z), 1)


def read_smpl_data(folder, set):

    sequences = [
        x.split(".")[0] for x in os.listdir(osp.join(folder, "sequenceFiles", set))
    ]

    imgname_list = []
    center_list = []
    scale_list = []
    gender_list = []
    betas_list = []
    root_orient_list = []
    pose_body_list = []
    j2d_list = []

    save_path = f"{folder}/3DPW_{set}.npz"

    for seq in tqdm(sequences):
        data_file = osp.join(folder, "sequenceFiles", set, seq + ".pkl")

        data = pkl.load(open(data_file, "rb"), encoding="latin1")

        img_dir = osp.join(folder, "imageFiles", seq)

        num_people = len(data["poses"])
        num_frames = len(data["img_frame_ids"])
        assert data["poses2d"][0].shape[0] == num_frames

        cam_ext = data["cam_poses"]
        cam_rot = cam_ext[:, :3, :3]

        intrinsics = data["cam_intrinsics"]
        focal, princpt = [intrinsics[0, 0], intrinsics[1, 1]], [
            intrinsics[0, 2],
            intrinsics[1, 2],
        ]

        valid = np.array(data["campose_valid"]).astype(bool)

        for p_id in range(num_people):
            j2d = data["poses2d"][p_id].transpose(0, 2, 1)

            img_paths = []
            for i_frame in range(num_frames):
                img_path = os.path.join(img_dir + "/image_{:05d}.jpg".format(i_frame))
                img_paths.append(img_path)

            bbox_params, time_pt1, time_pt2 = get_smooth_bbox_params(
                j2d, vis_thresh=VIS_THRESH, sigma=8
            )

            # process bbox_params
            c_x = bbox_params[:, 0]
            c_y = bbox_params[:, 1]
            scale_seq = bbox_params[:, 2]
            center_seq = np.vstack([c_x, c_y]).T

            img_paths_array = np.array(img_paths)[time_pt1:time_pt2]

            root_orient_seq = data["poses"][p_id][:, :3]
            rotmat_seq = batch_rodrigues(torch.from_numpy(root_orient_seq)).reshape(
                -1, 3, 3
            )
            rotmat_seq = torch.bmm(torch.from_numpy(cam_rot), rotmat_seq)
            root_orient_seq = rotation_matrix_to_angle_axis(rotmat_seq).cpu().numpy()

            pose_body_seq = data["poses"][p_id][:, 3:66]
            betas = data["betas"][p_id][:10]
            gender = data["genders"][p_id]

            j3d = data["jointPositions"][p_id]
            kpts2d = torch.zeros((j3d.shape[0], 24, 3))
            for t in range(j3d.shape[0]):
                kpts3d = (
                    np.dot(cam_rot[t], j3d[t].reshape(-1, 3).transpose(1, 0)).transpose(
                        1, 0
                    )
                    + cam_ext[t][:-1, -1]
                )

                kpts2d[t] = torch.Tensor(cam2pixel(kpts3d, focal, princpt)).transpose(
                    1, -1
                )

            valid_i = valid[p_id]

            for t, (
                img_path,
                center,
                scale,
                root_orient,
                pose_body,
                j2d,
            ) in enumerate(
                zip(
                    img_paths_array,
                    center_seq,
                    scale_seq,
                    root_orient_seq,
                    pose_body_seq,
                    kpts2d,
                )
            ):
                if valid_i[t]:
                    imgname_list.append(img_path.replace("\\", "/"))
                    center_list.append(center)
                    scale_list.append(scale)
                    gender_list.append(gender)
                    betas_list.append(betas)
                    root_orient_list.append(root_orient)
                    pose_body_list.append(pose_body)
                    j2d_list.append(j2d.numpy())

    np.savez(
        save_path,
        imgname=imgname_list,
        center=center_list,
        scale=scale_list,
        gender=gender_list,
        betas=betas_list,
        root_orient=root_orient_list,
        pose_body=pose_body_list,
        j2d=j2d_list,
    )


read_smpl_data("datasets/3DPW", "test")
read_smpl_data("datasets/3DPW", "validation")
read_smpl_data("datasets/3DPW", "train")
