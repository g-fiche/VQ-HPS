"""Inspired from https://github.com/facebookresearch/eft/blob/main/eft/db_processing/h36m.py"""

import os
import cv2
import glob
import numpy as np
import mat73
import matplotlib.pyplot as plt
from tqdm import tqdm


def h36m_train_extract(src_path, set, labels):
    # users in validation set
    if set == "training/subjects":
        user_list = [1, 5, 6, 7, 8]
    else:
        user_list = [9, 11]

    set_path = os.path.join(src_path, set)

    data = np.load(labels)
    valid_frames = data["imgname"]

    # go over each user
    for user_i in user_list:
        user_name = "S%d" % user_i
        print(user_name)
        user_path = os.path.join(set_path, user_name)

        imgname_user = f"Images/{user_name}"

        # path with GT bounding boxes
        bbox_path = os.path.join(user_path, "MySegmentsMat", "ground_truth_bb")
        # path with videos
        vid_path = os.path.join(user_path, "Videos")

        # go over all the sequences of each user
        seq_list = glob.glob(os.path.join(bbox_path, "*.mat"))
        seq_list.sort()
        for seq_i in tqdm(seq_list):
            # sequence info
            seq_name = seq_i.split("/")[-1]
            imgname_seq = os.path.join(imgname_user, seq_name[:-4])

            action, camera, _ = seq_name.split(".")
            action = action.replace(" ", "_")
            # irrelevant sequences
            if action == "_ALL":
                continue

            # bbox file
            # bbox_h5py = h5py.File(seq_i, "r")
            bbox_data = mat73.loadmat(seq_i)

            # video file
            vid_file = os.path.join(vid_path, seq_name.replace("mat", "mp4"))
            vidcap = cv2.VideoCapture(vid_file)

            # go over each frame of the sequence
            for frame_i in range(len(bbox_data["Masks"])):
                # read video frame
                success, image = vidcap.read()
                if not success:
                    break

                imgname = str(os.path.join(imgname_seq, "%06d.jpg" % (frame_i + 1)))
                imgname = imgname.replace(" ", "_")

                if imgname in valid_frames:
                    dest_fpath = f"datasets/h36m_test/{imgname}"
                    os.makedirs(os.path.dirname(dest_fpath), exist_ok=True)
                    plt.imsave(dest_fpath, np.flip(image, 2))


h36m_train_extract(
    "/media/fast/LaCie/data/video_datasets/Human3.6M",
    "testing/subject",
    "datasets/h36m_test/h36m_test.npz",
)
