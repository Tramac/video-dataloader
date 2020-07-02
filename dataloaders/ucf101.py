# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from vision import VisionDataset
from utils import make_dataset, VideoClips


class UCF101(VisionDataset):
    """
    Args:
        root (str): Root directory of the UCF101 Dataset.
        frames_per_clip (int): number of frames in a clip.
        fps (float): it will resample the video so that it has `fps`
        transform (callable, optional): A function/transform that  takes in a TxHxWxC video
            and returns a transformed version.

    Returns:
        video (Tensor[T, H, W, C]): the `T` video frames
        label (int): class of the video clip
    """
    def __init__(self, root, frames_per_clip, fps=4, video_width=171, video_height=128, 
                 crop_size=112, transform=None):
        super(UCF101, self).__init__(root)
        extensions = ('avi',)

        classes = list(sorted(os.listdir(root)))
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file=None)
        video_list = [x[0] for x in self.samples]
        print('Number of videos: {:d}'.format(len(self.samples)))


        clip_root = root.split('/')[-1].lower()
        self.video_clips = VideoClips(
            clip_root,
            video_list,
            frames_per_clip,
            fps = fps,
            video_width = video_width,
            video_height = video_height,
            crop_size = crop_size,
        )
        self.transform = transform


    def __getitem__(self, idx):
        video = self.video_clips.get_clip(idx)
        if self.transform is not None:
            video = self.transform(video)
        label = self.samples[idx][1]
        return video, label

    def __len__(self):
        return len(self.samples)

    def normalize(self, frames):
        for i, frame in enumerate(frames):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            frames[i] = frame

        return frames

    def to_tensor(self, inputs):
        return inputs.transpose((3, 0, 1, 2))


if __name__ == '__main__':
    dataset = UCF101('../UCF101/', 16)
    print(len(dataset))
