# -*- coding: utf-8 -*-
import os
import sys
import random
import subprocess
import numpy as np
from tqdm import tqdm
from PIL import Image
PYTHON3 = False if sys.version_info < (3, 0) else True
if not PYTHON3:
    import commands


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)

def make_dataset(dir, class_to_idx, extensions=None, is_valid_file=None):
    items = []
    dir = os.path.expanduser(dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = (path, class_to_idx[target])
                    items.append(item)
    return items

class VideoClips(object):
    """
    Given a list of video files, computes the subvideos of size 'frames_per_clip',
    Arguments:
        clip_dir (List[str]): paths to the frame files
        video_paths (List[str]): paths to the video files
        clip_length_in_frames (int): size of a clip in number of frames
        video_width (int): width of the frames
        video_height (int): height of the frames
        crop_size (int): size of the clip
        fps (float): resample all the videos
    """
    def __init__(self, clip_dir, video_paths, clip_length_in_frames, 
                 fps, video_width, video_height, crop_size):
        self.video_paths = video_paths
        self.clip_len = clip_length_in_frames
        self.video_width = video_width
        self.video_height = video_height
        self.crop_size = crop_size
        self.fps = fps

        # path to ffmpeg
        self.ffmpeg = 'ffmpeg'
        # path to ffprobe
        self.ffprobe = 'ffprobe'

        self.frame_paths = []
        if self._check_frames(clip_dir):
            for path in sorted(os.listdir(clip_dir)):
                self.frame_paths.append(os.path.join(clip_dir, path))
        else:
            print('Preprocessing of dataset...')
            for video_path in tqdm(video_paths):
                frame_path = self.cutframe(video_path, clip_dir)
                self.frame_paths.append(frame_path)


    def get_clip(self, idx):
        """
        Gets a subclip from a list of videos.
        Arguments:
            idx (int): index of the subclip. Must be between 0 and num videos.
        Returns:
            video (Tensor)
            label (Tensor)
        """
        frame_path = self.frame_paths[idx]
        video = self.read_frames(frame_path)
        clip = self.crop_clip(video, self.clip_len, self.crop_size)
        return clip

    def cutframe(self, video, save_dir, step=None):
        """videos to frames"""
        video_filename = os.path.basename(video).split('.')[0]
        save_dir = os.path.join(save_dir, video_filename)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if step is not None:
            real_fps = self._get_video_fps(video)
            fps = real_fps / float(step)
        cmd = "{} -threads {} -i '{}' -vf fps={} -s {}x{} -q:v 2 -v quiet -y {}/%5d.jpg".format(
            self.ffmpeg, 8, video, self.fps, self.video_width, self.video_height, save_dir)
        subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, shell=True).communicate()
        return save_dir

    def read_frames(self, frame_path):
        """
        Gets a 3d nparray from all frames.
        """
        frames = sorted([os.path.join(frame_path, img) for img in os.listdir(frame_path)])
        num_frames = len(frames)
        video = np.empty((num_frames, self.video_height, self.video_width, 3), np.dtype('float32'))
        # torch.utils.datasets.dataloader?
        for i, frame_name in enumerate(frames):
            frame = Image.open(frame_name).convert('RGB')
            video[i] = frame
        return video

    def crop_clip(self, video, clip_len, crop_size):
        """
        """
        # randomly select time index for temporal jittering
        time_idx = random.randint(0, video.shape[0] - clip_len)

        # Randomly select start indices in order to crop the video
        height_idx = random.randint(0, video.shape[1] - crop_size)
        width_idx = random.randint(0, video.shape[2] - crop_size)

        # Crop and jitter the video using indexing. The spatial crop is performed on
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        clip = video[time_idx: time_idx + clip_len,
                     height_idx: height_idx + crop_size,
                     width_idx: width_idx + crop_size, :]

        return clip

    def _get_video_fps(self, mp4):
        cmd = r" 2>&1 | grep 'Stream' |  grep 'Video'"
        cmd = self.ffprobe + ' ' + mp4 + cmd
        if PYTHON3:
            (status, output) = subprocess.getstatusoutput(cmd)
        else:
            (status, output) = commands.getstatusoutput(cmd)
        fps = None
        if status == 0 and output != '':
            fps = output
        if fps is not None:
            fps = re.search(r'\d+ fps', fps)
            if fps is not None:
                fps_v = fps.group(0).split()
                if len(fps_v) == 2:
                    fps = int(fps_v[0])
                else:
                    fps = None
        return fps

    def _check_frames(self, save_dir):
        """check if have cutframe"""
        if not os.path.exists(save_dir):
            return False

        if not os.listdir(save_dir):
            return False

        return True
