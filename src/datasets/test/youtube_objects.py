"""
Custom dataloader for YouTubeObjects dataset.
"""
import math
import torch
import torch.utils.data
import os
from PIL import Image
import glob
import logging

import src.datasets.transforms as T
from src.datasets import path_config as dataset_path_config
from src.datasets.test.utils import sample_frame_indices
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class YouTubeObjects(torch.utils.data.Dataset):

    def __init__(self, num_frames=6, min_size=360, seq_name=None, temporal_strides=None):
        super(YouTubeObjects, self).__init__()
        self.num_frames = num_frames
        self.dataset_path = dataset_path_config.youtube_objects_dataset_path
        self.files_list = dataset_path_config.youtube_objects_val_set_file
        self._transforms = make_validation_transforms(min_size=min_size)

        self.temporal_strides = temporal_strides

        self.frames_info = {}
        self.img_ids = []
        if seq_name is None:
            with open(self.files_list, 'r') as f:
                video_names = f.readlines()
                video_names = [name.strip() for name in video_names]
        else:
            video_names = [seq_name]
        for video_name in video_names:
            frames = sorted(glob.glob(os.path.join(self.dataset_path, video_name, '*.jpg')))
            self.frames_info[video_name] = [frame_path.split('/')[-1][:-4] for frame_path in frames]
            self.img_ids.extend([(video_name, frame_index) for frame_index in range(len(frames))])

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_ids_i = self.img_ids[idx]
        video_name, frame_index = img_ids_i
        all_samples = None
        all_targets = None

        req_keys = ['frame_ids', 'img_paths']
        for stride in self.temporal_strides:
            samples, targets = self._load_data(video_name, frame_index, stride)
            if all_samples is None:
                all_samples = samples
                all_targets = targets
            else:
                all_samples = torch.cat((all_samples, samples), dim=0)
                for k in req_keys:
                    if type(targets[k]) == list:
                        all_targets[k] += targets[k]
                    else:
                        all_targets[k] = torch.cat((all_targets[k], targets[k]), dim=0)
        return all_samples, all_targets

    def _load_data(self, video_name, frame_index, stride):
        img = []
        img_paths = []
        vid_len = len(self.frames_info[video_name])
        center_frame_name = self.frames_info[video_name][frame_index]
        frame_indices = sample_frame_indices(stride, frame_index, self.num_frames, vid_len)
        assert len(frame_indices) == self.num_frames
        frame_ids = []

        for frame_id in frame_indices:
            frame_name = self.frames_info[video_name][frame_id]
            frame_ids.append(frame_name)
            img_path = os.path.join(self.dataset_path, video_name, frame_name + '.jpg')
            img_i = Image.open(img_path).convert('RGB')
            img.append(img_i)
            img_paths.append(img_path)
        target = {'video_name': video_name, 'center_frame': center_frame_name, 'frame_ids': frame_ids,
                  'vid_len': vid_len, 'img_paths': img_paths}
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return torch.cat(img, dim=0), target


def make_validation_transforms(min_size=360):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return T.Compose([
        T.RandomResize([min_size], max_size=int(min_size * 1.8)),
        normalize,
    ])
