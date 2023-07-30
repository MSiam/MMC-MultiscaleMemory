"""
Davis dataloader for inference.
"""
import math
import torch
import torch.utils.data
import os
from PIL import Image
import cv2
import numpy as np
import glob
import logging
import src.datasets.transforms as T
from src.datasets import path_config as dataset_path_config
from src.datasets.test.utils import sample_frame_indices

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Davis16ValDataset(torch.utils.data.Dataset):
    def __init__(self, num_frames=6, val_size=473, temporal_strides=None):
        super(Davis16ValDataset, self).__init__()
        self.num_frames = num_frames
        self.split = 'val'
        self._transforms = make_validation_transforms(val_size=val_size)

        self.temporal_strides = temporal_strides

        self.davis16_val_seqs_file = dataset_path_config.davis16_val_seqs_file
        self.davis16_rgb_path = dataset_path_config.davis16_rgb_path
        self.davis16_gt_path = dataset_path_config.davis16_gt_path
        self.frames_info = {
            'davis': {},
        }
        self.img_ids = []
        logger.debug('loading davis 16 val seqs...')
        with open(self.davis16_val_seqs_file, 'r') as f:
            video_names = f.readlines()
            video_names = [name.strip() for name in video_names]
            logger.debug('davis-val num of videos: {}'.format(len(video_names)))
            for video_name in video_names:
                frames = sorted(glob.glob(os.path.join(self.davis16_gt_path, video_name, '*.png')))
                self.frames_info['davis'][video_name] = [frame_path.split('/')[-1][:-4] for frame_path in frames]
                self.img_ids.extend([('davis', video_name, frame_index) for frame_index in range(len(frames))])

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_ids_i = self.img_ids[idx]
        dataset, video_name, frame_index = img_ids_i
        all_samples = None
        all_targets = None

        req_keys = ['frame_ids', 'masks', 'mask_paths']
        for stride in self.temporal_strides:
            samples, targets = self._load_data(dataset, video_name, frame_index, stride)
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

    def _load_data(self, dataset, video_name, frame_index, stride):
        vid_len = len(self.frames_info[dataset][video_name])
        center_frame_name = self.frames_info[dataset][video_name][frame_index]

        frame_indices = sample_frame_indices(stride, frame_index, self.num_frames, vid_len)

        assert len(frame_indices) == self.num_frames
        frame_ids = []
        img = []
        masks = []
        mask_paths = []
        for frame_id in frame_indices:
            frame_name = self.frames_info[dataset][video_name][frame_id]
            frame_ids.append(frame_name)
            img_path = os.path.join(self.davis16_rgb_path, video_name, frame_name + '.jpg')
            gt_path = os.path.join(self.davis16_gt_path, video_name, frame_name + '.png')
            img_i = Image.open(img_path).convert('RGB')
            img.append(img_i)
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            gt[gt > 0] = 255
            masks.append(torch.Tensor(np.expand_dims(np.asarray(gt.copy()), axis=0)))
            mask_paths.append(gt_path)
        masks = torch.cat(masks, dim=0)
        target = {'dataset': dataset, 'video_name': video_name, 'center_frame': center_frame_name,
                  'frame_ids': frame_ids, 'masks': masks, 'vid_len': vid_len, 'mask_paths': mask_paths}
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return torch.cat(img, dim=0), target


def make_train_transform(train_size=None):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomResize(scales, max_size=800),
        T.PhotometricDistort(),
        T.Compose([
            T.RandomResize([500, 600, 700]),
            T.RandomSizeCrop(473, 750),
            T.RandomResize([train_size], max_size=int(1.8 * train_size)),  # for r50
        ]),
        normalize,
    ])


def make_validation_transforms(val_size=360):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return T.Compose([
        T.RandomResize([val_size], max_size=int(val_size * 2)),
        normalize,
    ])
