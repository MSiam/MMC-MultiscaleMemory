"""
Davis dataloader with train data augmented by ytvos.
# Added by @RK
"""
import math
from collections import defaultdict
import torch
import torch.utils.data
import os
from PIL import Image
import cv2
import numpy as np
import glob
import socket
import logging

import src.datasets.transforms as T

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class PathConfig:
    """
    This config is used to allow read data from local copy when available or from data server.
    Or even some data from local and some from the data server. Just add one more line for each path using the hostname
    as the key for whatever data folder has a local copy.
    It may have too many path config, but I found that is easier to load different data from different places, like images in one place,
    and soenet feature in another place.
    If you have riemann mounted on your machine, you can just run it and all data will be loaded from my data on riemann.
    """

    def __init__(self):
        # default paths from the data server riemann
        self.davis16_train_seqs_file = defaultdict(
            lambda: os.path.join('/local/riemann/home/rezaul/datasets/DAVIS-data/DAVIS_2016/train_seqs.txt'))
        self.davis16_val_seqs_file = defaultdict(
            lambda: os.path.join('/local/riemann/home/rezaul/datasets/DAVIS-data/DAVIS_2016/val_seqs.txt'))
        self.davis16_rgb_path = defaultdict(
            lambda: os.path.join('/local/riemann/home/rezaul/datasets/DAVIS-data/DAVIS_2016/JPEGImages/480p'))
        self.davis16_gt_path = defaultdict(
            lambda: os.path.join('/local/riemann/home/rezaul/datasets/DAVIS-data/DAVIS_2016/Annotations/480p'))
        self.davis16_soenet_feat_path = defaultdict(lambda: os.path.join(
            '/local/riemann/home/rezaul/datasets/DAVIS-data/DAVIS_2016_SOENet/soenet_g2_5taps_480p'))
        # newly added feature for whole video in one file, shape THWC
        self.davis16_rgb_path_npy = defaultdict(
            lambda: os.path.join('/local/riemann1/home/rezaul/datasets_vos/davis2016/JPEGImagesNPY/480p'))
        self.davis16_gt_path_npy = defaultdict(
            lambda: os.path.join('/local/riemann1/home/rezaul/datasets_vos/davis2016/AnnotationsNPY/480p'))
        self.davis16_soenet_feat_thwc_path = defaultdict(lambda: os.path.join(
            '/local/riemann1/home/rezaul/datasets_vos/davis16_soenet_feats/soenet_g2_5taps_480p_thwc_cat'))
        self.davis16_msoenet_feat_thwc_path = defaultdict(lambda: os.path.join(
            '/local/riemann1/home/rezaul/datasets_vos/davis16_soenet_feats/msoenet_g2_3taps_480p_thwc_cat'))

        self.ytvos19_rgb_path = defaultdict(
            lambda: os.path.join('/local/riemann/home/rezaul/dataset2/youtube-vos-2019/train/JPEGImages'))
        self.ytvos19_gt_path = defaultdict(
            lambda: os.path.join('/local/riemann/home/rezaul/dataset2/youtube-vos-2019/train/Annotations'))
        self.ytvos19_soenet_feat_path = defaultdict(lambda: os.path.join(
            '/local/riemann/home/rezaul/dataset2/youtube-vos-2019-soenet-feats/train/soenet_feats/soenet_g2_5taps'))
        # newly added feature for whole video in one file, shape THWC
        self.ytvos19_rgb_path_npy = defaultdict(
            lambda: os.path.join('/local/riemann1/home/rezaul/datasets_vos/youtube-vos-2019/train/JPEGImagesNPY'))
        self.ytvos19_gt_path_npy = defaultdict(
            lambda: os.path.join('/local/riemann1/home/rezaul/datasets_vos/youtube-vos-2019/train/AnnotationsNPY'))
        self.ytvos19_soenet_feat_thwc_path = defaultdict(lambda: os.path.join(
            '/local/riemann1/home/rezaul/datasets_vos/youtube-vos-2019-soenet-feats/soenet_g2_5taps_480p_thwc_cat'))
        self.ytvos19_msoenet_feat_thwc_path = defaultdict(lambda: os.path.join(
            '/local/riemann1/home/rezaul/datasets_vos/youtube-vos-2019-soenet-feats/train_msoenet_g2_3taps'))

        # host specific local paths ... having entry for riemann here is kind of redundant, but kept here for reference.
        self.davis16_train_seqs_file.update({
            'riemann': os.path.join('/local/riemann/home/rezaul/datasets/DAVIS-data/DAVIS_2016/train_seqs.txt'),
            'euler': os.path.join('/local/data1/rezaul/dataset_vos/DAVIS-data/DAVIS_2016/train_seqs.txt'),
            'alkhwarizmi': os.path.join('/local/riemann1/home/msiam/medvt_datasets/DAVIS_2016/train_seqs.txt'),
            'fourier': os.path.join('/local/data0/datasets_vos/DAVIS-data/DAVIS_2016/train_seqs.txt'),
            'vector': os.path.join('/ssd003/home/msiam/projects/video-transformers/medvt2/train_seqs.txt')
        })
        self.davis16_val_seqs_file.update({
            'riemann': os.path.join('/local/riemann/home/rezaul/datasets/DAVIS-data/DAVIS_2016/val_seqs.txt'),
            'euler': os.path.join('/local/data1/rezaul/dataset_vos/DAVIS-data/DAVIS_2016/val_seqs.txt'),
            'alkhwarizmi': os.path.join('/local/riemann1/home/msiam/medvt_datasets/DAVIS_2016/val_seqs.txt'),
            'fourier': os.path.join('/local/data0/datasets_vos/DAVIS-data/DAVIS_2016/val_seqs.txt'),
            'vector': os.path.join('/ssd003/home/msiam/projects/video-transformers/medvt2/val_seqs.txt')
        })
        self.davis16_rgb_path.update({
            'riemann': os.path.join('/local/riemann/home/rezaul/datasets/DAVIS-data/DAVIS_2016/JPEGImages/480p'),
            'euler': os.path.join('/local/data1/rezaul/dataset_vos/DAVIS-data/DAVIS_2016/JPEGImages/480p'),
            'alkhwarizmi': os.path.join('/local/riemann1/home/msiam/medvt_datasets/DAVIS_2016/JPEGImages/480p'),
            'fourier': os.path.join('/local/data0/datasets_vos/DAVIS-data/DAVIS_2016/JPEGImages/480p'),
            'vector': os.path.join('/scratch/ssd004/datasets/DAVIS/JPEGImages/480p/')
        })
        self.davis16_gt_path.update({
            'riemann': os.path.join('/local/riemann/home/rezaul/datasets/DAVIS-data/DAVIS_2016/Annotations/480p'),
            'euler': os.path.join('/local/data1/rezaul/dataset_vos/DAVIS-data/DAVIS_2016/Annotations/480p'),
            'alkhwarizmi': os.path.join('/local/riemann1/home/msiam/medvt_datasets/DAVIS_2016/Annotations/480p'),
            'fourier': os.path.join('/local/data0/datasets_vos/DAVIS-data/DAVIS_2016/Annotations/480p'),
            'vector': os.path.join('/scratch/ssd004/datasets/DAVIS/Annotations/480p/')
        })
        '''
        self.davis16_rgb_path_npy.update({
            'riemann': os.path.join('/local/riemann/home/rezaul/datasets/DAVIS-data/DAVIS_2016/JPEGImagesNPY/480p'),
            # 'euler': os.path.join('/local/data1/rezaul/dataset_vos/DAVIS-data/DAVIS_2016/JPEGImagesNPY/480p'),
            # 'fourier': os.path.join('/local/data0/datasets_vos/DAVIS-data/DAVIS_2016/JPEGImagesNPY/480p'),
        })
        self.davis16_gt_path_npy.update({
            'riemann': os.path.join('/local/riemann/home/rezaul/datasets/DAVIS-data/DAVIS_2016/AnnotationsNPY/480p'),
            # 'euler': os.path.join('/local/data1/rezaul/dataset_vos/DAVIS-data/DAVIS_2016/AnnotationsNPY/480p'),
            # 'fourier': os.path.join('/local/data0/datasets_vos/DAVIS-data/DAVIS_2016/AnnotationsNPY/480p'),
        })
        '''

        self.davis16_soenet_feat_path.update({
            'riemann': os.path.join(
                '/local/riemann/home/rezaul/datasets/DAVIS-data/DAVIS_2016_SOENet/soenet_g2_5taps_480p'),
            'euler': os.path.join('/local/data1/rezaul/dataset_vos/DAVIS-data/DAVIS_2016_SOENet/soenet_g2_5taps_480p'),
            'alkhwarizmi': os.path.join('/local/riemann1/home/msiam/medvt_datasets/DAVIS_2016_SOENet/soenet_g2_5taps_480p'),
            # 'fourier': os.path.join('/local/data1/rezaul/datasets_vos/DAVIS-data/DAVIS_2016_SOENet/soenet_g2_5taps_480p'),
        })

        self.ytvos19_rgb_path.update({
            'riemann': os.path.join('/local/riemann/home/rezaul/dataset2/youtube-vos-2019/train/JPEGImages'),
            'euler': os.path.join('/local/data0/rezaul/dataset_ytvos/youtube-vos-2019/train/JPEGImages'),
            'alkhwarizmi': os.path.join('/local/riemann1/home/msiam/medvt_datasets/youtube-vos-2019/train/JPEGImages'),
            'fourier': os.path.join('/local/data0/datasets_vos/youtube-vos-2019/train/JPEGImages'),
            'vector': os.path.join('/scratch/ssd004/datasets/YouTube-VOS/train/JPEGImages/')
        })
        self.ytvos19_gt_path.update({
            'riemann': os.path.join('/local/riemann/home/rezaul/dataset2/youtube-vos-2019/train/Annotations'),
            'euler': os.path.join('/local/data0/rezaul/dataset_ytvos/youtube-vos-2019/train/Annotations'),
            'alkhwarizmi': os.path.join('/local/riemann1/home/msiam/medvt_datasets/youtube-vos-2019/train/Annotations'),
            'fourier': os.path.join('/local/data0/datasets_vos/youtube-vos-2019/train/Annotations'),
            'vector': os.path.join('/scratch/ssd004/datasets/YouTube-VOS/train/Annotations/')
        })
        self.ytvos19_soenet_feat_path.update({
            # This is too big, like a few terra-bytes, so I am using one copy from riemann now, may take local copy later.
            'riemann': os.path.join(
                '/local/riemann/home/rezaul/dataset2/youtube-vos-2019-soenet-feats/train/soenet_feats/soenet_g2_5taps'),
            # 'euler': os.path.join('/local/data0/rezaul/dataset_ytvos/youtube-vos-2019-soenet-feats/train/soenet_g2_5taps'),
        })


class Davis16_YTVOS_DataLoaderV2:
    def __init__(self, is_train, num_frames=18, use_ytvos=False, soenet_feats=False, soe_feat_type='soenet',
                 use_flow=False, debug=False, train_size=300, val_size=396, random_range=False, seq=None):
        self.is_train = is_train
        self.num_frames = num_frames
        self.use_ytvos = use_ytvos
        self.use_soenet_feats = soenet_feats
        self.soe_feat_type = soe_feat_type
        self.use_flow = use_flow
        self.dt = 1
        self.random_range = random_range
        self.debug = debug  # This is used to debug with original image, without flip/crop, to test the dataloader
        logger.debug('soenet_feats:{} soenet_feat_type:{}'.format(soenet_feats, soe_feat_type))
        print('init dataset-> is_train:{} soenet_feats:{} soenet_feat_type:{}'.format(is_train, soenet_feats,
                                                                                      soe_feat_type))

        if self.use_soenet_feats:
            assert soe_feat_type in ['soenet', 'msoenet']

        self.split = 'train' if self.is_train else 'val'
        if self.is_train:
            self._transforms = make_train_transform_hr(train_size=train_size)
        else:  # val
            self._transforms = make_validation_transforms(val_size=val_size)
        # ########################################################################
        if self.debug:
            self.transform_debug = make_validation_transforms(val_size=300)
        # ########################################################################
        self.host_name = socket.gethostname()
        logger.debug('self.host_name:%s' % self.host_name)
        self.paths_config = PathConfig()
        if self.host_name not in self.paths_config.davis16_train_seqs_file:
            self.host_name = 'vector'

        self.davis16_train_seqs_file = self.paths_config.davis16_train_seqs_file[self.host_name]
        self.davis16_val_seqs_file = self.paths_config.davis16_val_seqs_file[self.host_name]
        self.davis16_rgb_path = self.paths_config.davis16_rgb_path[self.host_name]
        self.davis16_gt_path = self.paths_config.davis16_gt_path[self.host_name]

        self.davis16_rgb_path_npy = self.paths_config.davis16_rgb_path_npy[self.host_name]
        self.davis16_gt_path_npy = self.paths_config.davis16_gt_path_npy[self.host_name]
        # self.davis16_soenet_feat_path = self.paths_config.davis16_soenet_feat_path[self.host_name]
        self.davis16_soenet_feat_thwc_path = self.paths_config.davis16_soenet_feat_thwc_path[self.host_name]
        self.davis16_msoenet_feat_thwc_path = self.paths_config.davis16_msoenet_feat_thwc_path[self.host_name]

        self.ytvos19_rgb_path = self.paths_config.ytvos19_rgb_path[self.host_name]
        self.ytvos19_gt_path = self.paths_config.ytvos19_gt_path[self.host_name]
        self.ytvos19_rgb_path_npy = self.paths_config.ytvos19_rgb_path_npy[self.host_name]
        self.ytvos19_gt_path_npy = self.paths_config.ytvos19_gt_path_npy[self.host_name]
        self.ytvos19_soenet_feat_path = self.paths_config.ytvos19_soenet_feat_path[self.host_name]
        self.ytvos19_soenet_feat_thwc_path = self.paths_config.ytvos19_soenet_feat_thwc_path[self.host_name]
        self.ytvos19_msoenet_feat_thwc_path = self.paths_config.ytvos19_msoenet_feat_thwc_path[self.host_name]

        self.frames_info = {
            'davis': {},
            'ytvos': {}
        }
        self.img_ids = []
        if self.is_train:
            logger.debug('loading davis 16 train seqs...')
            with open(self.davis16_train_seqs_file, 'r') as f:
                video_names = f.readlines()
                video_names = [name.strip() for name in video_names]
                # video_names = ['train', 'bus', 'car-turn', 'car-shadow']
                logger.debug('davis-train num of videos: {}'.format(len(video_names)))
                for video_name in video_names:
                    frames = sorted(glob.glob(os.path.join(self.davis16_gt_path, video_name, '*.png')))
                    self.frames_info['davis'][video_name] = [frame_path.split('/')[-1][:-4] for frame_path in frames]
                    self.img_ids.extend([('davis', video_name, frame_index) for frame_index in range(len(frames))])
            if self.use_ytvos:
                logger.debug('loading ytvos train seqs...')
                video_names = glob.glob(self.ytvos19_gt_path + "/*/")
                video_names = [video_name.split('/')[-2] for video_name in video_names]
                # video_names = ['fcf637e3ab', '0b5b5e8e5a', ]
                # import ipdb; ipdb.set_trace()
                logger.debug('ytvos-train num of videos: {}'.format(len(video_names)))
                for video_name in video_names:
                    frames = sorted(glob.glob(os.path.join(self.ytvos19_gt_path, video_name, '*.png')))
                    self.frames_info['ytvos'][video_name] = [frame_path.split('/')[-1][:-4] for frame_path in frames]
                    self.img_ids.extend(
                        [('ytvos', video_name, frame_index) for frame_index in range(0, len(frames), 9)])
                    # have a look at this later ...
        else:
            logger.debug('loading davis 16 val seqs...')
            if seq is not None:
                video_names = [seq]
                print('seq:%s' % seq)
            else:
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
        vid_len = len(self.frames_info[dataset][video_name])
        center_frame_name = self.frames_info[dataset][video_name][frame_index]
        dt = self.dt
        if self.random_range:
            dt = np.random.choice([1, 2, 3, 4], p=[0.4, 0.3, 0.2, 0.1])
        deltas = list(range(-math.ceil((self.num_frames - 1) / 2), math.ceil(self.num_frames / 2), 1))
        frame_indices = [(frame_index+x*dt + vid_len) % vid_len for x in deltas]
        assert len(frame_indices) == self.num_frames
        frame_ids = []
        img = []
        img_paths = []
        masks = []
        mask_paths = []
        for frame_id in frame_indices:
            frame_name = self.frames_info[dataset][video_name][frame_id]
            frame_ids.append(frame_name)
            if dataset == 'ytvos':
                img_path = os.path.join(self.ytvos19_rgb_path, video_name, frame_name + '.jpg')
                gt_path = os.path.join(self.ytvos19_gt_path, video_name, frame_name + '.png')
            else:
                img_path = os.path.join(self.davis16_rgb_path, video_name, frame_name + '.jpg')
                gt_path = os.path.join(self.davis16_gt_path, video_name, frame_name + '.png')
            img_paths.append(img_path)
            img_i = Image.open(img_path).convert('RGB')
            img.append(img_i)
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            gt[gt > 0] = 255
            masks.append(torch.Tensor(np.expand_dims(np.asarray(gt.copy()), axis=0)))
            if not self.is_train:
                mask_paths.append(gt_path)
        masks = torch.cat(masks, dim=0)
        target = {'dataset': dataset, 'video_name': video_name, 'center_frame': center_frame_name,
                  'frame_ids': frame_ids, 'masks': masks, 'vid_len': vid_len, 'mask_paths': mask_paths,
                  'img_paths': img_paths}
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return torch.cat(img, dim=0), target


def make_train_transform_hr(train_size=None):
    print('using make_train_transform_hr> train_size:%d' % train_size)
    logger.debug('using make_train_transform_hr> train_size:%d' % train_size)
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
            # To suit the GPU memory the scale might be different
            T.RandomResize([train_size], max_size=int(1.8 * train_size)),  # for r50
            # T.RandomResize([320], max_size=416),  # for r50
            # T.RandomResize([280], max_size=504),#for r101
        ]),
        normalize,
    ])


def make_train_transform(train_size=300):
    print('using make_train_transform> train_size:%d' % train_size)
    logger.debug('using make_train_transform> train_size:%d' % train_size)
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768]
    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomResize(scales, max_size=800),
        T.PhotometricDistort(),
        T.Compose([
            T.RandomResize([400, 500, 600]),
            T.RandomSizeCrop(384, 600),
            # To suit the GPU memory the scale might be different
            T.RandomResize([train_size], max_size=int(train_size * 1.8)),  # for r50
            # T.RandomResize([280], max_size=504),#for r101
        ]),
        normalize,
    ])


def make_validation_transforms(val_size=360):
    print('using make_validation_transforms> val_size:%d' % val_size)
    logger.debug('using make_validation_transforms> val_size:%d' % val_size)
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return T.Compose([
        T.RandomResize([val_size], max_size=int(val_size * 2)),
        normalize,
    ])
