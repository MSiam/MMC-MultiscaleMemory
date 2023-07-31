# MMC: Multiscale Memory

This repository is the official implementation of [Multiscale Memory Comparator Transformer for Few-Shot Video Segmentation](https://arxiv.org/abs/2307.07812).


## Environment Setup
The used python and libraries:
* Python 3.7
* Torch 1.9.0
* Torchvision 0.10.0

## Dataset

We have used three publicly available dataset for evaluation. Download the datasets following the corresponding paper/project. For MoCA we used the 88 videos, preprocessing and the evaluation scheme from the [MotionGrouping paper](https://github.com/charigyang/motiongrouping)

- [DAVIS16](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Perazzi_A_Benchmark_Dataset_CVPR_2016_paper.pdf)
- [MoCA](https://openaccess.thecvf.com/content/ACCV2020/papers/Lamdouar_Betrayed_by_Motion_Camouflaged_Object_Discovery_via_Motion_Segmentation_ACCV_2020_paper.pdf)
- [Youtube-Objects](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6248065)

## Pre-trained Models

Download pretrained models [here](https://www.dropbox.com/scl/fi/91059ds8o8eoizyzgc7hy/weights_ablation_avos_msmemory.zip?rlkey=5xge28u8ef8ey2z4nalvf725e&dl=0). It includes the weights for the baseline multiscale query and our multiscale memory models with VideoSwin and R101.

Names of models inside the checkpoint for VideoSwin:
* Baseline - Multiscale Query: ms_qry_5frames
* Multiscale Memory Bidirectional: ms_qry_memory_5frames
* Multiscale Memory Stacked: swin-msqry-mem-nobidir

## Evaluation
Only use "_adaptive" if GPU memory does not fit. Otherwise remove it for the decoder type to become only "multiscale_query_memory_nobidir". Results in the arxiv paper are reported using val_size 440 to fit in GPU memory for both baseline and our method. However, after our addition of the adaptive technique we can evaluate with 473 similar to SOA methods.

* Youtube Objects
```
python inference.py --model_path CKPT_PATH --dataset ytbo --val_size 473 --output_dir OUT_DIR --decoder_type multiscale_query_memory_nobidir_adaptive
```

* DAVIS16
```
python inference.py --model_path CKPT_PATH --dataset davis --val_size 473 --output_dir OUT_DIR --decoder_type multiscale_query_memory_nobidir_adaptive --aug
```

* MoCA 
```
python inference.py --model_path CKPT_PATH --dataset moca --val_size 473 --output_dir OUT_DIR --decoder_type multiscale_query_memory_nobidir_adaptive
```

## Results

Our model achieves the following performance w.r.t multiscale query transformer decoder and other SOA methods in terms of mIoU:

| Inference method       | DAVIS'16     | MoCA              | YouTube-Objects   |
|------------------------|--------------| ------------------| ------------------|
| AGS                    | 79.7         | -                 | 69.7              | 
| COSNet                 | 80.5         | 50.7              | 70.5              |
| AGNN                   | 80.7         | -                 | 70.8              |
| MATNet                 | 82.4         | 64.2              | 69.0              |
| RTNet                  | 85.6         | 60.7              | 71.0              |
| Multiscale Query       | 83.7         | 77.4              | 76.8              |
|------------------------|--------------| ------------------| ------------------|
| Multiscale Memory(Ours)| 86.1         | 80.3              | 78.2              |

### Acknowledgement
We would like to thank the [MED-VT](https://github.com/rkyuca/medvt) open-source code that was used to build the baseline for our project.

