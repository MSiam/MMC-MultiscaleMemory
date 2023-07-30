import numpy as np
import sys
import cv2
import os
from PIL import Image


def read_imgs(imgs_dir):
    imgs = []
    # imgs2 = []
    #    req_sqs = ['drinking_2']
    for root, dirs, files in os.walk(imgs_dir):
        files = sorted(files)
        for f in range(len(files)):
            #            cont = False
            #            for r in req_sqs:
            #                if r in files[f]:
            #                    cont = True
            #            if cont:
            img = cv2.imread(imgs_dir + files[f])
            # img = cv2.resize(img, (img.shape[1] / 2, img.shape[0] / 2))
            imgs.append(img)
    return imgs


def gen_video_back(imgs, vidfile):
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    video = cv2.VideoWriter(vidfile, fourcc, 10, (imgs[0].shape[1], imgs[0].shape[0]))
    for i in range(len(imgs)):
        video.write(imgs[i])
    cv2.destroyAllWindows()
    video.release()


def gen_video(imgs, vidfile):
    width = imgs[0].shape[1]
    height = imgs[0].shape[0]
    # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    # video = cv2.VideoWriter('%s.avi'%vidfile, fourcc, 15, (imgs[0].shape[1], imgs[0].shape[0]))
    # video = cv2.VideoWriter('%s.avi'%vidfile, 0, 1,  (imgs[0].shape[1], imgs[0].shape[0]))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('%s.avi'%vidfile, fourcc, 10, (width, height))

    for i in range(len(imgs)):
        video.write(imgs[i])
    cv2.destroyAllWindows()
    video.release()



if __name__ == "__main__":
    """
    argv[1]: input images path
    argv[2] : vidfile
    """
    imgs = read_imgs(sys.argv[1])
    gen_video(imgs, sys.argv[2])

