import numpy as np
import sys
import cv2
import os
# import Image
from PIL import Image


def PIL2array(img):
    return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 4)


def create_overlay(img, mask, colors):
    im = Image.fromarray(np.uint8(img))
    im = im.convert('RGBA')

    mask_color = np.zeros((mask.shape[0], mask.shape[1], 3))
    if len(colors) == 3:
        mask_color[mask == colors[1], 0] = 255
        mask_color[mask == colors[1], 1] = 255
        mask_color[mask == colors[2], 0] = 255
    else:
        mask_color[mask == colors[1], 2] = 255

    overlay = Image.fromarray(np.uint8(mask_color))
    overlay = overlay.convert('RGBA')

    im = Image.blend(im, overlay, 0.7)
    blended_arr = PIL2array(im)[:, :, :3]
    img2 = img.copy()
    img2[mask == colors[1], :] = blended_arr[mask == colors[1], :]
    return img2


def main():
    if len(sys.argv) < 2:
        raise ValueError('Dataset name???')
    dataset = sys.argv[1]
    assert dataset in ['davis', 'moca']

    if dataset == 'davis':
        main_dir = '/local/riemann/home/rezaul/datasets/DAVIS-data/DAVIS_2016/JPEGImages/480p'
        annots_path = '/local/riemann/home/rezaul/datasets/DAVIS-data/DAVIS_2016/Annotations/480p'
        rtnet_davis = '/local/riemann/home/rezaul/projects/medvt2-main/results4figures_swin/davis/DAVIS_RX50_crf/DAVIS'
        baseline_mask_dir = '/local/riemann/home/rezaul/projects/transformer-vos-main/results/viz_ablation_ss_vs_ms/enc_ss_dec_ss_ab03/davis_473/bin_mask'  # ss-ss
        swinb_mask_dir = '/local/riemann/home/rezaul/projects/medvt2-main/results4figures_swin/davis/swinb_davis_crf'  # swinb ss-ms
        # mask_dir = '/local/riemann/home/rezaul/projects/transformer-vos-main/results/supplement_video/davis/davis_binmask'
        mask_dir = rtnet_davis
        out_dir = '/local/riemann/home/rezaul/projects/medvt2-main/results4figures_swin/davis/davis_rtnet_overlay_red'
        # out_dir = mask_dir + '_overlay_red'
        # seq_names = sorted(os.listdir(main_dir))
        DAVIS_seqs = ["blackswan", "bmx-trees", "breakdance", "camel", "car-roundabout", "car-shadow", "cows",
                      "dance-twirl", "dog", "drift-chicane", "drift-straight", "goat", "horsejump-high", "kite-surf",
                      "libby", "motocross-jump", "paragliding-launch", "parkour", "scooter-black", "soapbox"]
        seq_names = DAVIS_seqs
        seq_names = ['breakdance', 'car-roundabout', 'dance-twirl']
        # seq_names = ['breakdance', 'drift-chicane', 'scooter-black', 'horsejump-high', 'dance-twirl', 'car-roundabout']
    elif dataset == 'moca':
        main_dir = '/local/riemann/home/msiam/MoCA_filtered2/JPEGImages'
        annots_path = '/local/riemann/home/msiam/MoCA_filtered2/Annotations'
        moca_rtnet_masks = '/local/riemann/home/rezaul/projects/medvt2-main/results4figures_swin/moca/results_rtnet_moca'
        moca_resnet_ssss_mask = '/local/riemann/home/rezaul/projects/transformer-vos-main/results/viz_ablation_ss_vs_ms/enc_ss_dec_ss_ab03/moca_473'
        # mask_dir = '/local/riemann/home/rezaul/projects/transformer-vos-main/results/viz_ablation_ss_vs_ms/enc_ss_dec_ss_ab03/moca_473'  # ss-ss
        swinb_mask_dir = '/local/riemann/home/rezaul/projects/medvt2-main/results4figures_swin/moca/moca_swinb_ssms_473_msc'  # ms-ms
        # mask_dir = '/local/riemann/home/rezaul/projects/transformer-vos-main/results/supplement_video/moca/binmask'
        rtnet_mask = '/local/riemann/home/rezaul/projects/medvt2-main/results4figures_swin/moca/results_rtnet_final_moca'
        mask_dir = moca_rtnet_masks
        # out_dir = mask_dir + '_overlay_red'
        # mask_dir = annots_path
        out_dir = '/local/riemann/home/rezaul/projects/medvt2-main/results4figures_swin/moca/moca_rtnet_overlay_red'
        # seq_names = sorted(os.listdir(main_dir))
        # seq_names = ['arabian_horn_viper', 'hedgehog_1', 'flounder_6', 'egyptian_nightjar']
        # import ipdb; ipdb.set_trace()
        seq_names_file = open(
            '/local/riemann/home/rezaul/projects/medvt2-main/results4figures_swin/moca/moca_seq_names.txt', 'r')
        seq_names = seq_names_file.readlines()
        seq_names = [seq.strip() for seq in seq_names]
        seq_names = ['flounder_6', 'flatfish_1', 'black_cat_1']
    else:
        exit()

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for d in seq_names:
        if not os.path.exists(os.path.join(mask_dir, d)):
            print('mask dir not exists: %s' % str(os.path.join(mask_dir, d)))
            continue
        print('current sequence:%s' % d)
        if not os.path.exists(os.path.join(out_dir, d)):
            os.makedirs(os.path.join(out_dir, d))

        files = sorted(os.listdir(os.path.join(main_dir, d)))
        for i in range(len(files)):
            if files[i].split('.')[1] != 'jpg':
                continue
            print(files[i])
            img = cv2.imread(os.path.join(main_dir, d, files[i]))
            mask_file = os.path.join(mask_dir, d, files[i].split('.')[0] + '.png')
            if not os.path.exists(mask_file):
                mask_file = os.path.join(mask_dir, d, '%s_%s.png' % (d, files[i].split('.')[0]))  # for rtnet files
            mask2 = cv2.imread(mask_file, 0)
            if mask2 is None:
                print('mask not found for %s' % d)
                continue
            if mask2.shape[0] != img.shape[0]:
                img = cv2.resize(img, mask2.shape[::-1])
            # mask2[mask2 > 0.5] = 255
            # mask2[mask2 < 0.5] = 0
            overlay = create_overlay(img, mask2, [0, 255])
            cv2.imwrite(os.path.join(out_dir, d, files[i]), overlay)


main()
