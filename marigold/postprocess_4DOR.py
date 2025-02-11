import os
import glob
import numpy as np
from tqdm import tqdm
import scipy
import cv2
from PIL import Image
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--take', type=int, default=1)
    parser.add_argument('--resize', type=int, default=1)
    return parser.parse_args()


def rescale_lstsq_scipy(im, depth):
    nonzero_idxs = np.nonzero(depth)
    nonzero_depth = depth[nonzero_idxs].flatten()
    nonzero_im = im[nonzero_idxs].flatten()
    nonzero_im = nonzero_im[:, np.newaxis]**[0, 2]
    p, _, _, _ = scipy.linalg.lstsq(nonzero_im, nonzero_depth)
    im = p[0] + p[1]*im**2
    im = np.clip(im, a_min=0.0, a_max=None)
    return im


if __name__ == '__main__':
    args = parse_arguments()
    print(f"!!! START WITH POSTPROCESSING TAKE {args.take} !!!")

    data_dir = f'/home/geratsbga1/4D-OR/export_holistic_take{args.take}_processed'
    path_input = os.path.join(data_dir, 'colorimage_' + str(args.resize))
    path_output = os.path.join(data_dir, 'depthimage_marigold_' + str(args.resize))
    depthimage_path = os.path.join(data_dir, 'depthimage_' + str(args.resize))

    if not os.path.exists(depthimage_path):
        os.makedirs(depthimage_path)

    files = glob.iglob(os.path.join(path_output, 'depth_npy', '*.npy'))
    for f in tqdm(files):
        im = np.load(f)
        im_name = os.path.split(f)[-1].replace('_pred.npy', '').replace('color', 'depth')
        orig_im = cv2.imread(os.path.join(data_dir, 'depthimage', im_name+'.tiff'), -1)
        orig_im = cv2.resize(orig_im, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
        im = rescale_lstsq_scipy(im, orig_im)

        im = im.astype(np.uint16)
        im = Image.fromarray(im)
        im.save(os.path.join(path_output, im_name+'.tiff'))

        orig_im = orig_im.astype(np.uint16)
        orig_im = Image.fromarray(orig_im)
        orig_im.save(os.path.join(depthimage_path, im_name+'.tiff'))
        