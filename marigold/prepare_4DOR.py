import os
import glob
import shutil
from tqdm import tqdm
import cv2
import json
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--take', type=int, default=1)
    parser.add_argument('--resize', type=int, default=1)
    parser.add_argument('--start_scene', type=int, default=None)
    parser.add_argument('--end_scene', type=int, default=None)
    parser.add_argument('--stride', type=int, default=1)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    print(f"!!! START WITH PREPROCESSING TAKE {args.take} !!!")
    
    data_dir = f'/home/geratsbga1/4D-OR/export_holistic_take{args.take}_processed'
    path_input = os.path.join(data_dir, 'colorimage_' + str(args.resize))
    path_output = os.path.join(data_dir, 'depthimage_marigold_' + str(args.resize))

    # Create input and output dir if not exist
    if not os.path.exists(path_input):
        os.makedirs(path_input)
    if not os.path.exists(path_output): 
        os.makedirs(path_output)

    # Remove all files in input dir
    files = glob.iglob(os.path.join(path_input, '*'))
    for f in files:
        os.remove(f)

    # Remove all files in output dir
    files = glob.iglob(os.path.join(path_output, '*'))
    for f in files:
        if os.path.isfile(f):
            os.remove(f)
        else:
            shutil.rmtree(f)

    if args.start_scene is None and args.end_scene is None:
        # Copy all image files of take to input dir
        files = glob.iglob(os.path.join(data_dir, 'colorimage', '*.jpg'))
    else:
        files = []
        json_path = os.path.join(data_dir, 'timestamp_to_pcd_and_frames_list.json')
        with open(json_path, 'r') as file:
            timestamps = json.load(file)
        for ts in timestamps[args.start_scene:args.end_scene:args.stride]:
            for c in range(1, 7):
                scene_id = ts[1]['color_'+str(c)]
                im_name = f'camera0{c}_colorimage-{scene_id}.jpg'
                files.append(os.path.join(data_dir, 'colorimage', im_name))
        
    if args.resize == 1:
        for f in tqdm(files):
            shutil.copy(f, path_input)
    elif args.resize > 1:
        for f in tqdm(files):
            im = cv2.imread(f)
            H, W, _ = im.shape
            H_new = H // args.resize
            W_new = W // args.resize
            im = cv2.resize(im, (W_new, H_new), interpolation=cv2.INTER_AREA)
            img_name = os.path.split(f)[-1]
            cv2.imwrite(os.path.join(path_input, img_name), im)
    else:
        print("ERROR: cannot make images larger")