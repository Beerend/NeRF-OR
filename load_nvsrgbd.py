import numpy as np
import os
import cv2 as cv
import torch
import scipy
from read_colmap import read_images_binary, read_points3d_binary
from run_nerfor_helpers import get_rays_np


def pad_poses(p):
  """ Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]. """
  bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
  return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p):
  """ Remove the homogeneous bottom row from [..., 4, 4] pose matrices. """
  return p[..., :3, :4]


def normalize(x):
  """ Normalization helper function. """
  return x / np.linalg.norm(x)


def viewmatrix(lookdir, up, position, subtract_position=False):
  """ Construct lookat view matrix. """
  vec2 = normalize((lookdir - position) if subtract_position else lookdir)
  vec0 = normalize(np.cross(up, vec2))
  vec1 = normalize(np.cross(vec2, vec0))
  m = np.stack([vec0, vec1, vec2, position], axis=1)
  return m


def poses_avg(poses):
  """ New pose using average position, z-axis, and up vector of input poses. """
  position = poses[:, :3, 3].mean(0)
  z_axis = poses[:, :3, 2].mean(0)
  up = poses[:, :3, 1].mean(0)
  cam2world = viewmatrix(z_axis, up, position)
  return cam2world


def recenter_poses(poses):
  """ Recenter poses around the origin. """
  cam2world = poses_avg(poses)
  poses = np.linalg.inv(pad_poses(cam2world)) @ pad_poses(poses)
  return unpad_poses(poses)


def read_cameras_information(readdir, bd_factor=0.75):
    """ Read camera information from COLMAP-generated 'poses_bounds.npy' file. """

    # Read file and shape into correct format
    cam_info = {}
    poses_bounds = np.load(os.path.join(readdir, 'poses_bounds.npy'))
    poses = poses_bounds[:, :-2].reshape([-1, 3, 5])
    bounds = poses_bounds[:, -2:]
    hwf = poses[:, :, -1]
    
    # Correct rotation matrix ordering (and drop 5th column of poses)
    fix_rotation = np.array([
        [0, -1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    poses = poses[:, :3, :4] @ fix_rotation

    # Rescale according to a default bd factor
    scale = 1.0 / (bounds.min() * bd_factor)
    poses[:, :3, 3] *= scale
    bounds *= scale

    # Recenter poses
    poses = recenter_poses(poses)

    # For each camera, put information in dictionary
    for c in range(poses.shape[0]):
        cam_info[c] = {
            'H': hwf[c, 0],
            'W': hwf[c, 1],
            'pp': [hwf[c, 1] / 2.0, hwf[c, 0] / 2.0],
            'focal': [hwf[c, 2], hwf[c, 2]],
            'color2world': poses[c],
        }
    
    return cam_info, bounds


def create_camera_poses(cams_info, image_scale):
    """ Put extrinsic and intrinsic camera parameters in array format. """
    cam_poses = {}
    for c in cams_info.keys():
        camera_pose = np.zeros((3, 6), dtype=np.float32)
        camera_pose[:, :4] = cams_info[c]['color2world'][:3]
        H, W = cams_info[c]['H'], cams_info[c]['W']
        camera_pose[0, 4] = round(H / image_scale)
        camera_pose[0, 5] = round(W / image_scale)
        camera_pose[1, 4] = cams_info[c]['focal'][0] / image_scale
        camera_pose[1, 5] = cams_info[c]['focal'][1] / image_scale
        camera_pose[2, 4] = cams_info[c]['pp'][0] / image_scale
        camera_pose[2, 5] = cams_info[c]['pp'][1] / image_scale
        cam_poses[c] = camera_pose
    return cam_poses


def get_colmap_poses(images):
    """ Get camera poses from COLMAP binary images file. """
    poses = []
    for i in images:
        R = images[i].qvec2rotmat()
        t = images[i].tvec.reshape([3, 1])
        bottom = np.array([0, 0, 0, 1.0]).reshape([1, 4])
        w2c = np.concatenate([np.concatenate([R, t], axis=1), bottom], axis=0)
        c2w = np.linalg.inv(w2c)
        poses.append(c2w)
    return np.array(poses)


def read_colmap_depths(readdir, bd_factor=0.75, factor=1.0, max_error=0.1, top_k=None):
    """ Read depth values from points found during COLMAP:
     - factor: downscaling factor for images (1 is original size)
     - max_error: maximum reprojection error for points to be included
     - top_k: maximum number of points to include per image
     """
    images = read_images_binary(os.path.join(readdir, 'sparse/0/images.bin'))
    points = read_points3d_binary(os.path.join(readdir, 'sparse/0/points3D.bin'))
    poses = get_colmap_poses(images)

    poses_bounds = np.load(os.path.join(readdir, 'poses_bounds.npy'))
    H = int(poses_bounds[0, 4] / factor)
    W = int(poses_bounds[0, 9] / factor)
    bounds = poses_bounds[:, -2:].transpose([1, 0])
    bounds = np.moveaxis(bounds, -1, 0).astype(np.float32)
    scale = 1.0 / (bounds.min() * bd_factor)

    colmap_depths = []
    for pose, img_id in zip(poses, sorted(list(images))):
        depth_img = np.zeros((H, W), dtype=np.float32)

        coords_list = []
        depth_list = []
        error_list = []
        for i in range(len(images[img_id].xys)):
            point2D = images[img_id].xys[i]
            point3D_id = images[img_id].point3D_ids[i]
            if point3D_id == -1:
                continue

            point3D = points[point3D_id].xyz
            depth = (pose[:3, 2].T @ (point3D - pose[:3, 3])) * scale
            if depth < bounds[img_id - 1, 0] * scale or depth > bounds[img_id - 1, 1] * scale:
                continue
                
            error = points[point3D_id].error
            if error > max_error:
                continue

            coords_list.append(np.rint(np.array(point2D / factor)).astype(int))
            depth_list.append(depth)
            error_list.append(error)

        coords_list = np.array(coords_list)
        depth_list = np.array(depth_list)
        error_list = np.array(error_list)

        sorted_idxs = np.argsort(error_list)
        coords_list = coords_list[sorted_idxs]
        depth_list = depth_list[sorted_idxs]
        error_list = error_list[sorted_idxs]

        select = len(error_list)
        if top_k is not None:
            select = min(select, top_k)

        for coords, depth in zip(reversed(coords_list[:select]), reversed(depth_list[:select])):
            depth_img[coords[1], coords[0]] = depth

        colmap_depths.append(depth_img)
        mean_error = np.nan if select  == 0 else np.mean(error_list[:select])
        print(f'>> Constructed COLMAP image {img_id} with {select} depth values ({max_error=}, {top_k=}) (mean_error={mean_error:.3f})')
    
    return np.array(colmap_depths)


def read_color_image(readdir, img_name, width=None, height=None, img_scale=None):
    """ Read color images, convert to RGB and normalize. """
    if img_scale == 4.0:  # Shortcut for images with 4x downsampling
        img = cv.imread(os.path.join(readdir, 'images_4', img_name), cv.IMREAD_COLOR)
    elif img_scale == 2.0:  # Shortcut for images with 2x downsampling
        img = cv.imread(os.path.join(readdir, 'images_2', img_name), cv.IMREAD_COLOR)
    else:
        img = cv.imread(os.path.join(readdir, 'images', img_name), cv.IMREAD_COLOR)
        if width and height:
            img = cv.resize(img, (width, height), interpolation=cv.INTER_NEAREST)
    if img is None:
        print(f'Error: could not read {img_name}')
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = np.float32(img) / 255.0
    return img


def read_depth_image(readdir, img_folder, img_name, width=None, height=None, invert=False, convert_to_meters=True):
    """ Read depth images and convert to meters. """
    img = cv.imread(os.path.join(readdir, img_folder, img_name), -1)  # -1 is to read the image as int16
    if width and height:
            img = cv.resize(img, (width, height), interpolation=cv.INTER_NEAREST)
    if img is None:
        print(f'Error: could not read {img_name}')
    img = np.float32(img)
    img /= 1000.0  # millimeters to meters
    return img


def read_depth_npy(readdir, img_name, img_scale=None):
    """ Read estimated depth images from Marigold. """
    suffix = '_' + str(int(img_scale)) if img_scale > 1.0 else ''
    img = np.load(os.path.join(readdir, 'depth_maps_marigold' + suffix, 'depth_npy', img_name.replace('.png', '_pred.npy')))
    return img


def align_depths_with_colmap(depths, cm_depths):
    """ Using least squares, align scale of sensor depths with COLMAP depths. """
    depths_aligned = {'train': [], 'test': []}
    values_x = []
    values_y = []

    for depth, cm_depth in zip(depths['train'], cm_depths):
        mask = depth * cm_depth
        nonzero_idxs = np.nonzero(mask)
        values_x.append(depth[nonzero_idxs].flatten())
        values_y.append(cm_depth[nonzero_idxs].flatten())

    values_x = np.concatenate(values_x)
    values_y = np.concatenate(values_y)
    values_x = values_x[:, np.newaxis]
    gain, _, _, _ = np.linalg.lstsq(values_x, values_y, rcond=None)
    gain = gain[0]
    mean_error = np.mean(np.abs(values_x * gain - values_y))
    print(f'>> Aligned depths with COLMAP depths ({gain=:.3f}) ({mean_error=:.3f})')

    for depth in depths['train']:
        depth_aligned = depth * gain
        depth_aligned = depth_aligned * np.where(depth > 0.0, 1.0, 0.0)
        depths_aligned['train'].append(depth_aligned)

    for depth in depths['test']:
        depth_aligned = depth * gain
        depth_aligned = depth_aligned * np.where(depth > 0.0, 1.0, 0.0)
        depths_aligned['test'].append(depth_aligned)

    return {'train': np.array(depths_aligned['train']), 'test': np.array(depths_aligned['test'])}


def align_estimated_depths(est_depths, depths):
    """ Using least squares, align scale of estimated depths with sensor depths. """
    depths_aligned = []
    for est_depth, depth in zip(est_depths, depths):
        nonzero_idxs = np.nonzero(depth)
        nonzero_depth = depth[nonzero_idxs].flatten()
        nonzero_est = est_depth[nonzero_idxs].flatten()
        nonzero_est = nonzero_est[:, np.newaxis]**[0, 2]
        offset, gain = scipy.linalg.lstsq(nonzero_est, nonzero_depth)[0]
        depth_aligned = gain * est_depth**2 + offset
        depths_aligned.append(depth_aligned)
    return np.array(depths_aligned)


def render_path_spiral(poses, zrate=0.5, n_rots=2, n_frames=120):
    """ Virtual camera path that is a spiral motion around a set of camera poses. """
    intrinsics = poses[0, :, 4:]
    focal = np.mean(poses[:, 1, 4:])
    positions = poses[:, :3, 3]
    radii = np.percentile(np.abs(positions), 90, 0)
    radii = np.concatenate([radii, [1.0]])

    render_poses = []
    cam2world = poses_avg(poses[:, :, :4])
    up = poses[:, :, 1].mean(0)
    for theta in np.linspace(0.0, 2.0 * np.pi * n_rots, n_frames, endpoint=False):
        t = radii * [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0]
        position = cam2world @ t
        lookat = cam2world @ [0, 0, -focal, 1.0]
        z_axis = position - lookat
        render_poses.append(viewmatrix(z_axis, up, position))
    render_poses = np.stack(render_poses, axis=0)
    render_poses = np.concatenate([render_poses, np.repeat(intrinsics[np.newaxis, :, :],
                                        repeats=render_poses.shape[0], axis=0)], axis=2)
    return render_poses


def convert_to_array(item, dtype=None):
    """ Convert dictionary or list to numpy array. """
    if isinstance(item, dict):
        item = {key: np.array(item[key], dtype=dtype) for key in item.keys()}
    elif isinstance(item, list):
        item = np.array(item, dtype=dtype)
    return item


def print_dataset_info(images, poses, depths, bounds, bounding_box, est_depths):
    """ Print information about the dataset. """
    print("> Loaded NVS-RGBD dataset:")
    print(f">> Images: {images['train'].shape} (train), {images['test'].shape} (test)")
    print(f">> Poses: {poses['train'].shape} (train), {poses['test'].shape} (test), {poses['render'].shape} (render)")
    print(f">> Depths: {depths['train'].shape} (train), {depths['test'].shape} (test)")
    print(f">> Depth bounds: {bounds[0]:.3f} (min), {bounds[1]:.3f} (max)")
    print(f">> Scene bounding box: {bounding_box[0]} (min), {bounding_box[1]} (max)")
    if est_depths is not None:
        print(f">> Estimated depth images: {est_depths.shape}")


def get_bbox_3d(poses, bounds):
    """ Calculate the boundaries of the scene in 3D for hash-grid encoding. """
    min_bound = [np.inf, np.inf, np.inf]
    max_bound = [-np.inf, -np.inf, -np.inf]

    for pose in poses:
        H, W = pose[0, 4:].astype(int)
        cam_intrinsics = pose[:, 4:]
        cam_pose = pose[:, :4]
        rays_o, rays_d = get_rays_np(cam_intrinsics, cam_pose)
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)

        for i in [0, W-1, H*W-W, H*W-1]:
            min_point = rays_o[i] + np.min(bounds) * rays_d[i]
            max_point = rays_o[i] + np.max(bounds) * rays_d[i]

            for i in range(3):
                if (min_bound[i] > min_point[i]):
                    min_bound[i] = min_point[i]
                if (max_bound[i] < min_point[i]):
                    max_bound[i] = min_point[i]

            for i in range(3):
                if (min_bound[i] > max_point[i]):
                    min_bound[i] = max_point[i]
                if (max_bound[i] < max_point[i]):
                    max_bound[i] = max_point[i]

    bounding_box = (
        torch.tensor(min_bound, dtype=torch.float32) - torch.tensor([0.1, 0.1, 0.1]),
        torch.tensor(max_bound, dtype=torch.float32) + torch.tensor([0.1, 0.1, 0.1])
    )
    return bounding_box


def load_NVSRGBD_data(datadir, recording_device, scene, img_scale=1.0, depth_gradients=False, render_steps=None):
    """ Create NVS-RGBD dataset:
    - datadir: directory where the data is stored
    - recording_device: name of the recording device (only 'kinect' is supported)
    - scene: ID of the scene to include
    - img_scale: downscaling factor for images (1 for original size)
    - depth_gradients: include estimated depth gradients for training
    - render_steps: number of frames to include in render video
    """
    
    # Read camera information
    bd_factor = 0.75
    readdir = os.path.join(datadir, recording_device, 'scene0' + str(scene))
    cams_info, colmap_bounds = read_cameras_information(readdir, bd_factor=bd_factor)
    cam_poses = create_camera_poses(cams_info, img_scale)
    colmap_depths = read_colmap_depths(readdir, bd_factor=bd_factor, factor=img_scale, max_error=0.1)

    # Initiate lists to store data temporarily
    images = {'train': [], 'test': []}
    poses = {'train': [], 'test': [], 'render': []}
    depths = {'train': [], 'test': []}
    est_depths = [] if depth_gradients else None

    # Create list of all images
    suffix = '_' + str(int(img_scale)) if img_scale > 1 else ''
    img_names = sorted([f for f in os.listdir(os.path.join(readdir, 'images' + suffix)) if f.endswith('.png')])

    # Per camera and scene, get the data
    for c, img_name in enumerate(img_names):

        # Get camera height and width
        camera_pose = cam_poses[c]
        H, W = camera_pose[0, 4:].astype(int)

        # Read colorimage
        color_img = read_color_image(readdir, img_name, width=W, height=H, img_scale=img_scale)

        # Read depth image
        depth_img = read_depth_image(readdir, 'depth_maps', img_name, width=W, height=H)
            
        # Read estimated depth image if necessary
        if depth_gradients and '_train' in img_name:
            est_depth_img = read_depth_npy(readdir, img_name, img_scale=img_scale)  # Marigold
            est_depths.append(est_depth_img)

        # Put the data in the right lists
        if '_train' in img_name:
            images['train'].append(color_img)
            poses['train'].append(camera_pose)
            depths['train'].append(depth_img)
        else:
            images['test'].append(color_img)
            poses['test'].append(camera_pose)
            depths['test'].append(depth_img)
            
    # Align sensor depth with COLMAP depth
    depths = align_depths_with_colmap(depths, colmap_depths)

    # Align estimated depth with sensor depth
    est_depths = align_estimated_depths(est_depths, depths['train']) if est_depths is not None else None

    # Create camera poses for render video
    poses['render'] = render_path_spiral(np.array(poses['train']), zrate=0.5, n_rots=2, n_frames=render_steps)

    # Convert lists to numpy arrays
    images = convert_to_array(images)
    poses = convert_to_array(poses)
    depths = convert_to_array(depths)
    est_depths = convert_to_array(est_depths) if est_depths is not None else None

    # Train with dense estimated depths instead of sensor depths
    # depths['train'] = est_depths

    # Align depth images and colmap bounds
    colmap_min, colmap_max = np.min(colmap_bounds), np.max(colmap_bounds)
    print(f'>> Colmap bounds: {colmap_min:.3f} (min), {colmap_max:.3f} (max)')

    # Set bounds
    max_depths = [colmap_max, np.max(depths['train'])]
    if est_depths is not None:
        max_depths.append(np.max(est_depths))
    bounds = np.array([0.0, np.max(max_depths)  * 1.1])

    # Set bounding box of scene
    bounding_box = get_bbox_3d(poses['train'], bounds)

    print_dataset_info(images, poses, depths, bounds, bounding_box, est_depths)
    return images, poses, depths, bounds, bounding_box, est_depths