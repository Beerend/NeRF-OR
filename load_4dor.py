import numpy as np
import os
import json
import cv2 as cv
import torch
from scipy.spatial.transform import Rotation


def read_cameras_information(readdir, cameras):
    """ Read camera information from JSON files. """
    cam_info = {}
    for c in cameras:
        cam = str(c)
    
        # Read JSON
        with open(os.path.join(readdir, 'camera0' + cam + '.json'), 'r') as file:
            data = json.load(file)
            file.close()

        # Depth camera to world transform
        r = data['value0']['camera_pose']['rotation']
        t = data['value0']['camera_pose']['translation']
        depth2world = np.zeros((4, 4), dtype=np.float32)
        depth2world[:3, :3] = Rotation.from_quat([r['x'], r['y'], r['z'], r['w']]).as_matrix()
        depth2world[:, 3] = [t['m00'], t['m10'], t['m20'], 1]

        # Color to depth transform
        r = data['value0']['color2depth_transform']['rotation']
        t = data['value0']['color2depth_transform']['translation']
        color2depth = np.zeros((4, 4), dtype=np.float32)
        color2depth[:3, :3] = Rotation.from_quat([r['x'], r['y'], r['z'], r['w']]).as_matrix()
        color2depth[:, 3] = [t['m00'], t['m10'], t['m20'], 1]
        
        # Color camera to world transform
        color2world = np.dot(depth2world, color2depth)

        # Intrinsics
        color_parameters = data['value0']['color_parameters']
        H = color_parameters['height']
        W = color_parameters['width']
        pp = [color_parameters['c_x'], color_parameters['c_y']]
        focal = [color_parameters['fov_x'], color_parameters['fov_y']]

        # Change coordinate system: from (right, up, backward) to (right, down, forward)
        correction = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        
        # Put in dictionary
        cam_info[c] = {
            'H': H,
            'W': W,
            'pp': pp,
            'focal': focal,
            'correction': correction,
            'color2world': color2world,
            'depth2world': depth2world
        }
        
    return cam_info


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


def read_color_image(readdir, cam, frame_num, width=None, height=None, img_scale=None):
    """ Read color images, convert to RGB and normalize. """
    img_name = 'camera0' + cam + '_colorimage-' + frame_num + '.jpg'
    if img_scale == 4.0:  # Shortcut for images with 4x downsampling
        img = cv.imread(os.path.join(readdir, 'colorimage_4', img_name), cv.IMREAD_COLOR)
    else:
        img = cv.imread(os.path.join(readdir, 'colorimage', img_name), cv.IMREAD_COLOR)
        if width and height:
            img = cv.resize(img, (width, height), interpolation=cv.INTER_NEAREST)
    if img is None:
        print(f'Error: could not read {img_name}')
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = np.float32(img) / 255.0
    return img


def read_depth_image(readdir, imagefolder, cam, frame_num, width=None, height=None, img_scale=None):
    """ Read depth images and convert to meters. """
    img_name = 'camera0' + cam + '_depthimage-' + frame_num + '.tiff'
    if img_scale == 4.0:  # Shortcut for images with 4x downsampling
        img = cv.imread(os.path.join(readdir, imagefolder + '_4', img_name), -1)
    else:
        img = cv.imread(os.path.join(readdir, imagefolder, img_name), -1)  # -1 is to read the image as int16
        if width and height:
            img = cv.resize(img, (width, height), interpolation=cv.INTER_NEAREST)
    if img is None:
        print(f'Error: could not read {img_name}')
    img = np.float32(img)
    img /= 1000.0  # millimeters to meters
    return img


def update_boundaries(bounds, values):
    """ Track minimum and maximum distance from camera to objects. """
    min_value = np.min(values)
    max_value = np.max(values)
    if min_value < bounds[0]:
        bounds[0] = min_value
    if max_value > bounds[1]:
        bounds[1] = max_value
    return bounds


def create_virtual_camera_path(cam_poses, start_cam_id, end_cam_id, render_steps=25):
    """ Interpolate virtual camera path between two poses. """
    virtual_path = np.zeros((render_steps + 2, 3, 6), dtype=np.float32)
    virtual_path[0] = cam_poses[start_cam_id]
    for step in range(1, render_steps + 1):
        fraction = step / float(render_steps + 1)
        virtual_path[step] = (1.0 - fraction) * cam_poses[start_cam_id] + fraction * cam_poses[end_cam_id]
    virtual_path[-1] = cam_poses[end_cam_id]
    return virtual_path


def create_render_video_poses(cam_poses, render_steps=25):
    """ Virtual camera path between all six cameras in the dataset. """
    render_poses = np.concatenate([
        create_virtual_camera_path(cam_poses, 5, 1, render_steps),
        create_virtual_camera_path(cam_poses, 1, 6, render_steps),
        create_virtual_camera_path(cam_poses, 6, 2, render_steps),
        create_virtual_camera_path(cam_poses, 2, 3, render_steps),
        create_virtual_camera_path(cam_poses, 3, 4, render_steps)
    ], axis=0)
    return render_poses


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


def render_path_spiral(pose, zrate=0.5, n_rots=2, n_frames=120):
    """ Virtual camera path that is a spiral motion around a fixed camera pose. """
    intrinsics = pose[:, 4:]
    focal = np.mean(pose[1, 4:])
    radii = np.array([0.1, 0.1, 0.1, 1.0])

    render_poses = []
    cam2world = viewmatrix(pose[:3, 2], pose[:3, 1], pose[:3, 3])
    up = pose[:, 1]
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


def lookat_matrix(target, position):
    """ Calculate lookat matrix. """
    mz = normalize(position - target)
    mx = normalize(np.cross([0., 1., 0.], mz))
    my = normalize(np.cross(mz, mx))
    return np.stack([mx, my, mz], axis=1)


def render_path_circle(pose, n_frames=120, depth=1.0):
    """ Virtual camera path that is a circular motion around a point in front of the given camera pose. """
    # Find look-at point in front of camera
    mid_point = np.array([0., 0., -depth, 1.])
    # Project look-at point to world coordinates
    cam2world = np.concatenate([pose[:, :4], [[0., 0., 0., 1.]]], axis=0)
    mid_point = cam2world @ mid_point
    mid_point = mid_point[:3]
    # Calculate distance from camera to look-at point in XY-plane
    x, z, y = pose[:, 3]
    radius = np.linalg.norm([x - mid_point[0], y - mid_point[2]])
    # Offset to get pose as starting camera position
    dist = y - mid_point[2]
    offset = np.arcsin(dist / radius)
    # Get points on circle moving around look-at point with same distance
    points_on_circle = np.array([[
        np.cos(2 * np.pi / n_frames * x + offset) * radius + mid_point[0],
        np.sin(2 * np.pi / n_frames * x + offset) * radius + mid_point[2]] for x in range(n_frames)])
    # Get camera positions of these points with equal height (Z-axis)
    translations = np.stack([points_on_circle[:, 0], np.array([z]).repeat(n_frames), points_on_circle[:, 1]], axis=1)
    # Calculate look-at matrices and create render poses
    render_poses = []
    for t in translations:
        r = lookat_matrix(mid_point, t)
        p = np.zeros_like(pose)
        p[:, :3] = r
        p[:, 3] = t
        p[:, 4:] = pose[:, 4:]
        render_poses.append(p)
    return np.stack(render_poses)


def convert_to_array(item, dtype=None):
    """ Convert dictionary or list to numpy array. """
    if isinstance(item, dict):
        item = {key: np.array(item[key], dtype=dtype) for key in item.keys()}
    elif isinstance(item, list):
        item = np.array(item, dtype=dtype)
    return item


def print_dataset_info(images, poses, depths, bounds, bounding_box, est_depths, time_values):
    """ Print information about the dataset. """
    print("> Loaded TUM 4D-OR dataset:")
    print(f">> Images: {images['train'].shape} (train), {images['test'].shape} (test)")
    print(f">> Poses: {poses['train'].shape} (train), {poses['test'].shape} (test), {poses['render'].shape} (render)")
    print(f">> Depths: {depths.shape}")
    print(f">> Depth bounds: {bounds[0]:.3f} (min), {bounds[1]:.3f} (max)")
    print(f">> Scene bounding box: {bounding_box[0]} (min), {bounding_box[1]} (max)")
    if est_depths is not None:
        print(f">> Estimated depth images: {est_depths.shape}")
    if time_values is not None:
        print(f">> Time values: {time_values['train'].shape} (train), {time_values['test'].shape} (test), {time_values['render'].shape} (render)")


def load_4DOR_data(datadir, take, train_cams, test_cams, scene_id_start, scene_id_end=None, stride=1, offset=0,
                   img_scale=1, depth_gradients=False, render_steps=None, render_path='circle'):
    """ Create 4D-OR dataset:
    - datadir: directory where the data is stored
    - take: surgical procedure number to include
    - train_cams: list of camera IDs to use for training
    - test_cams: list of camera IDs to use for testing
    - scene_id_start: first scene ID to include (0, 1, 2, ...)
    - scene_id_end: last scene ID to include (None if training on static scene only)
    - stride: step size for including scene IDs
    - offset: used for multi-GPU training, such that each GPU gets different scenes
    - img_scale: downscaling factor for images (1 for original size)
    - depth_gradients: include estimated depth gradients for training
    - render_steps: number of frames to include in render video
    - render_path: render video type ('circle' or 'spiral')
    """

    # Read camera information
    readdir = os.path.join(datadir, 'export_holistic_take' + str(take) + '_processed')
    all_cameras = list(set(train_cams + test_cams))
    cams_info = read_cameras_information(readdir, all_cameras)
    cam_poses = create_camera_poses(cams_info, img_scale)

    # Select scene IDs to be read
    scene_ids = range(offset + scene_id_start, scene_id_end + 1, stride) if scene_id_end else [scene_id_start]
    
    # Read list that converts scene IDs to frame numbers
    with open(os.path.join(readdir, 'timestamp_to_pcd_and_frames_list.json'), 'r') as file:
        timestamps = json.load(file)

    # Initiate lists to store data temporarily
    images = {'train': [], 'test': []}
    poses = {'train': [], 'test': [], 'render': []}
    depths = []
    bounds = [np.inf, -np.inf]
    est_depths = [] if depth_gradients else None
    time_values = {'train': [], 'test': [], 'render': []} if scene_id_end else None

    # Per camera and scene, get the data
    for c in all_cameras:

        # Get camera height and width
        camera_pose = cam_poses[c]
        H, W = camera_pose[0, 4:].astype(int)
        cam = str(c)

        for scene_id in scene_ids:
            # Get timestamp from JSON data
            color_frame_num = timestamps[scene_id][1]['color_' + cam]
            depth_frame_num = timestamps[scene_id][1]['depth_' + cam]

            # Read images and update min and max boundaries
            color_img = read_color_image(readdir, cam, color_frame_num, width=W, height=H, img_scale=img_scale)
            depth_img = read_depth_image(readdir, 'depthimage', cam, depth_frame_num, width=W, height=H, img_scale=img_scale)

            # Read estimated depth image if necessary
            if depth_gradients and c in train_cams:
                # est_depth_img = read_depth_image(readdir, 'depthimage_dpt', cam, depth_frame_num, width=W, height=H, img_scale=img_scale)
                est_depth_img = read_depth_image(readdir, 'depthimage_marigold', cam, depth_frame_num, width=W, height=H, img_scale=img_scale)
                est_depths.append(est_depth_img)

            # Put the data in the right lists
            if c in train_cams:
                images['train'].append(color_img)
                poses['train'].append(camera_pose)
                depths.append(depth_img)
                bounds = update_boundaries(bounds, depth_img)
                if time_values:
                    time_values['train'].append(scene_id)
            if c in test_cams:
                images['test'].append(color_img)
                poses['test'].append(camera_pose)
                if time_values:
                    time_values['test'].append(scene_id)
            
    # Create camera poses for render video
    if render_path == 'circle':
        poses['render'] = render_path_circle(poses['test'][0], n_frames=render_steps)
    elif render_path == 'spiral':
        poses['render'] = render_path_spiral(poses['test'][0], zrate=0.1, n_rots=2, n_frames=render_steps)
    else:
        poses['render'] = create_render_video_poses(cam_poses, render_steps=render_steps)
    
    if time_values:
        render_times = np.linspace(scene_id_start, scene_id_end, len(poses['render']))
        time_values['render'] = np.rint(render_times)  # Round to nearest integer

    # Convert lists to numpy arrays
    images = convert_to_array(images)
    poses = convert_to_array(poses)
    depths = convert_to_array(depths)
    bounds = convert_to_array(bounds)
    est_depths = convert_to_array(est_depths) if est_depths else None
    time_values = convert_to_array(time_values, dtype=np.float32) if time_values else None

    # Set bounding box of scene, with 4 dimensions if time values are given
    if time_values:
        bounding_box = (
            torch.tensor([-4.0, -1.0, -4.0, scene_id_start]),
            torch.tensor([4.0, 3.0, 4.0, scene_id_end])
        )
    else:
        bounding_box = (
            torch.tensor([-4.0, -1.0, -4.0]),
            torch.tensor([4.0, 3.0, 4.0])
        )

    print_dataset_info(images, poses, depths, bounds, bounding_box, est_depths, time_values)
    return images, poses, depths, bounds, bounding_box, est_depths, time_values
