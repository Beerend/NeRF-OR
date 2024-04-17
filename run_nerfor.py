import os
import numpy as np
import imageio
import time
import configargparse
from tqdm import tqdm, trange
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from collections import OrderedDict

from skimage.metrics import structural_similarity
from lpips import LPIPS

from radam import RAdam
from loss import total_variation_loss
from load_4dor import load_4DOR_data
from load_nvsrgbd import load_NVSRGBD_data
from run_nerfor_helpers import img2mse, to8b, mse2psnr, get_embedder, get_rays, get_rays_np, sample_pdf, NeRF, NeRFSmall


def setup_dataparallel(rank, world_size):
    """ Create environment for multi-GPU training. """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_dataparallel():
    """ Clean up environment for multi-GPU training."""
    dist.destroy_process_group()


def load_on_single_gpu(model_dict):
    """ If model is saved in multi-GPU format, convert to single-GPU. """
    if 'module.' in list(model_dict.keys())[0]:
        new_model_dict = OrderedDict()
        for k, v in model_dict.items():
            name = k.replace('module.', '')
            new_model_dict[name] = v
        return new_model_dict
    return model_dict


def batchify(fn, chunk):
    """ Apply function to inputs in chunks. """
    # If chunk is None, return the function as is
    if chunk is None:
        return fn
    # Else return function that splits inputs in chunk size batches and applies the function to each
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """ Embed the input, send through network function in chunks and return outputs. """
    # Embed flattened inputs
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    # If embedded with hash encoding, it returns a mask as well
    keep_mask = None
    if not torch.is_tensor(embedded):
        embedded, keep_mask = embedded

    # Embed flattened viewing directions
    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape[0], inputs.shape[1], 3)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    # Send through network
    outputs_flat = batchify(fn, netchunk)(embedded)

    # When using hash encoding, mask out invalid points
    if keep_mask is not None:
        outputs_flat[~keep_mask, -1] = 0

    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """ Render rays in chunks and concatenate results. """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])
    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(cam_intr, chunk=1024*32, rays=None, c2w=None, near=0., far=1., use_viewdirs=False, time_values=None, **kwargs):
    """ Render a batch of rays or a full image:
    - cam_intr: camera intrinsic parameters
    - chunk: number of rays processed in parallel
    - rays: ray origins and directions
    - c2w: camera-to-world transformation matrix (if rendering full image)
    - near: closest point to sample along rays
    - far: furthest point to sample along rays
    - use_viewdirs: whether to use viewing directions
    - time_values: when training dynamic scenes, time values for each ray
    """
    # Boolean for using time values
    use_time_values = True if (time_values is not False and time_values is not None) else False
    
    # If camera matrix is given, render a full image, else render individual rays
    if c2w is not None:
        rays_o, rays_d = get_rays(cam_intr, c2w)
        if use_time_values:
            H, W = cam_intr[0].astype(int)
            rays_t = time_values.expand(H*W, 1)
    else:
        if use_time_values:
            rays_t, rays_o, rays_d = rays
        else:
            rays_o, rays_d = rays

    if use_viewdirs:
        viewdirs = rays_d
        # viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    # Get shape of rays
    sh = rays_d.shape
    
    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    # Expand far and near values to match rays shape
    far = far * torch.ones_like(rays_d[...,:1])
    near = near * torch.ones_like(rays_d[...,:1])
    
    # Combine data into rays
    rays = torch.cat([rays_o, rays_d, near, far], -1)  # [B, 8(ro+rd+n+f)]
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)  # [B, 11(ro+rd+n+f+vd)]
    if use_time_values:
        rays = torch.cat([rays_t[:, :1], rays], -1)  # [B, 12(t+ro+rd+n+f+vd)]
    
    # Render rays in batches and reshape outputs to match rays dimensions
    all_ret = batchify_rays(rays, chunk, time_values=use_time_values, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    # Return rgb, depth and acc first, put rest of outputs in a dictionary
    k_extract = ['rgb_map', 'depth_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, cam_intrs, chunk, render_kwargs, gt_imgs=None, gt_depth_imgs=None, savedir=None, render_factor=0, render_times=None):
    """ Render images along a virtual camera path:
    - render_poses: camera poses along the virtual path
    - cam_intrs: camera intrinsic parameters
    - chunk: number of rays processed in parallel
    - render_kwargs: rendering arguments
    - gt_imgs: ground truth images for calculation of evaluation metrics
    - gt_depth_imgs: ground truth depth images for calculating error in predicted depth maps
    - savedir: directory to save rendered images
    - render_factor: factor to downsample rendered images (keep 0 for same resolution as training images)
    - render_times: time values for each render pose when training dynamic scenes
    """
    # Get camera intrinsic parameters and adapt if rendering at lower resolution
    near = render_kwargs['near']
    far = render_kwargs['far']
    render_cam_intrs = cam_intrs
    if render_factor>0:
        render_cam_intrs[:, 0] = np.rint(render_cam_intrs[:, 0] / render_factor)
        render_cam_intrs[:, 1:] /= render_factor

    # Create lists for images, depths and PSNRs
    rgbs = []
    depths = []
    psnrs = []
    ssims = []
    lpips = []
    depth_errors = []

    # Initiate LPIPS function
    lpips_fn = LPIPS(net='vgg').cuda()

    # Render poses in a loop
    for i, c2w in enumerate(tqdm(render_poses)):
        # Render image
        time_values = render_times[i] if render_times is not None else None
        rgb, depth, _ = render(render_cam_intrs[i], chunk=chunk, c2w=c2w[:3,:4], time_values=time_values, **render_kwargs)

        # Get output and normalize depth to [0,1]
        rgb = rgb.cpu().numpy()
        depth = depth.cpu().numpy()
        norm_depth = (depth - near) / (far - near)

        # append data to lists
        rgbs.append(rgb)
        depths.append(norm_depth)
                
        # If ground truth images are given, calculate PSNR
        if gt_imgs is not None and render_factor==0:
            psnrs.append(-10. * np.log10(np.mean(np.square(rgb - gt_imgs[i]))))
            ssims.append(structural_similarity(rgb, gt_imgs[i], multichannel=True, data_range=1.0))
            lpips.append(lpips_fn(torch.tensor(rgb).cuda().permute(2, 0, 1).unsqueeze(0),
                                  torch.tensor(gt_imgs[i]).cuda().permute(2, 0, 1).unsqueeze(0)).cpu().item())
        
        if gt_depth_imgs is not None and render_factor==0:
            # Least squares before calcualting the depth error
            nonzero_idxs = np.nonzero(gt_depth_imgs[i])
            values_x = depth[nonzero_idxs].flatten()
            values_y = gt_depth_imgs[i][nonzero_idxs].flatten()
            values_x = np.vstack([values_x, np.ones(len(values_x))]).T
            gain, offset = np.linalg.lstsq(values_x, values_y, rcond=None)[0]
            depth_aligned = depth[nonzero_idxs].flatten() * gain + offset
            depth_mse = np.mean((depth_aligned - values_y)**2)
            depth_errors.append(depth_mse)
        
        # Make plot of the results if savedir is given
        if savedir is not None:
            fig = plt.figure(figsize=(25, 10))

            # Plot RGB image
            ax = fig.add_subplot(1, 2, 1)
            ax.imshow(to8b(rgb))
            ax.axis('off')

            # Plot depth image
            ax = fig.add_subplot(1, 2, 2)
            # im = ax.imshow(depth, cmap='turbo_r', vmin=near, vmax=far)
            im = ax.imshow(depth, cmap='turbo_r')
            ax.axis('off')

            # Add colorbar to depth image
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')

            # Save image
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            plt.savefig(filename, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

            # Save images also as separate color and depth images
            plt.imsave(os.path.join(savedir, 'color_{:03d}.png'.format(i)), to8b(rgb))
            plt.imsave(os.path.join(savedir, 'depth_{:03d}.png'.format(i)), depth, cmap='turbo_r')
            plt.imsave(os.path.join(savedir, 'norm_depth_{:03d}.png'.format(i)), depth, cmap='turbo_r', vmin=near, vmax=far)
    
    # Stack data
    rgbs = np.stack(rgbs, 0)
    depths = np.stack(depths, 0)
    
    # If PSNRs are calculated, print average and save to file
    if len(psnrs) > 0:
        avg_psnr = sum(psnrs) / len(psnrs)
        avg_ssim = sum(ssims) / len(ssims)
        avg_lpips = sum(lpips) / len(lpips)
        
        avg_depth_error = 0.0
        if len(depth_errors) > 0:
            avg_depth_error = sum(depth_errors) / len(depth_errors)

        print(f">> Average PSNR: {avg_psnr:.3f} | Average SSIM: {avg_ssim:.3f} | Average LPIPS: {avg_lpips:.3f} | MSE depth: {avg_depth_error:.6f}")
        with open(os.path.join(savedir, "test_psnrs_avg{:0.2f}.pkl".format(avg_psnr)), "wb") as fp:
            pickle.dump(psnrs, fp)
        with open(os.path.join(savedir, "test_ssims_avg{:0.2f}.pkl".format(avg_ssim)), "wb") as fp:
            pickle.dump(ssims, fp)
        with open(os.path.join(savedir, "test_lpips_avg{:0.2f}.pkl".format(avg_lpips)), "wb") as fp:
            pickle.dump(lpips, fp)
        os.rename(savedir, f'{savedir}_{avg_psnr:.2f}_{avg_ssim:.2f}_{avg_lpips:.2f}')
        
    return rgbs, depths


def create_nerf(args):
    """ Build NeRF-OR model, including embedders and neural networks. """
    # Set arguments for embedder of sampled locations
    if args.i_embed == 0:
        # Positional encoding
        embed_kwargs = {'include_input': True, 'input_dims': 3, 'max_freq_log2': args.multires-1, 'num_freqs': args.multires,
                        'log_sampling': True, 'periodic_fns': [torch.sin, torch.cos]}
    elif args.i_embed == 1:
        # Hash encoding
        embed_kwargs = {'hash_enc_4d': args.hash_enc_4d, 'bounding_box': args.bounding_box, 'finest_resolution': args.finest_res, 'device': args.rank}
        if args.hash_enc_4d:
            embed_kwargs['log2_hashmap_size_3D'] = args.log2_hashmap_size_3D
            embed_kwargs['log2_hashmap_size_4D'] = args.log2_hashmap_size_4D
            embed_kwargs['base_resolution_time'] = args.base_res_time
            embed_kwargs['finest_resolution_time'] = args.finest_res_time
        else:
            embed_kwargs['log2_hashmap_size'] = args.log2_hashmap_size_3D
    else:
        embed_kwargs = {}

    # Get embedder of sampled locations
    embed_fn, input_ch = get_embedder(args.i_embed, embed_kwargs)

    # When using hash encoding, apply DDP to embedder and print info
    if args.i_embed==1:
        # Make trainable for multi-GPU
        embed_fn = embed_fn.to(args.rank)
        if args.multi_gpu:
            embed_fn = DDP(embed_fn, device_ids=[args.rank], output_device=args.rank, find_unused_parameters=True)

        # Print embedder info
        print(f'>> Hash encoding ({input_ch}-dim): 4D={args.hash_enc_4d}, L=16, T=2^{args.log2_hashmap_size_3D} (3D), T=2^{args.log2_hashmap_size_4D} (4D), F=2, N_min=16, N_max={args.finest_res}')
        box_min, box_max = args.bounding_box
        if args.hash_enc_4d:
            grid_size = (box_max - box_min) / torch.tensor([args.finest_res, args.finest_res, args.finest_res, args.finest_res_time])
        else:
            grid_size = (box_max - box_min) / args.finest_res
        print('>> Finest grid sizes:', ['{:.3f}'.format(i) for i in grid_size])
        embedding_params = list(embed_fn.parameters(recurse=True))
        num_params = sum(p.numel() for p in embedding_params if p.requires_grad)
        print(f'>> Num params (trainable) in hash encoding: {num_params}')
    
    if args.use_viewdirs:
        # Set arguments for embedder of viewing directions
        if args.i_embed_views == 0:
            # Positional encoding
            embed_kwargs = {'include_input': True, 'input_dims': 3, 'max_freq_log2': args.multires-1, 'num_freqs': args.multires,
                            'log_sampling': True, 'periodic_fns': [torch.sin, torch.cos]}
        elif args.i_embed_views == 1:
            print('Cannot choose hash encoding for viewing directions!')
            assert False
        else:
            embed_kwargs = {}
    
        # Get embedder of viewing directions
        embeddirs_fn, input_ch_views = get_embedder(args.i_embed_views, embed_kwargs)
        viewdir_enc_name = 'positional' if args.i_embed_views==0 else 'spherical' if args.i_embed_views==2 else 'unknown'
        print(f'>> Using viewdirs, with {input_ch_views}-dim {viewdir_enc_name}-encoding')
   
    else:
        input_ch_views = 0
        embeddirs_fn = None

    # Get coarse model
    if args.i_embed == 1:
        model = NeRFSmall(num_layers=2, hidden_dim=args.netwidth, geo_feat_dim=15, num_layers_color=args.netdepth+1,
                          hidden_dim_color=args.netwidth, input_ch=input_ch, input_ch_views=input_ch_views).to(args.rank)
    else:
        output_ch = 5 if args.N_importance > 0 else 4
        skips = [4]
        model = NeRF(D=args.netdepth, W=args.netwidth, input_ch=input_ch, output_ch=output_ch, skips=skips,
                     input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(args.rank)
        
    # Set coarse model to be trainable for multi-GPU and print parameter info
    if args.multi_gpu:
        model = DDP(model, device_ids=[args.rank], output_device=args.rank, find_unused_parameters=True)
    grad_vars = list(model.parameters())
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'>> Num params (trainable) in coarse model: {num_params}')

    # Get fine model
    model_fine = None
    if args.N_importance > 0:
        if args.i_embed == 1:
            model_fine = NeRFSmall(num_layers=2, hidden_dim=args.netwidth_fine, geo_feat_dim=15, num_layers_color=args.netdepth_fine+1,
                                   hidden_dim_color=args.netwidth_fine, input_ch=input_ch, input_ch_views=input_ch_views).to(args.rank)
        else:
            model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine, input_ch=input_ch, output_ch=output_ch, skips=skips,
                              input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(args.rank)
            
        # Set fine model to be trainable for multi-GPU and print parameter info
        if args.multi_gpu:
            model_fine = DDP(model_fine, device_ids=[args.rank], output_device=args.rank, find_unused_parameters=True)
        grad_vars += list(model_fine.parameters())
        num_params = sum(p.numel() for p in model_fine.parameters() if p.requires_grad)
        print(f'>> Num params (trainable) in fine model: {num_params}')

    # Set network query function
    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn, embed_fn=embed_fn,
                                                                         embeddirs_fn=embeddirs_fn, netchunk=args.netchunk)
    
    # Create optimizer
    if args.i_embed == 1:
        optimizer = RAdam([{'params': grad_vars, 'weight_decay': 1e-6}, {'params': embedding_params, 'eps': 1e-15}],
                          lr=args.lrate, betas=(0.9, 0.99))
    else:
        optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))
    
    # Find checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(args.basedir, args.expname, f) for f in sorted(os.listdir(os.path.join(args.basedir, args.expname))) if 'tar' in f]
    print('>> Found ckpts:', ckpts)

    # If checkpoint is found, reload model weights and optimizer state
    start = 0
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]

        # Map locations and load checkpoint
        map_location = {'cuda:0': 'cuda:%d' % args.rank}
        print('>> Reloading from:', ckpt_path, '| Map location:', map_location)
        ckpt = torch.load(ckpt_path, map_location=map_location)

        # Load items
        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        if args.multi_gpu:
            model.module.load_state_dict(ckpt['network_fn_state_dict'])
        else:
            model.load_state_dict(ckpt['network_fn_state_dict'])

        if args.multi_gpu:
            model_fine.module.load_state_dict(ckpt['network_fine_state_dict'])
        else:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

        if args.i_embed == 1:
            if args.multi_gpu:
                embed_fn.module.load_state_dict(ckpt['embed_fn_state_dict'])
            else:
                embed_fn.load_state_dict(ckpt['embed_fn_state_dict'])

    # Build render kwargs for training
    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'embed_fn': embed_fn,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
        'lindisp': args.lindisp
    }

    # Build render kwargs for testing
    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.0

    return render_kwargs_train, render_kwargs_test, start, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0.0, white_bkgd=False, device='cuda'):
    """ Convert raw model outputs to RGB and depth maps. """
    # Define function to convert raw values into alpha
    raw2alpha = lambda raw, dists, act_fn = F.relu: 1.0 - torch.exp(-act_fn(raw) * dists)

    # Get distances between samples
    dists = z_vals[...,1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.tensor([1e10], device=device).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    # Put RGB to [0,1] with sigmoid
    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]

    # When training, add noise to raw output
    noise = noise = torch.randn(raw[..., 3].shape, device=device) * raw_noise_std if raw_noise_std > 0.0 else 0.0

    # Calculate alpha, weights, RGB and depth
    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]
    # depth_map = torch.sum(weights * z_vals, -1) / torch.sum(weights, -1)
    depth_map = torch.sum(weights * z_vals, -1)
    # disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map)
    # disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map) / torch.sum(weights, -1)
    acc_map = torch.sum(weights, -1)

    # Depth variance
    depth_var_map = torch.sum(weights * (depth_map[..., None] - z_vals)**2, -1)

    # If white background is desired, add white where there is no opacity
    if white_bkgd:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    # Calculate weights sparsity loss
    sparsity_loss = Categorical(probs = torch.cat([weights, 1.0-weights.sum(-1, keepdim=True)+1e-6], dim=-1)).entropy()

    return rgb_map, weights, depth_map, depth_var_map, sparsity_loss


def render_rays(ray_batch, network_fn, network_query_fn, N_samples, retraw=False, lindisp=False, perturb=0.0, N_importance=0,
                network_fine=None, white_bkgd=False, raw_noise_std=0.0, time_values=False, device='cuda', **kwargs):
    """ Render a batch of rays:
    - ray_batch: rays, including origins, directions, bounds, viewing directions and time values
    - network_fn: coarse model predicting color and material densities
    - network_query_fn: function to send sampled points through model
    - N_samples: number of sampled points per ray for coarse model
    - retraw: return raw model outputs as well
    - lindisp: use linear sampling in depth space
    - perturb: add noise to sampled locations, necessary during training
    - N_importance: number of sampled points per ray for fine model
    - network_fine: fine model predicting color and material densities
    - white_bkgd: add white instead of black background to RGB map
    - raw_noise_std: noise added to raw model outputs, necessary during training
    - time_values: time values for each ray, necessary for dynamic scenes
    - device: device to run model on
    """
    # Split ray batch into origins, directions, bounds, viewing directions and times    
    if time_values:
        rays_t, rays_o, rays_d = ray_batch[:, :1], ray_batch[:, 1:4], ray_batch[:, 4:7]
        bounds = torch.reshape(ray_batch[:, 7:9], [-1, 1, 2])
        viewdirs = ray_batch[:, 9:12] if ray_batch.shape[-1] > 9 else None
    else:
        rays_o, rays_d = ray_batch[:, :3], ray_batch[:, 3:6]
        bounds = torch.reshape(ray_batch[:, 6:8], [-1 ,1, 2])
        viewdirs = ray_batch[:, 8:11] if ray_batch.shape[-1] > 8 else None
    near, far = bounds[...,0], bounds[..., 1]
    
    # Determine sample locations in Z-direction
    t_vals = torch.linspace(0., 1., steps=N_samples, device=device)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))
    z_vals = z_vals.expand([rays_o.shape[0], N_samples])

    # Slightly randomize sampling locations
    if perturb > 0.0:
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        t_rand = torch.rand(z_vals.shape, device=device)
        z_vals = lower + (upper - lower) * t_rand

    # Transform into locations in 3D space (and time if applicable)
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
    if time_values:
        pts = torch.cat([pts, rays_t.unsqueeze(1).repeat(1, N_samples, 1)], 2)

    # Send points through network and obtain results
    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, weights, depth_map, _, sparsity_loss = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, device=device)

    # Sample more points for fine model
    if N_importance > 0:
        rgb_map_0, depth_map_0, sparsity_loss_0 = rgb_map, depth_map, sparsity_loss

        # Sample new locations based on weights from coarse model
        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.0), device=device)
        z_samples = z_samples.detach()
        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

        # Transform into locations in 3D space (and time if applicable)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]
        if time_values:
            pts = torch.cat([pts, rays_t.unsqueeze(1).repeat(1, N_samples + N_importance, 1)], 2)

        # Send points through network and obtain results
        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, run_fn)
        rgb_map, weights, depth_map, depth_var_map, sparsity_loss = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, device=device)

    # Return results in dictionary
    ret = {'rgb_map' : rgb_map, 'depth_map' : depth_map, 'depth_var_map': depth_var_map, 'sparsity_loss': sparsity_loss}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['depth0'] = depth_map_0
        ret['sparsity_loss0'] = sparsity_loss_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)

    return ret


def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument('--dataset', type=str, default='4D-OR')
    parser.add_argument("--datadir", type=str, default='/home/geratsbga1/4D-OR',
                        help='input data directory')

    # training options
    parser.add_argument("--N_iters", type=int, default=50000,
                        help='number of iterations')
    parser.add_argument("--netdepth", type=int, default=2,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=64,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=2,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=64,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=float, default=0.1,
                        help='exponential learning rate decay (in total over all iterations)')
    parser.add_argument("--chunk", type=int, default=1024*32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--jit_gpu", action='store_true',
                        help='set True if Tensors of ray batches should be send to GPU just in time, instead of whole images at the start')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.0,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=1,
                        help='set 1 for hashed embedding, 0 for default positional encoding, 2 for spherical')
    parser.add_argument("--hash_enc_4d", action='store_true',
                        help='set true for 4D hash encoding that embeds both space and time')
    parser.add_argument("--i_embed_views", type=int, default=2,
                        help='set 1 for hashed embedding, 0 for default positional encoding, 2 for spherical')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # TUM 4D-OR dataset flags
    parser.add_argument('--take', type=int, default=1)
    parser.add_argument('--scene_id_start', type=int, default=0)
    parser.add_argument('--scene_id_end', type=int, default=None)
    parser.add_argument('--scene_id_stride', type=int, default=1)
    parser.add_argument('--train_cams', type=int, default=[1, 2, 3, 4, 5, 6], nargs='*')
    parser.add_argument('--test_cams', type=int, default=[1, 2, 3, 4, 5, 6], nargs='*')
    parser.add_argument('--image_scale', type=float, default=4.0)
    parser.add_argument('--render_steps', type=int, default=25)
    parser.add_argument('--render_path', type=str, default='circle')

    # NVS-RGBD dataset flags
    parser.add_argument('--recording_device', type=str, default='kinect')
    parser.add_argument('--scene', type=int, default=1)

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')

    ## llff flags
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    
    # Depth-supervision options
    parser.add_argument("--huber_loss", action='store_true',
                        help='use huber loss instead of MSE')
    parser.add_argument("--depth_supervision", action='store_true',
                        help='use depth supervision')
    parser.add_argument("--depth_alpha", type=float, default=0.1,
                        help='depth alpha used for loss')
    parser.add_argument("--depth_alpha_decay", type=float, default=0.1,
                        help='depth alpha decay (in total over all iterations)')
    parser.add_argument('--normalize_depth', action='store_true',
                        help='normalize depth before calculating loss')
    parser.add_argument('--relative_loss', action='store_true',
                        help='use relative depth loss')
    parser.add_argument("--depth_gradients", action='store_true',
                        help='supervise with gradients in estimated depth images')
    parser.add_argument("--grad_patch_size", type=int, default=8,
                        help='size of patches sampled for depth gradient supervision')
    parser.add_argument("--grad_res_levels", type=int, default=4,
                        help='number of resolution levels to use for depth gradient supervision')
    parser.add_argument("--depth_beta", type=float, default=0.2,
                        help='depth beta to weight depth gradient loss')
    parser.add_argument("--depth_beta_decay", type=float, default=0.1,
                        help='depth beta decay (in total over all iterations)')
    
    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=1000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=5000,
                        help='frequency of render_poses video saving')
    parser.add_argument("--plot_input_data", action='store_true',
                        help='plot and save input data with matplotlib')

    # Hash encoding options
    parser.add_argument("--finest_res",   type=int, default=512,
                        help='finest resolution for hashed embedding')
    parser.add_argument("--log2_hashmap_size_3D",   type=int, default=15,
                        help='log2 of hashmap size')
    parser.add_argument("--log2_hashmap_size_4D",   type=int, default=24,
                        help='log2 of hashmap size')
    parser.add_argument('--base_res_time', type=int, default=2,
                        help='base resolution for hash embedding of time dimension')
    parser.add_argument('--finest_res_time', type=int, default=32,
                        help='finest resolution for hash embedding of time dimension')
    parser.add_argument("--sparse-loss-weight", type=float, default=1e-10,
                        help='learning rate')
    parser.add_argument("--tv-loss-weight", type=float, default=1e-6,
                        help='learning rate')

    # Sample 3D space
    parser.add_argument("--sample_only", action='store_true',
                        help='do not optimize, only sample 3D space as a regular grid')
    parser.add_argument("--sample_step_size", type=float, default=0.1,
                        help='step size of points sampled in all 3 spatial dimensions')
    
    return parser


def train(rank, world_size):
    """ Main function for training NeRF-OR. """
    # Setup script arguments
    parser = config_parser()
    args = parser.parse_args()
    print(f'> Started script with PID={os.getpid()}')

    # Setup dataparallel training
    args.multi_gpu = world_size > 1
    args.rank = rank
    np.random.seed(rank)
    print(f'> Multi GPU training: {args.multi_gpu} | Rank: {rank}')
    if args.multi_gpu:
        setup_dataparallel(rank, world_size)
    
    # Load dataset
    print('\n> Loading data...')
    if args.dataset == '4D-OR':
        stride = int(args.scene_id_stride * world_size) if args.scene_id_end is not None else args.scene_id_stride
        data_kwargs = {'scene_id_end': args.scene_id_end, 'stride': stride, 'offset': rank, 'img_scale': args.image_scale,
                    'depth_gradients': args.depth_gradients, 'render_steps': args.render_steps, 'render_path': args.render_path}
        images, poses, depths, bounds, bounding_box, est_depths, time_values = load_4DOR_data(args.datadir, args.take, args.train_cams,
                                                                                            args.test_cams, args.scene_id_start, **data_kwargs)
        depths_test = None
    elif args.dataset == 'NVS-RGBD':
        data_kwargs = {'img_scale': args.image_scale, 'depth_gradients': args.depth_gradients, 'render_steps': args.render_steps}
        images, poses, depths, bounds, bounding_box, est_depths = load_NVSRGBD_data(args.datadir, args.recording_device,
                                                                                    args.scene, **data_kwargs)
        depths_test = depths['test']
        depths = depths['train']

    # Get scene dimensions
    H, W = poses['train'][0, 0, 4:].astype(int)
    args.bounding_box = bounding_box
    near = np.min(bounds)
    far = np.max(bounds)
    print(f'>> Final depth bounds: {near=:.3f}, {far=:.3f}')
    
    # Get poses for rendering
    render_poses = poses['test'][:, :, :4] if args.render_test else poses['render'][:, :, :4]
    render_intrs = poses['test'][:, :, 4:] if args.render_test else poses['render'][:, :, 4:]

    # Set base directory and experiment name
    basedir = args.basedir
    expname = args.expname

    # Create log dir and copy the arguments and config file
    if args.rank == 0:
        os.makedirs(os.path.join(basedir, expname), exist_ok=True)
        f = os.path.join(basedir, expname, 'args.txt')
        with open(f, 'w') as file:
            for arg in sorted(vars(args)):
                attr = getattr(args, arg)
                file.write('{} = {}\n'.format(arg, attr))
        if args.config is not None:
            f = os.path.join(basedir, expname, 'config.txt')
            with open(f, 'w') as file:
                file.write(open(args.config, 'r').read())

    # Create NeRF model
    print('\n> Loading model...')
    render_kwargs_train, render_kwargs_test, start, optimizer = create_nerf(args)

    # Update arguments
    global_step = start
    extra_args = {'near' : near, 'far' : far, 'device' : args.rank}
    render_kwargs_train.update(extra_args)
    render_kwargs_test.update(extra_args)

    # Move test data to GPU
    print('\n> Moving test data to GPU...')
    poses_test = torch.tensor(poses['test'][:, :, :4], device=args.rank)
    if args.scene_id_end:
        test_times = torch.tensor(time_values['test'], device=args.rank)

    # Move render data to GPU
    print('\n> Moving render data to GPU...')
    render_poses = torch.tensor(render_poses, device=args.rank)
    if args.scene_id_end:
        render_times = time_values['test'] if args.render_test else time_values['render']
        render_times = torch.tensor(render_times, device=args.rank)
    
    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('\n> Render only...')
        with torch.no_grad():
            gt_imgs = images['test'] if args.render_test else None
            gt_depth_imgs = depths_test if args.render_test else None
            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            if args.scene_id_end:
                rgbs, depths = render_path(render_poses, render_intrs, args.chunk, render_kwargs_test, gt_imgs=gt_imgs, gt_depth_imgs=gt_depth_imgs,
                                           savedir=testsavedir, render_factor=args.render_factor, render_times=render_times)
            else:
                rgbs, depths = render_path(render_poses, render_intrs, args.chunk, render_kwargs_test, gt_imgs=gt_imgs, gt_depth_imgs=gt_depth_imgs,
                                           savedir=testsavedir, render_factor=args.render_factor)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(os.path.join(testsavedir, 'depth.mp4'), to8b(depths), fps=30, quality=8)
            return

    # Plot data for debugging
    if args.plot_input_data:
        print('\n> Creating plots for debugging...')
        import matplotlib.pyplot as plt
        for i, img in enumerate(images['train']):
            plt.imsave(f'debug/rgb_image_{i}_train.jpg', img)
        for i, img in enumerate(images['test']):
            plt.imsave(f'debug/rgb_image_{i}_test.jpg', img)
        for i, depth in enumerate(depths):
            plt.imsave(f'debug/depth_img_{i}.jpg', depth, cmap='turbo_r', vmin=near, vmax=far)
        if est_depths is not None:
            for i, est_depth in enumerate(est_depths):
                plt.imsave(f'debug/est_depth_{i}.jpg', est_depth, cmap='turbo_r', vmin=near, vmax=far)
    
    # Get rays from training images
    print('\n> Get rays for training...')
    rays = np.stack([get_rays_np(i, p) for i, p in zip(poses['train'][:, :, 4:], poses['train'][:, :, :4])])
    rays_rgb = np.concatenate([rays, images['train'][:, None]], 1) # [N, ro+rd+rgb, H, W, 3]
    rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
    print(f'>> Rays RGB: {rays_rgb.shape}')

    # Add time values to rays
    time_values_in_rays = True if args.scene_id_end is not None else False
    if time_values_in_rays:
        train_times = np.repeat(time_values['train'], H*W, 0)  # [N*H*W]
        train_times = np.repeat(train_times[:, np.newaxis, np.newaxis], 3, 2)  # [N*H*W, 1, 3]
        train_times = np.reshape(train_times, (-1, H, W, 1, 3))  # [N, H, W, 1, 3]
        rays_rgb = np.concatenate([train_times, rays_rgb], 3)  # [N, H, W, t+ro+rd+rgb, 3]
        print(f'>> Rays RGB (after adding time values): {rays_rgb.shape}')

    # Move training data to GPU (if JIT, only create tensors)
    print(f'\n> Moving training data to GPU... (JIT={args.jit_gpu})')
    if args.jit_gpu:
        rays_rgb = torch.as_tensor(rays_rgb, dtype=torch.float, device=torch.device('cpu'))
        if args.depth_supervision:
            rays_depth = torch.as_tensor(depths, dtype=torch.float, device=torch.device('cpu'))
        if args.depth_gradients:
            rays_dense = torch.as_tensor(est_depths, dtype=torch.float, device=torch.device('cpu'))
    else:
        rays_rgb = torch.tensor(rays_rgb, device=args.rank)
        if args.depth_supervision:
            rays_depth = torch.tensor(depths, device=args.rank)
        if args.depth_gradients:
            rays_dense = torch.tensor(est_depths, device=args.rank)

    # Set parameters before training loop
    loss_list = []
    psnr_list = []
    time_list = []
    start = start + 1
    N_rand = args.N_rand
    N_iters = args.N_iters + 1
    N_imgs = rays_rgb.shape[0]
    start_time = time.time()

    # Initiate patch-size, resolution levels and start up-left corner indexes of random patches
    patch_size = args.grad_patch_size
    resolution_levels = args.grad_res_levels
    resolutions = [2**level for level in range(resolution_levels)]  # downsampling: 1x, 2x, 4x, ...
    num_patches = N_rand // ((patch_size**2) * resolution_levels)

    # Start training loop
    print('\n> Start training...')
    for i in trange(start, N_iters):
        optimizer.zero_grad()

        # Given the indexes, sample random patches from rays and estimated depth values on different resolution levels
        patches_rays = []
        target_depths = []
        target_grads = []
        for res in resolutions:
            x_idxs = np.random.randint(0, W - patch_size * res + 1, size=num_patches)
            y_idxs = np.random.randint(0, H - patch_size * res + 1, size=num_patches)
            i_idxs = np.random.randint(0, N_imgs, size=num_patches)
            patches_rays.append(torch.stack([rays_rgb[i, y:y+(patch_size*res):res, x:x+(patch_size*res):res] for i, y, x in zip(i_idxs, y_idxs, x_idxs)]))  # [N, H, W, ?, 3]
            if args.depth_supervision:
                target_depths.append(torch.stack([rays_depth[i, y:y+(patch_size*res):res, x:x+(patch_size*res):res] for i, y, x in zip(i_idxs, y_idxs, x_idxs)]))  # [N, H, W]
            if args.depth_gradients:
                target_grads.append(torch.stack([rays_dense[i, y:y+(patch_size*res):res, x:x+(patch_size*res):res] for i, y, x in zip(i_idxs, y_idxs, x_idxs)]))  # [N, H, W]

        patches_rays = torch.cat(patches_rays, dim=0)  # L*N, H, W, ?, 3
        patches_rays = patches_rays.view(num_patches * resolution_levels * patch_size * patch_size, patches_rays.shape[3], 3)  # L*N*H*W, ?, 3
        patches_rays = torch.transpose(patches_rays, 0, 1)  # ?, L*N*H*W, 3
        patches_rays, target_rgbs = torch.split(patches_rays, [patches_rays.shape[0]-1, 1], dim=0)  # ([?-1, L*N*H*W, 3], [L*N*H*W, 3])

        # Make target depths
        if args.depth_supervision:
            target_depths = torch.cat(target_depths, dim=0)  # N*L, H, W

        # Calculate target gradients
        if args.depth_gradients:
            target_grads = torch.cat(target_grads, dim=0)  # N*L, H, W
            target_grads = torch.stack(torch.gradient(target_grads, dim=(1, 2)))  # 2, N*L, H, W
        
        # If sending data to GPU in JIT manner, now load to GPU
        if args.jit_gpu:
            patches_rays = patches_rays.to(args.rank)
            target_rgbs = target_rgbs.to(args.rank)
            if args.depth_supervision:
                target_depths = target_depths.to(args.rank)
            if args.depth_gradients:
                target_grads = target_grads.to(args.rank)
        
        # Render random patches, extract depth values and calculate its gradients
        pred_rgbs, pred_depths, extras = render(None, chunk=args.chunk, rays=patches_rays, time_values=time_values_in_rays, **render_kwargs_train)
        pred_depths = pred_depths.view(num_patches * resolution_levels, patch_size, patch_size)  # N*L, H, W
        pred_grads = torch.stack(torch.gradient(pred_depths, dim=(1, 2)))  # 2, N*L, H, W

        # Calculate RGB loss
        img_loss = img2mse(pred_rgbs, target_rgbs)
        psnr = mse2psnr(img_loss, device=args.rank)
        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_rgbs)
            img_loss = img_loss + img_loss0
        
        # Calculate depth loss
        depth_loss = 0.0
        if args.depth_supervision:
            mask_zero_depths = torch.where(target_depths > 0.0, 1.0, 0.0)
            num_nonzero = torch.sum(mask_zero_depths)
            if num_nonzero > 0.0:
                if args.huber_loss:
                    error = F.smooth_l1_loss(pred_depths, target_depths, reduction='none', beta=1.0)
                else:
                    error = (pred_depths - target_depths)**2
                error = error * mask_zero_depths
                depth_loss = torch.sum(error) / num_nonzero

        # Compare gradients and calculate loss
        depth_grad_loss = 0.0
        if args.depth_gradients:
            target_grads = torch.stack([target_grads[0], target_grads[1], torch.full_like(target_grads[0], 1)], dim=3)  # N*L, H, W, 3
            target_normal_vector = torch.linalg.norm(target_grads, dim=3)  #N*L, H, W
            target_normals = target_grads / target_normal_vector[..., None]  # N*L, H, W, 3

            pred_grads = torch.stack([pred_grads[0], pred_grads[1], torch.full_like(pred_grads[0], 1)], dim=3)  # N*L, H, W, 3
            pred_normal_vector = torch.linalg.norm(pred_grads, dim=3)  #N*L, H, W
            pred_normals = pred_grads / pred_normal_vector[..., None]  # N*L, H, W, 3
            
            depth_grad_loss = torch.sum(pred_normals - target_normals, dim=3)  # N*L, H, W
            depth_grad_loss = torch.mean(depth_grad_loss**2)
        
        # Weight depth loss with lambda value
        if args.depth_supervision:
            alpha = args.depth_alpha * (args.depth_alpha_decay ** (global_step / args.N_iters))
            depth_loss = alpha * depth_loss

        # Weight loss with beta value
        if args.depth_gradients:
            beta = args.depth_beta * (args.depth_beta_decay ** (global_step / args.N_iters))
            depth_grad_loss = beta * depth_grad_loss

        # Calculate sparsity loss
        sparsity_loss = args.sparse_loss_weight*(extras["sparsity_loss"].sum() + extras["sparsity_loss0"].sum())

        # Calculate total loss
        loss = img_loss + sparsity_loss + depth_loss + depth_grad_loss

        # For the first iterations, add total variation loss
        if args.i_embed==1 and i<1000:
            embed_module = render_kwargs_train["embed_fn"].module if args.multi_gpu else render_kwargs_train["embed_fn"]
            n_levels = embed_module.n_levels
            min_res = embed_module.base_resolution
            max_res = embed_module.finest_resolution
            if args.hash_enc_4d:
                tv_loss_3d = sum(total_variation_loss(embed_module.embedders[1].embeddings[i], min_res, max_res, i, embed_module.log2_hashmap_size_3D,
                                                      n_levels=n_levels, device=args.rank) for i in range(n_levels))
                tv_loss_4d = sum(total_variation_loss(embed_module.embedders[0].embeddings[i], min_res, max_res, i, embed_module.log2_hashmap_size_4D,
                                                      n_levels=n_levels, device=args.rank) for i in range(n_levels))
                TV_loss = tv_loss_4d + tv_loss_3d
            else:
                TV_loss = sum(total_variation_loss(embed_module.embeddings[i], min_res, max_res, i, embed_module.log2_hashmap_size, n_levels=n_levels,
                                                   device=args.rank) for i in range(n_levels))
            loss = loss + args.tv_loss_weight * TV_loss

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Update learning rate
        new_lrate = args.lrate * (args.lrate_decay ** (global_step / args.N_iters))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        # Get time
        iteration_time = time.time() - start_time

        # Save weights
        if i%args.i_weights == 0 and args.rank == 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            network_module = render_kwargs_train["network_fn"].module if args.multi_gpu else render_kwargs_train["network_fn"]
            network_fine_module = render_kwargs_train["network_fine"].module if args.multi_gpu else render_kwargs_train["network_fine"]
            if args.i_embed == 1:
                embed_module = render_kwargs_train["embed_fn"].module if args.multi_gpu else render_kwargs_train["embed_fn"]
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': network_module.state_dict(),
                    'network_fine_state_dict': network_fine_module.state_dict(),
                    'embed_fn_state_dict': embed_module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
            else:
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': network_module.state_dict(),
                    'network_fine_state_dict': network_fine_module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
            print('> Saved checkpoints at', path)

        # Render and write video
        if i%args.i_video == 0 and i > 0 and args.rank == 0:
            print('\n> Render video...')
            renders_dir = os.path.join(basedir, expname, 'render_depths_{:06d}'.format(i))
            os.makedirs(renders_dir, exist_ok=True)
            with torch.no_grad():
                if args.scene_id_end is not None:
                    rgbs, depths = render_path(render_poses, render_intrs, args.chunk, render_kwargs_test, render_factor=args.render_factor,
                                               render_times=render_times, savedir=renders_dir)
                else:
                    rgbs, depths = render_path(render_poses, render_intrs, args.chunk, render_kwargs_test, render_factor=args.render_factor,
                                               savedir=renders_dir)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'depth.mp4', to8b(depths), fps=30, quality=8)

        # Render test set
        if i%args.i_testset == 0 and i > 0 and args.rank == 0:
            print('\n> Render test set...')
            testsavedir = os.path.join(basedir, expname, f'testset_cam{args.test_cams[0]}_{i:06d}')  # CHANGE THIS BACK TO 'testset_{:06d}'.format(i)
            os.makedirs(testsavedir, exist_ok=True)
            with torch.no_grad():
                if args.scene_id_end:
                    render_path(poses_test, poses['test'][:, :, 4:], args.chunk, render_kwargs_test, gt_imgs=images['test'], gt_depth_imgs=depths_test,
                                savedir=testsavedir, render_factor=args.render_factor, render_times=test_times)
                else:
                    render_path(poses_test, poses['test'][:, :, 4:], args.chunk, render_kwargs_test, gt_imgs=images['test'], gt_depth_imgs=depths_test,
                                savedir=testsavedir, render_factor=args.render_factor)
        
        # Log training progress
        if i%args.i_print == 0:
            depth_loss_value = depth_loss.item() if args.depth_supervision else -1.0
            depth_grad_value = depth_grad_loss.item() if args.depth_gradients else -1.0
            tqdm.write(f"[GPU{args.rank}] [TRAIN] Iter: {i} Loss: {loss.item():.3f}  RGB: {img_loss.item():.3f}, Sparsity: {sparsity_loss.item():.3f}, Depth: {depth_loss_value:.3f} (MSE), {depth_grad_value:.3f} (grads), PSNR: {psnr.item():.2f}")
            if args.rank == 0:
                loss_list.append(loss.item())
                psnr_list.append(psnr.item())
                time_list.append(iteration_time)
                loss_psnr_time = {"losses": loss_list, "psnr": psnr_list, "time": time_list}
                with open(os.path.join(basedir, expname, "loss_vs_time.pkl"), "wb") as fp:
                    pickle.dump(loss_psnr_time, fp)

        # Update global step with one
        global_step += 1

    # When done, cleanup multi-GPU process
    if args.multi_gpu:
        cleanup_dataparallel()


if __name__=='__main__':
    world_size = torch.cuda.device_count()
    print('> Number of availble GPUs:', world_size)

    if world_size > 1:
        mp.spawn(train, args=[world_size], nprocs=world_size)
        print('>> Starting multi-GPU process...')
    else:
        print('>> Starting single-GPU process...')
        train(0, world_size)