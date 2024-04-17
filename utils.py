import torch


def hash(coords, log2_hashmap_size):
    '''
    coords: this function can process upto 7 dim coordinates
    log2T:  logarithm of T w.r.t 2
    '''
    primes = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]

    xor_result = torch.zeros_like(coords)[..., 0]
    for i in range(coords.shape[-1]):
        xor_result ^= coords[..., i]*primes[i]

    return torch.tensor((1<<log2_hashmap_size)-1).to(xor_result.device) & xor_result


def get_voxel_vertices_4d(xyzt, bounding_box, resolution, resolution_time, log2_hashmap_size, device='cuda'):
    '''
    xyzt: 4D coordinates of samples, with space coordinates xyz and time t. Bx4
    bounding_box: min and max xyzt coordinates of box
    resolution: number of voxels per axis
    resolution_time: number of voxels per axis over the time dimension
    '''
    box_min, box_max = bounding_box  # ([4], [4])

    keep_mask = xyzt==torch.max(torch.min(xyzt, box_max), box_min)
    if not torch.all(xyzt <= box_max) or not torch.all(xyzt >= box_min):
        xyzt = torch.clamp(xyzt, min=box_min, max=box_max)

    resolutions = torch.tensor([resolution, resolution, resolution, resolution_time], device=device)
    grid_size = (box_max - box_min) / resolutions  # [4]
    
    bottom_left_start_idx = torch.floor((xyzt - box_min) / grid_size).int()  # [B, 4]
    voxel_min_vertex = bottom_left_start_idx * grid_size + box_min  # [B, 4]
    voxel_max_vertex = voxel_min_vertex + torch.tensor([1.0, 1.0, 1.0, 1.0], device=device) * grid_size  # [B, 4]

    box_offsets_4d = torch.tensor([[[i,j,k,l] for i in [0, 1] for j in [0, 1] for k in [0, 1] for l in [0, 1]]], device=device)  # [1, 16, 3]
    voxel_indices = bottom_left_start_idx.unsqueeze(1) + box_offsets_4d  # [B, 16, 4]
    hashed_voxel_indices = hash(voxel_indices, log2_hashmap_size)  # [B, 16]
    
    return voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices, keep_mask


def get_voxel_vertices(xyz, bounding_box, resolution, log2_hashmap_size, device='cuda'):
    '''
    xyz: 3D coordinates of samples. B x 3
    bounding_box: min and max x,y,z coordinates of object bbox
    resolution: number of voxels per axis
    '''
    box_min, box_max = bounding_box

    keep_mask = xyz==torch.max(torch.min(xyz, box_max), box_min)
    if not torch.all(xyz <= box_max) or not torch.all(xyz >= box_min):
        xyz = torch.clamp(xyz, min=box_min, max=box_max)

    grid_size = (box_max-box_min)/resolution
    
    bottom_left_idx = torch.floor((xyz-box_min)/grid_size).int()
    voxel_min_vertex = bottom_left_idx*grid_size + box_min
    voxel_max_vertex = voxel_min_vertex + torch.tensor([1.0, 1.0, 1.0], device=device) * grid_size

    box_offsets = torch.tensor([[[i,j,k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]], device=device)  # [1, 8, 3]
    voxel_indices = bottom_left_idx.unsqueeze(1) + box_offsets
    hashed_voxel_indices = hash(voxel_indices, log2_hashmap_size)

    return voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices, keep_mask
