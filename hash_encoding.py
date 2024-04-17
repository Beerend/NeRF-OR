import torch
import torch.nn as nn
from utils import get_voxel_vertices, get_voxel_vertices_4d


class DoubleHashEmbedder(nn.Module):
    # Hash encoding for spatiotemporal data, with a 3D and 4D component

    def __init__(self, bounding_box, n_levels=16, n_features_per_level=2, log2_hashmap_size_3D=15, log2_hashmap_size_4D=24, base_resolution=16,
                 finest_resolution=512, base_resolution_time=2, finest_resolution_time=32, modus='add', device='cuda'):
        super(DoubleHashEmbedder, self).__init__()

        # Set parameters
        self.bounding_box = bounding_box
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size_3D = log2_hashmap_size_3D
        self.log2_hashmap_size_4D = log2_hashmap_size_4D
        self.base_resolution = torch.tensor(base_resolution, device=device)
        self.finest_resolution = torch.tensor(finest_resolution, device=device)
        self.base_resolution_time = torch.tensor(base_resolution_time, device=device)
        self.finest_resolution_time = torch.tensor(finest_resolution_time, device=device)
        self.modus = modus

        # Initiate 3D and 4D hash encoders
        bounding_box_3d = (bounding_box[0][:3], bounding_box[1][:3])
        self.embedders = nn.ModuleList([
            HashEmbedder4D(bounding_box, n_levels, n_features_per_level, log2_hashmap_size_4D, base_resolution,
                           finest_resolution, base_resolution_time, finest_resolution_time, device=device),
            HashEmbedder(bounding_box_3d, n_levels, n_features_per_level, log2_hashmap_size_3D, base_resolution,
                         finest_resolution, device=device)
        ])

        # Define function for combining 3D and 4D hash encoding outputs
        if self.modus == 'concat':
            self.out_dim = self.embedders[0].out_dim + self.embedders[1].out_dim
        elif self.modus == 'add':
            self.out_dim = self.embedders[0].out_dim
        else:
            print(f'Error, uknown modus for adding 3D and 4D hash encodings: {self.modus}')
            assert False

    def forward(self, xyzt):
        # (x, y, z, t) is 4D point position: [B, 4]
        embedded_4d, keep_mask_4d = self.embedders[0](xyzt)
        embedded_3d, keep_mask_3d = self.embedders[1](xyzt[:, :3])
        if self.modus == 'concat':
            embedded = torch.cat([embedded_4d, embedded_3d], dim=-1)
        elif self.modus == 'add':
            embedded = embedded_4d + embedded_3d
        keep_mask = keep_mask_4d * keep_mask_3d
        return embedded, keep_mask


class HashEmbedder4D(nn.Module):
    # Hash encoding for spatiotemporal data

    def __init__(self, bounding_box, n_levels=16, n_features_per_level=2, log2_hashmap_size=19, base_resolution=16,
                 finest_resolution=512, base_resolution_time=2, finest_resolution_time=32, device='cuda'):
        super(HashEmbedder4D, self).__init__()

        # Set parameters
        self.bounding_box = (bounding_box[0].to(device), bounding_box[1].to(device))
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = torch.tensor(base_resolution, device=device)
        self.finest_resolution = torch.tensor(finest_resolution, device=device)
        self.base_resolution_time = torch.tensor(base_resolution_time, device=device)
        self.finest_resolution_time = torch.tensor(finest_resolution_time, device=device)
        self.device = device

        # Calculate dimensions
        self.out_dim = self.n_levels * self.n_features_per_level
        self.b = torch.exp((torch.log(self.finest_resolution) - torch.log(self.base_resolution)) / (n_levels-1))
        self.b_time = torch.exp((torch.log(self.finest_resolution_time) - torch.log(self.base_resolution_time)) / (n_levels-1))

        # Create embedding module and initialize weights
        self.embeddings = nn.ModuleList([
            nn.Embedding(2**self.log2_hashmap_size, self.n_features_per_level) for i in range(n_levels)
        ])
        for i in range(n_levels):
            nn.init.uniform_(self.embeddings[i].weight, a=-0.0001, b=0.0001)

    def quadrilinear_interp(self, x, voxel_min_vertex, voxel_max_vertex, voxel_embedds):
        '''
        x: [B, 4]
        voxel_min_vertex: [B, 4]
        voxel_max_vertex: [B, 4]
        voxel_embedds: [B, 16, 2]
        out: [B, 2]
        '''
        # source: https://en.wikipedia.org/wiki/Trilinear_interpolation
        weights = (x - voxel_min_vertex) / (voxel_max_vertex - voxel_min_vertex)  # [B, 4]
        
        # step 0
        # 0->0000, 1->0001, 2->0010, 3->0011, 4->0100, 5->0101, 6->0110, 7->0111,
        # 8->1000, 9->1001, 10->1010, 11->1011, 12->1100, 13->1101, 14->1110, 15->1111
        c000 = voxel_embedds[:,0]*(1-weights[:,0][:,None]) + voxel_embedds[:,8]*weights[:,0][:,None]
        c001 = voxel_embedds[:,1]*(1-weights[:,0][:,None]) + voxel_embedds[:,9]*weights[:,0][:,None]
        c010 = voxel_embedds[:,2]*(1-weights[:,0][:,None]) + voxel_embedds[:,10]*weights[:,0][:,None]
        c011 = voxel_embedds[:,3]*(1-weights[:,0][:,None]) + voxel_embedds[:,11]*weights[:,0][:,None]
        c100 = voxel_embedds[:,4]*(1-weights[:,0][:,None]) + voxel_embedds[:,12]*weights[:,0][:,None]
        c101 = voxel_embedds[:,5]*(1-weights[:,0][:,None]) + voxel_embedds[:,13]*weights[:,0][:,None]
        c110 = voxel_embedds[:,6]*(1-weights[:,0][:,None]) + voxel_embedds[:,14]*weights[:,0][:,None]
        c111 = voxel_embedds[:,7]*(1-weights[:,0][:,None]) + voxel_embedds[:,15]*weights[:,0][:,None]

        # step 1
        c00 = c000*(1-weights[:,1][:,None]) + c100*weights[:,1][:,None]
        c01 = c001*(1-weights[:,1][:,None]) + c101*weights[:,1][:,None]
        c10 = c010*(1-weights[:,1][:,None]) + c110*weights[:,1][:,None]
        c11 = c011*(1-weights[:,1][:,None]) + c111*weights[:,1][:,None]

        # step 2
        c0 = c00*(1-weights[:,2][:,None]) + c10*weights[:,2][:,None]
        c1 = c01*(1-weights[:,2][:,None]) + c11*weights[:,2][:,None]

        # step 3
        c = c0*(1-weights[:,3][:,None]) + c1*weights[:,3][:,None]

        return c

    def forward(self, x):
        # (x, t) is 4D point position: [B, 4]
        x_embedded_all = []
        for i in range(self.n_levels):
            resolution = torch.floor(self.base_resolution * self.b**i)
            resolution_time = torch.floor(self.base_resolution_time * self.b_time**i)
            voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices, keep_mask = get_voxel_vertices_4d(x,
                self.bounding_box, resolution, resolution_time, self.log2_hashmap_size, device=self.device)
            voxel_embedds = self.embeddings[i](hashed_voxel_indices)  # [B, 16, 2]
            x_embedded = self.quadrilinear_interp(x, voxel_min_vertex, voxel_max_vertex, voxel_embedds)
            x_embedded_all.append(x_embedded)

        keep_mask = keep_mask.sum(dim=-1)==keep_mask.shape[-1]
        return torch.cat(x_embedded_all, dim=-1), keep_mask


class HashEmbedder(nn.Module):
    # Hash encoding for spatial data

    def __init__(self, bounding_box, n_levels=16, n_features_per_level=2,\
                log2_hashmap_size=19, base_resolution=16, finest_resolution=512, device='cuda'):
        super(HashEmbedder, self).__init__()

        # Initiate parameters
        self.bounding_box = (bounding_box[0].to(device), bounding_box[1].to(device))
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = torch.tensor(base_resolution, device=device)
        self.finest_resolution = torch.tensor(finest_resolution, device=device)
        self.device = device

        # Calculate dimensions
        self.out_dim = self.n_levels * self.n_features_per_level
        self.b = torch.exp((torch.log(self.finest_resolution)-torch.log(self.base_resolution))/(n_levels-1))

        # Create embedding module and initialize weights
        self.embeddings = nn.ModuleList([nn.Embedding(2**self.log2_hashmap_size, \
                                        self.n_features_per_level) for i in range(n_levels)])
        for i in range(n_levels):
            nn.init.uniform_(self.embeddings[i].weight, a=-0.0001, b=0.0001)        

    def trilinear_interp(self, x, voxel_min_vertex, voxel_max_vertex, voxel_embedds):
        '''
        x: B x 3
        voxel_min_vertex: B x 3
        voxel_max_vertex: B x 3
        voxel_embedds: B x 8 x 2
        '''
        # source: https://en.wikipedia.org/wiki/Trilinear_interpolation
        weights = (x - voxel_min_vertex)/(voxel_max_vertex-voxel_min_vertex) # B x 3

        # step 1
        # 0->000, 1->001, 2->010, 3->011, 4->100, 5->101, 6->110, 7->111
        c00 = voxel_embedds[:,0]*(1-weights[:,0][:,None]) + voxel_embedds[:,4]*weights[:,0][:,None]
        c01 = voxel_embedds[:,1]*(1-weights[:,0][:,None]) + voxel_embedds[:,5]*weights[:,0][:,None]
        c10 = voxel_embedds[:,2]*(1-weights[:,0][:,None]) + voxel_embedds[:,6]*weights[:,0][:,None]
        c11 = voxel_embedds[:,3]*(1-weights[:,0][:,None]) + voxel_embedds[:,7]*weights[:,0][:,None]

        # step 2
        c0 = c00*(1-weights[:,1][:,None]) + c10*weights[:,1][:,None]
        c1 = c01*(1-weights[:,1][:,None]) + c11*weights[:,1][:,None]

        # step 3
        c = c0*(1-weights[:,2][:,None]) + c1*weights[:,2][:,None]

        return c

    def forward(self, x):
        # x is 3D point position: B x 3
        x_embedded_all = []
        for i in range(self.n_levels):
            resolution = torch.floor(self.base_resolution * self.b**i)
            voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices, keep_mask = get_voxel_vertices(\
                                                x, self.bounding_box, \
                                                resolution, self.log2_hashmap_size, device=self.device)
            
            voxel_embedds = self.embeddings[i](hashed_voxel_indices)

            x_embedded = self.trilinear_interp(x, voxel_min_vertex, voxel_max_vertex, voxel_embedds)
            x_embedded_all.append(x_embedded)

        keep_mask = keep_mask.sum(dim=-1)==keep_mask.shape[-1]
        return torch.cat(x_embedded_all, dim=-1), keep_mask


class SHEncoder(nn.Module):
    # Spherical Harmonics encoder

    def __init__(self, input_dim=3, degree=4):
        super().__init__()
        self.input_dim = input_dim
        self.degree = degree
        assert self.input_dim == 3
        assert self.degree >= 1 and self.degree <= 5
        self.out_dim = degree ** 2

        self.C0 = 0.28209479177387814
        self.C1 = 0.4886025119029199
        self.C2 = [
            1.0925484305920792,
            -1.0925484305920792,
            0.31539156525252005,
            -1.0925484305920792,
            0.5462742152960396
        ]
        self.C3 = [
            -0.5900435899266435,
            2.890611442640554,
            -0.4570457994644658,
            0.3731763325901154,
            -0.4570457994644658,
            1.445305721320277,
            -0.5900435899266435
        ]
        self.C4 = [
            2.5033429417967046,
            -1.7701307697799304,
            0.9461746957575601,
            -0.6690465435572892,
            0.10578554691520431,
            -0.6690465435572892,
            0.47308734787878004,
            -1.7701307697799304,
            0.6258357354491761
        ]

    def forward(self, input, **kwargs):
        result = torch.empty((*input.shape[:-1], self.out_dim), dtype=input.dtype, device=input.device)
        x, y, z = input.unbind(-1)

        result[..., 0] = self.C0
        if self.degree > 1:
            result[..., 1] = -self.C1 * y
            result[..., 2] = self.C1 * z
            result[..., 3] = -self.C1 * x
            if self.degree > 2:
                xx, yy, zz = x * x, y * y, z * z
                xy, yz, xz = x * y, y * z, x * z
                result[..., 4] = self.C2[0] * xy
                result[..., 5] = self.C2[1] * yz
                result[..., 6] = self.C2[2] * (2.0 * zz - xx - yy)
                #result[..., 6] = self.C2[2] * (3.0 * zz - 1) # xx + yy + zz == 1, but this will lead to different backward gradients, interesting...
                result[..., 7] = self.C2[3] * xz
                result[..., 8] = self.C2[4] * (xx - yy)
                if self.degree > 3:
                    result[..., 9] = self.C3[0] * y * (3 * xx - yy)
                    result[..., 10] = self.C3[1] * xy * z
                    result[..., 11] = self.C3[2] * y * (4 * zz - xx - yy)
                    result[..., 12] = self.C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
                    result[..., 13] = self.C3[4] * x * (4 * zz - xx - yy)
                    result[..., 14] = self.C3[5] * z * (xx - yy)
                    result[..., 15] = self.C3[6] * x * (xx - 3 * yy)
                    if self.degree > 4:
                        result[..., 16] = self.C4[0] * xy * (xx - yy)
                        result[..., 17] = self.C4[1] * yz * (3 * xx - yy)
                        result[..., 18] = self.C4[2] * xy * (7 * zz - 1)
                        result[..., 19] = self.C4[3] * yz * (7 * zz - 3)
                        result[..., 20] = self.C4[4] * (zz * (35 * zz - 30) + 3)
                        result[..., 21] = self.C4[5] * xz * (7 * zz - 3)
                        result[..., 22] = self.C4[6] * (xx - yy) * (7 * zz - 1)
                        result[..., 23] = self.C4[7] * xz * (xx - 3 * yy)
                        result[..., 24] = self.C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))

        return result
