import nvdiffrast.torch as dr
import torch

from meshopt_lib.core.opt import MeshOptimizer

class MeshOpt:
    
    def __init__(self, mv, proj, image_size):
        self._mv = mv # C x 4 x 4
        self._mvp = proj @ mv # C x 4 x 4
        self._image_size = image_size
        self._glctx = dr.RasterizeCudaContext()

    def transform_pos(self, mtx, pos):
        posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], axis=1)
        return torch.matmul(posw, mtx.transpose(-2, -1))

    def render_silhouette(
        self, 
        vertices,
        faces
    ):
        faces = faces.type(torch.int32)

        # transform vertices to clip space
        # - vertices_clip: num_cameras x num_verts x 4
        vertices_clip = self.transform_pos(self._mvp, vertices) 

        # rasterization
        rast_out,_ = dr.rasterize(self._glctx, vertices_clip, faces, resolution=self._image_size)

        # vertex attributes = all ones
        shape = (1, vertices.size(0), vertices.size(1)+1)
        verts_attr    = torch.ones(size=shape, dtype=torch.float32, device=vertices.device)
        silhouette, _ = dr.interpolate(verts_attr, rast_out, faces)

        alpha = torch.clamp(rast_out[..., -1:], max=1) #C,H,W,1
        silhouette = torch.concat((silhouette, alpha), dim=-1) #C,H,W,4

        # antialiasing
        # - normal: num_cameras x h x w x 4
        silhouette = dr.antialias(silhouette, rast_out, vertices_clip, faces) #C,H,W,4
        silhouette = silhouette[...,0].contiguous()

        return silhouette

    def render_depth(
        self,
        vertices,
        faces,
        normalize=False
    ):
        faces = faces.type(torch.int32)

        # transform vertices to clip space
        # - vertices_clip: num_cameras x num_verts x 4
        vertices_clip = self.transform_pos(self._mvp, vertices) 
        
        # rasterization
        rast_out,_ = dr.rasterize(
            self._glctx, 
            vertices_clip, 
            faces, 
            resolution=self._image_size
        )

        # convert vertex positions from world frame to camera frame
        # - verts_camera: num_cameras x num_verts x 4
        verts_camera = self.transform_pos(self._mv, vertices)

        # vertex attribute buffer
        # - verts_attr: num_cameras x num_verts x 3
        shape = (verts_camera.size(0), verts_camera.size(1), 3)
        verts_attr = torch.zeros(size=shape, dtype=torch.float32, device='cuda')

        # duplicate z coordinates
        verts_attr[..., 0] = verts_camera[..., 2]
        verts_attr[..., 1] = verts_camera[..., 2]
        verts_attr[..., 2] = verts_camera[..., 2]

        # interpolate depth
        # - background: 0
        # - far: smaller (negative)
        # - near: bigger (negative)
        # - shape: num_cameras x h x w x 3
        depth, _ = dr.interpolate(verts_attr, rast_out, faces)

        # inverse the depth
        # - background: 0
        # - far: smaller (positive)
        # - near: bigger (positive)
        # - shape: num_cameras x h x w x 3
        depth = torch.where(rast_out[..., -1:] != 0, -1.0 / depth, 0.0)

        # antialiasing
        # - depth: num_cameras x h x w x 4
        depth = dr.antialias(depth, rast_out, vertices_clip, faces)

        # - depth: num_cameras x h x w
        depth = depth[...,0].contiguous()

        # normalizing
        if normalize:
            depth = depth / torch.max(depth)

        return depth

    def render_normal(
        self,
        vertices,
        vert_normals,
        faces,
        model
    ):
        faces = faces.type(torch.int32)

        # transform vertices to clip space
        # - vertices_clip: num_cameras x num_verts x 4
        vertices_clip = self.transform_pos(self._mvp, vertices) 
        
        # rasterization
        rast_out,_ = dr.rasterize(
            self._glctx, 
            vertices_clip, 
            faces, 
            resolution=self._image_size
        )

        # convert vertex normals from world frame to camera frame
        # - vert_normals_camera: num_cameras x num_verts x 3
        if model == 'Wonder3D_renderer':
            vert_normals_camera = self.transform_pos(self._mv.inverse().transpose(-2, -1), vert_normals)[..., :3].contiguous()

        # Wonder3D
        elif model == 'Wonder3D_output':
            vert_normals_camera = vert_normals.clone()

        # ControlNet
        elif model == 'ControlNet':
            vert_normals_camera = vert_normals.clone()
            vert_normals_camera[..., 0] *= -1.0

        # OmniData
        #vert_normals_camera[..., 1] *= -1.0
        #vert_normals_camera[..., 2] *= -1.0

        # interpolate vertex normals
        # - pixel_normals: num_cameras x h x w x 3
        pixel_normals,_ = dr.interpolate(vert_normals_camera, rast_out, faces)

        # get the silhouette
        # - alpha: num_cameras x h x w x 1
        alpha = torch.clamp(rast_out[..., -1:], max=1)

        # concatenate the normals and the silhouette
        # - output: num_cameras x h x w x 4
        output = torch.concat((pixel_normals, alpha), dim=-1)

        # antialiasing
        # - normal: num_cameras x h x w x 4
        normal = dr.antialias(output, rast_out, vertices_clip, faces)

        # - normal: num_cameras x 3 x h x w
        normal = normal[...,:3].transpose(3,2).transpose(2,1).contiguous()

        return normal
