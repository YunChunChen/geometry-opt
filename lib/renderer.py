import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr

class Renderer:

    def __init__(self, cfg):
        self.resolution = cfg.renderer.resolution
        self.glctx = self.get_rasterizor()

        # camera intrinsic parameters
        self.fl = torch.tensor([0.15], dtype=torch.float32, device='cuda', requires_grad=True)
        self.n  = torch.tensor([ 0.1], dtype=torch.float32, device='cuda', requires_grad=False)
        self.f  = torch.tensor([10.0], dtype=torch.float32, device='cuda', requires_grad=False)

        # camera extrinsic parameters
        self.x = torch.tensor([ 0.0], dtype=torch.float32, device='cuda', requires_grad=True)
        self.y = torch.tensor([ 0.0], dtype=torch.float32, device='cuda', requires_grad=True)
        self.z = torch.tensor([-3.5], dtype=torch.float32, device='cuda', requires_grad=True)

    def transform_pos(self, mtx, pos):
        posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], axis=1)
        return torch.matmul(posw, mtx.t())[None, ...]

    def get_camera_intrinsics(self):
        return self.fl, self.n, self.f

    def get_camera_extrinsics(self):
        return self.x, self.y, self.z

    def init_proj(self):
        zero = torch.tensor([0.0], dtype=torch.float32, device='cuda')
        one  = torch.tensor([1.0], dtype=torch.float32, device='cuda')

        proj_mat = torch.stack((
            1.0/self.fl, zero, zero, zero,
            zero, 1.0/self.fl, zero, zero,
            zero, zero, -(self.f+self.n)/(self.f-self.n), -(2*self.f*self.n)/(self.f-self.n),
            zero, zero, -1*one, zero
        )).view(4,4).cuda()

        return proj_mat

    def init_mv(self):
        zero = torch.tensor([0.0], dtype=torch.float32, device='cuda')
        one  = torch.tensor([1.0], dtype=torch.float32, device='cuda')

        mv_mat = torch.stack((
             one, zero, zero, self.x,
            zero,  one, zero, self.y,
            zero, zero,  one, self.z,
            zero, zero, zero,    one
        )).view(4,4).cuda()

        return mv_mat

    def get_rasterizor(self):
        glctx = dr.RasterizeCudaContext()
        return glctx

    def render_depth(self, mesh, normalize=False):
        verts = mesh.get_vertices().clone()
        faces = mesh.get_faces().clone()

        # initialize the projection and model view matrices
        proj_mat = self.init_proj()
        mv_mat   = self.init_mv()

        # compute the mvp matrix
        mvp_mat = torch.matmul(proj_mat, mv_mat)

        # transform vertices to clip space
        verts_clip = self.transform_pos(mvp_mat, verts)

        # rasterization
        rast, _ = dr.rasterize(
            self.glctx, 
            verts_clip, 
            faces, 
            resolution=[self.resolution, self.resolution]
        )

        # vertex coordinate transformation
        verts_camera = self.transform_pos(mv_mat, verts)

        # vertex attribute buffer
        verts_attr = torch.zeros(size=verts.size(), dtype=torch.float32, device='cuda').unsqueeze(0)
        
        # duplicate z coordinates
        verts_attr[:,:,0] = verts_camera[:,:,2]
        verts_attr[:,:,1] = verts_camera[:,:,2]
        verts_attr[:,:,2] = verts_camera[:,:,2]

        # interpolate depth
        # - background: 0
        # - far: smaller (negative)
        # - near: bigger (negative)
        # - shape: 1 x h x w x 3
        depth, _ = dr.interpolate(verts_attr, rast, faces)

        # inverse the depth
        # - background: 0
        # - far: smaller (positive)
        # - near: bigger (positive)
        # - shape: 1 x h x w x 3
        depth = torch.where(rast[..., -1:] != 0, -1.0 / depth, 0.0)

        # antialias 
        # - shape: h x w
        depth = dr.antialias(depth, rast, verts_clip, faces)[0,:,:,0]

        # flip the depth (memory order in OpenGL)
        depth = torch.flip(depth, [0])

        # normalize depth
        if normalize:
            depth = depth / torch.max(depth)

        return depth
    
    def render_silhouette(self, mesh):
        verts = mesh.get_vertices().clone()
        faces = mesh.get_faces().clone()

        # initialize the projection and model view matrices
        proj_mat = self.init_proj()
        mv_mat   = self.init_mv()

        # compute the mvp matrix
        mvp_mat = torch.matmul(proj_mat, mv_mat)

        # transform vertices to clip space
        verts_clip = self.transform_pos(mvp_mat, verts)

        # rasterization
        rast, _ = dr.rasterize(
            self.glctx, 
            verts_clip, 
            faces, 
            resolution=[self.resolution, self.resolution]
        )
        # vertex attributes = all ones
        shape = (1, verts.size(0), verts.size(1)+1)
        verts_attr    = torch.ones(size=shape, dtype=torch.float32, device=verts.device)
        silhouette, _ = dr.interpolate(verts_attr, rast, faces)

        # antialias
        # - shape: h x w
        silhouette = dr.antialias(silhouette, rast, verts_attr, faces)[0,:,:,0]

        # flip the silhouette
        silhouette = torch.flip(silhouette, [0])

        return silhouette

    def render_normal(self, mesh, antialias=False):
        verts = mesh.get_vertices().clone()
        faces = mesh.get_faces().clone()
        vert_normals = mesh.get_vertex_normals().clone()

        # initialize the projection and model view matrices
        proj_mat = self.init_proj()
        mv_mat   = self.init_mv()

        # compute the mvp matrix
        mvp_mat = torch.matmul(proj_mat, mv_mat)

        # transform vertices to clip space
        verts_clip = self.transform_pos(mvp_mat, verts)

        # rasterization
        rast, _ = dr.rasterize(
            self.glctx, 
            verts_clip, 
            faces, 
            resolution=[self.resolution, self.resolution]
        )

        # vertex normals transformation world frame -> camera frame
        # - shape: 1 x num_verts x 3
        vert_normals_camera = self.transform_pos(mv_mat.inverse().t(), vert_normals)[:,:,:3].contiguous()
        #vert_normals_camera = self.transform_pos(mv_mat, vert_normals)[:,:,:3].contiguous()
        #vert_normals_camera = vert_normals.unsqueeze(0) # -> inner product is correct
        vert_normals_camera[:,:,0] *= -1.0
        #vert_normals_camera[:,:,1] *= -1.0
        #vert_normals_camera[:,:,2] *= -1.0

        # interpolate vertex normals
        # - shape: 1 x h x w x 3
        pixel_normals, _ = dr.interpolate(vert_normals_camera, rast, faces)

        # fill in zeros
        # - background: [0, 0, 0]
        # - shape: 1 x h x w x 3
        zero_tensor = torch.zeros(3, dtype=torch.float32, device='cuda')
        #zero_tensor = torch.tensor([0.0, 1.0, 1.0], dtype=torch.float32, device='cuda')
        pixel_normals = torch.where(rast[..., -1:] != 0, pixel_normals, zero_tensor)

        # antialias 
        # - shape: h x w x 3
        if antialias:
            pixel_normals = dr.antialias(pixel_normals, rast, verts_clip, faces)[0,:,:,:]
        else:
            pixel_normals = pixel_normals[0,:,:,:]

        # flip the depth (memory order in OpenGL)
        pixel_normals = torch.flip(pixel_normals, [0])

        # transpose axes
        # - shape: h x w x 3 -> 3 x h x w
        pixel_normals = pixel_normals.transpose(2,1).transpose(1,0).contiguous()

        # normalize the normal length
        # - background: 0
        # - foreground: 1
        pixel_normals = F.normalize(pixel_normals, p=2, dim=0)

        return pixel_normals
    
    def depth_to_pointcloud(self, depth):
        # initialize the projection and model view matrices
        proj_mat = self.init_proj()
        mv_mat   = self.init_mv()

        # compute the mvp matrix
        mvp_mat = torch.matmul(proj_mat, mv_mat)
        
        # get mask
        y, x = torch.where(depth != 0.0)

        h, w = depth.size()

        # get depth
        z = depth[y, x]

        x = (x - w/2.0) / w
        y = (y - h/2.0) / h

        # point cloud camera frame
        ones = torch.ones(x.shape, dtype=torch.float32, device='cuda')
        pc = torch.stack((x, y, z, ones), dim=-1)

        # point cloud world frame
        pc = torch.matmul(pc, mv_mat.inverse().t())

        return pc
