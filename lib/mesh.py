import igl
import trimesh
import trimesh.transformations as tra
import numpy as np
from gpytoolbox import remesh_botsch

import torch

class Mesh:

    def __init__(self, cfg):
        verts, faces = self.load_mesh(
            mesh_path=cfg.data.mesh_path, 
            mesh_scale=cfg.data.mesh_scale,
            num_subdiv=cfg.data.num_subdiv,
        )

        self.verts = torch.tensor(verts, dtype=torch.float32, device='cuda')
        self.faces = torch.tensor(faces, dtype=torch.int32, device='cuda')

    def load_mesh(self, mesh_path, mesh_scale, num_subdiv=0):
        # read mesh
        verts, faces = igl.read_triangle_mesh(mesh_path)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        
        # recenter mesh
        mesh.vertices = mesh.vertices - mesh.centroid

        # rotate the mesh
        rot_mat = tra.euler_matrix(np.pi/4, -np.pi/4, 0)
        mesh.apply_transform(rot_mat)

        verts = mesh.vertices.copy()
        faces = mesh.faces.copy()

        # get bbox length
        bbox_length = self.get_bbox_length(verts)
        # resize mesh
        verts = verts / bbox_length
        # scale mesh
        verts = verts * mesh_scale
        # subdivide mesh
        if num_subdiv > 0:
            verts, faces = igl.upsample(verts, faces, num_subdiv)
        return verts, faces

    def recenter_mesh(self, verts):
        mesh_center = np.mean(verts, axis=0)
        verts = verts - mesh_center
        return verts

    def get_bbox_length(self, verts):
        bbox_length = np.sum((np.max(verts, axis=0) - np.min(verts, axis=0)) ** 2.0) ** 0.5
        return bbox_length

    def get_face_normals(self):
        faces = torch.transpose(self.faces, 0, 1).long()
        verts = torch.transpose(self.verts, 0, 1)

        v = [
            verts.index_select(1, faces[0]),
            verts.index_select(1, faces[1]),
            verts.index_select(1, faces[2])
        ]

        c = torch.cross(v[1] - v[0], v[2] - v[0])
        face_normals = c / torch.norm(c, dim=0)
        return face_normals

    def safe_acos(self, x):
        return torch.acos(x.clamp(min=-1, max=1))

    def get_vertex_normals(self):
        faces = torch.transpose(self.faces, 0, 1).long()
        verts = torch.transpose(self.verts, 0, 1)

        v = [
            verts.index_select(1, faces[0]),
            verts.index_select(1, faces[1]),
            verts.index_select(1, faces[2])
        ]

        # get face normals
        c = torch.cross(v[1] - v[0], v[2] - v[0])
        face_normals = c / torch.norm(c, dim=0)

        vert_normals = torch.zeros_like(verts)
        for i in range(3):
            d0 = v[(i + 1) % 3] - v[i]
            d0 = d0 / torch.norm(d0)
            d1 = v[(i + 2) % 3] - v[i]
            d1 = d1 / torch.norm(d1)
            d = torch.sum(d0*d1, 0)
            face_angle = self.safe_acos(torch.sum(d0*d1, 0))
            nn =  face_normals * face_angle
            for j in range(3):
                vert_normals[j].index_add_(0, faces[i], nn[j])
        vert_normals = (vert_normals / torch.norm(vert_normals, dim=0)).transpose(0, 1)
        return vert_normals

    def get_vertices(self):
        return self.verts

    def get_faces(self):
        return self.faces

    def update_vertices(self, verts):
        self.verts = verts
        return

    def write_mesh(self, save_path):
        verts = self.verts.cpu().numpy()
        faces = self.faces.cpu().numpy()
        igl.write_triangle_mesh(save_path, verts, faces)
        return

    def compute_average_edge_length(self):
        face_verts = self.verts[self.faces.long()]
        v0 = face_verts[:, 0]
        v1 = face_verts[:, 1]
        v2 = face_verts[:, 2]
        A = (v1 - v2).norm(dim=1)
        B = (v0 - v2).norm(dim=1)
        C = (v0 - v1).norm(dim=1)
        avg_edge_length = (A + B + C).sum() / self.faces.shape[0] / 3
        return avg_edge_length

    def remove_duplicates(self):
        unique_verts, inverse = torch.unique(self.verts, dim=0, return_inverse=True)
        new_faces = inverse[self.faces.long()]
        self.verts = unique_verts.to(self.verts.dtype).contiguous()
        self.faces = new_faces.to(self.faces.dtype).contiguous()
        return

    def remesh(self):
        # get average edge length
        avg_edge_length = self.compute_average_edge_length().detach().cpu().numpy()

        # move data to cpu
        verts_cpu = self.verts.detach().cpu().numpy().astype(np.double)
        faces_cpu = self.faces.detach().cpu().numpy().astype(np.int32)

        # remesh
        remeshed_verts, remeshed_faces = remesh_botsch(
            verts_cpu, 
            faces_cpu, 
            5, 
            avg_edge_length * 0.8, 
            True
        )

        # overwrite self.verts and self.faces
        self.verts = torch.tensor(remeshed_verts, dtype=torch.float32, device='cuda')
        self.faces = torch.tensor(remeshed_faces, dtype=torch.int32, device='cuda')

        # clean mesh
        self.remove_duplicates()

        return
