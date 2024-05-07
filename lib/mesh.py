import igl
import trimesh

from meshopt_lib.util.func import load_obj, normalize_vertices
from meshopt_lib.core.calc_vertex_normals import calc_vertex_normals

class Mesh:

    def __init__(self, mesh_path, mesh_scale):
        self.verts, self.faces = load_obj(mesh_path)
        self.verts = normalize_vertices(self.verts)
        self.scale_mesh(mesh_scale)

    def scale_mesh(self, mesh_scale):
        self.verts = self.verts * mesh_scale

    def get_vertices(self):
        return self.verts

    def get_faces(self):
        return self.faces

    def get_vertex_normals(self):
        return calc_vertex_normals(self.verts, self.faces)

    def update_vertices(self, verts):
        self.verts = verts

    def update_faces(self, faces):
        self.faces = faces

    def save_mesh(self, save_path):
        verts = self.verts.data.cpu().numpy()
        faces = self.faces.data.cpu().numpy()
        igl.write_triangle_mesh(save_path, verts, faces)

    def save_glb(self, save_path):
        verts = self.verts.cpu().numpy()
        faces = self.faces.cpu().numpy()
        mesh = trimesh.Trimesh(
            vertices=self.verts.data.cpu().numpy(),
            faces=self.faces.data.cpu().numpy()
        )
        s = trimesh.Scene([mesh])
        s.export(file_obj=save_path)
