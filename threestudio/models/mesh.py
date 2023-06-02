from __future__ import annotations

import numpy as np
import torch
import torchvision
import torch.nn.functional as F

import threestudio
from threestudio.utils.ops import dot
from threestudio.utils.typing import *

class Mesh:
    def __init__(
        self,
        v_pos: Float[Tensor, "Nv 3"],
        t_pos_idx: Integer[Tensor, "Nf 3"],
        v_nrm = None,
        t_nrm_idx = None,
        v_tex = None,
        t_tex_idx = None,

        material=None,
        **kwargs,
    ) -> None:
        self.v_pos: Float[Tensor, "Nv 3"] = v_pos
        self.t_pos_idx: Integer[Tensor, "Nf 3"] = t_pos_idx
        self._v_nrm: Optional[Float[Tensor, "Nv 3"]] = v_nrm
        self._t_nrm_idx = t_nrm_idx
        self._v_tng: Optional[Float[Tensor, "Nv 3"]] = None
        self._v_tex: Optional[Float[Tensor, "Nt 3"]] = v_tex
        self._t_tex_idx: Optional[Float[Tensor, "Nf 3"]] = t_tex_idx
        self._v_rgb: Optional[Float[Tensor, "Nv 3"]] = None
        self._edges: Optional[Integer[Tensor, "Ne 2"]] = None
        self.material = material
        self.extras: Dict[str, Any] = {}

        for k, v in kwargs.items(): self.add_extra(k, v)
    def load_from_file(path): return load_obj(path)

    def add_extra(self, k, v) -> None:
        self.extras[k] = v

    def remove_outlier(self, outlier_n_faces_threshold: Union[int, float]) -> Mesh:
        if self.requires_grad:
            threestudio.debug("Mesh is differentiable, not removing outliers")
            return self

        # use trimesh to first split the mesh into connected components
        # then remove the components with less than n_face_threshold faces
        import trimesh

        # construct a trimesh object
        mesh = trimesh.Trimesh(
            vertices=self.v_pos.detach().cpu().numpy(),
            faces=self.t_pos_idx.detach().cpu().numpy(),
        )

        # split the mesh into connected components
        components = mesh.split(only_watertight=False)
        # log the number of faces in each component
        threestudio.debug(
            "Mesh has {} components, with faces: {}".format(
                len(components), [c.faces.shape[0] for c in components]
            )
        )

        n_faces_threshold: int
        if isinstance(outlier_n_faces_threshold, float):
            # set the threshold to the number of faces in the largest component multiplied by outlier_n_faces_threshold
            n_faces_threshold = int(
                max([c.faces.shape[0] for c in components]) * outlier_n_faces_threshold
            )
        else:
            # set the threshold directly to outlier_n_faces_threshold
            n_faces_threshold = outlier_n_faces_threshold

        # log the threshold
        threestudio.debug(
            "Removing components with less than {} faces".format(n_faces_threshold)
        )

        # remove the components with less than n_face_threshold faces
        components = [c for c in components if c.faces.shape[0] >= n_faces_threshold]

        # log the number of faces in each component after removing outliers
        threestudio.debug(
            "Mesh has {} components after removing outliers, with faces: {}".format(
                len(components), [c.faces.shape[0] for c in components]
            )
        )
        # merge the components
        mesh = trimesh.util.concatenate(components)

        # convert back to our mesh format
        v_pos = torch.from_numpy(mesh.vertices).to(self.v_pos)
        t_pos_idx = torch.from_numpy(mesh.faces).to(self.t_pos_idx)

        clean_mesh = Mesh(v_pos, t_pos_idx)
        # keep the extras unchanged

        if len(self.extras) > 0:
            clean_mesh.extras = self.extras
            threestudio.debug(
                f"The following extra attributes are inherited from the original mesh unchanged: {list(self.extras.keys())}"
            )
        return clean_mesh

    @property
    def requires_grad(self):
        return self.v_pos.requires_grad

    @property
    def v_nrm(self):
        if self._v_nrm is None:
            self._v_nrm = self._compute_vertex_normal()
        return self._v_nrm

    @property
    def v_tng(self):
        if self._v_tng is None:
            self._v_tng = self._compute_vertex_tangent()
        return self._v_tng

    @property
    def v_tex(self):
        if self._v_tex is None:
            self._v_tex, self._t_tex_idx = self._unwrap_uv()
        return self._v_tex

    @property
    def t_tex_idx(self):
        if self._t_tex_idx is None:
            self._v_tex, self._t_tex_idx = self._unwrap_uv()
        return self._t_tex_idx

    @property
    def v_rgb(self):
        return self._v_rgb

    @property
    def edges(self):
        if self._edges is None:
            self._edges = self._compute_edges()
        return self._edges

    def _compute_vertex_normal(self):
        i0 = self.t_pos_idx[:, 0]
        i1 = self.t_pos_idx[:, 1]
        i2 = self.t_pos_idx[:, 2]

        v0 = self.v_pos[i0, :]
        v1 = self.v_pos[i1, :]
        v2 = self.v_pos[i2, :]

        face_normals = torch.cross(v1 - v0, v2 - v0)

        # Splat face normals to vertices
        v_nrm = torch.zeros_like(self.v_pos)
        v_nrm.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
        v_nrm.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
        v_nrm.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)

        # Normalize, replace zero (degenerated) normals with some default value
        v_nrm = torch.where(
            dot(v_nrm, v_nrm) > 1e-20, v_nrm, torch.as_tensor([0.0, 0.0, 1.0]).to(v_nrm)
        )
        v_nrm = F.normalize(v_nrm, dim=1)

        if torch.is_anomaly_enabled():
            assert torch.all(torch.isfinite(v_nrm))

        return v_nrm

    def _compute_vertex_tangent(self):
        vn_idx = [None] * 3
        pos = [None] * 3
        tex = [None] * 3
        for i in range(0, 3):
            pos[i] = self.v_pos[self.t_pos_idx[:, i]]
            tex[i] = self.v_tex[self.t_tex_idx[:, i]]
            vn_idx[i] = self.t_nrm_idx[:, i]

        tangents = torch.zeros_like(self.v_nrm)
        tansum = torch.zeros_like(self.v_nrm)

        # Compute tangent space for each triangle
        uve1 = tex[1] - tex[0]
        uve2 = tex[2] - tex[0]
        pe1 = pos[1] - pos[0]
        pe2 = pos[2] - pos[0]

        nom = pe1 * uve2[..., 1:2] - pe2 * uve1[..., 1:2]
        denom = uve1[..., 0:1] * uve2[..., 1:2] - uve1[..., 1:2] * uve2[..., 0:1]

        # Avoid division by zero for degenerated texture coordinates
        tang = nom / torch.where(
            denom > 0.0, torch.clamp(denom, min=1e-6), torch.clamp(denom, max=-1e-6)
        )

        # Update all 3 vertices
        for i in range(0, 3):
            idx = vn_idx[i][:, None].repeat(1, 3)
            tangents.scatter_add_(0, idx, tang)  # tangents[n_i] = tangents[n_i] + tang
            tansum.scatter_add_(
                0, idx, torch.ones_like(tang)
            )  # tansum[n_i] = tansum[n_i] + 1
        tangents = tangents / tansum

        # Normalize and make sure tangent is perpendicular to normal
        tangents = F.normalize(tangents, dim=1)
        tangents = F.normalize(tangents - dot(tangents, self.v_nrm) * self.v_nrm)

        if torch.is_anomaly_enabled():
            assert torch.all(torch.isfinite(tangents))

        return tangents

    def _unwrap_uv(
        self, xatlas_chart_options: dict = {}, xatlas_pack_options: dict = {}
    ):
        threestudio.info("Using xatlas to perform UV unwrapping, may take a while ...")

        import xatlas

        atlas = xatlas.Atlas()
        atlas.add_mesh(
            self.v_pos.cpu().numpy(),
            self.t_pos_idx.cpu().numpy(),
        )
        co = xatlas.ChartOptions()
        po = xatlas.PackOptions()
        for k, v in xatlas_chart_options.items():
            setattr(co, k, v)
        for k, v in xatlas_pack_options.items():
            setattr(po, k, v)
        atlas.generate(co, po)
        vmapping, indices, uvs = atlas.get_mesh(0)
        vmapping = (
            torch.from_numpy(
                vmapping.astype(np.uint64, casting="same_kind").view(np.int64)
            )
            .to(self.v_pos.device)
            .long()
        )
        uvs = torch.from_numpy(uvs).to(self.v_pos.device).float()
        indices = (
            torch.from_numpy(
                indices.astype(np.uint64, casting="same_kind").view(np.int64)
            )
            .to(self.v_pos.device)
            .long()
        )
        return uvs, indices

    def unwrap_uv(
        self, xatlas_chart_options: dict = {}, xatlas_pack_options: dict = {}
    ):
        self._v_tex, self._t_tex_idx = self._unwrap_uv(
            xatlas_chart_options, xatlas_pack_options
        )

    def set_vertex_color(self, v_rgb):
        assert v_rgb.shape[0] == self.v_pos.shape[0]
        self._v_rgb = v_rgb

    def _compute_edges(self):
        # Compute edges
        edges = torch.cat(
            [
                self.t_pos_idx[:, [0, 1]],
                self.t_pos_idx[:, [1, 2]],
                self.t_pos_idx[:, [2, 0]],
            ],
            dim=0,
        )
        edges = edges.sort()[0]
        edges = torch.unique(edges, dim=0)
        return edges

    def unit_mesh(self, s=1):
      def aabb(mesh):
        return torch.min(mesh.v_pos, dim=0).values, torch.max(mesh.v_pos, dim=0).values

      with torch.no_grad():
        vmin, vmax = aabb(self)
        scale = 2 / torch.max(vmax - vmin).item()
        scale = scale * s
        shift = (vmax + vmin) / 2
        v_pos = self.v_pos - shift  # Center mesh on origin
        v_pos = v_pos * scale       # Rescale to unit size

      self.v_pos = v_pos
      return self

    def normal_consistency(self) -> Float[Tensor, ""]:
        edge_nrm: Float[Tensor, "Ne 2 3"] = self.v_nrm[self.edges]
        nc = (
            1.0 - torch.cosine_similarity(edge_nrm[:, 0], edge_nrm[:, 1], dim=-1)
        ).mean()
        return nc


# yoinked from nvdiffmodeling
import os
def load_obj(filename, mtl_override=None, device="cuda", default_bsdf="diffuse"):
    print(f"Loading OBJ from {filename}")
    def _find_mat(materials, name):
      for mat in materials:
        if mat['name'] == name: return mat
      return materials[0]
    obj_path = os.path.dirname(filename)

    # Read entire file
    with open(filename) as f:
        lines = f.readlines()

    # Load materials
    all_materials = [{
        'name' : '_default_mat',
        'bsdf' : 'falcor',
        'kd'   : torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device=device),
        'ks'   : torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device),
    }]
    if mtl_override is None:
        for line in lines:
            if len(line.split()) == 0: continue
            if line.split()[0] == 'mtllib':
                # Read in entire material library
                all_materials += load_mtl(os.path.join(obj_path, line.split()[1]), device, default_bsdf)
    else:
        all_materials += load_mtl(mtl_override, device=device, default_bsdf=default_bsdf)

    # load vertices
    vertices, texcoords, normals  = [], [], []
    for line in lines:
        if len(line.split()) == 0: continue

        prefix = line.split()[0].lower()
        if prefix == 'v':
            vertices.append([float(v) for v in line.split()[1:]])
        elif prefix == 'vt':
            val = [float(v) for v in line.split()[1:]]
            texcoords.append([val[0], 1.0 - val[1]])
        elif prefix == 'vn':
            normals.append([float(v) for v in line.split()[1:]])

    # load faces
    activeMatIdx = None
    used_materials = []
    faces, tfaces, nfaces, mfaces = [], [], [], []
    for line in lines:
        if len(line.split()) == 0: continue

        prefix = line.split()[0].lower()
        if prefix == 'usemtl': # Track used materials
            mat = _find_mat(all_materials, line.split()[1])
            if not mat in used_materials:
                used_materials.append(mat)
            activeMatIdx = used_materials.index(mat)
        elif prefix == 'f': # Parse face
            vs = line.split()[1:]
            nv = len(vs)
            vv = vs[0].split('/')
            while len(vv) < 3: vv.append("")
            v0 = int(vv[0]) - 1
            t0 = int(vv[1]) - 1 if vv[1] != "" else -1
            n0 = int(vv[2]) - 1 if vv[2] != "" else -1
            for i in range(nv - 2): # Triangulate polygons
                vv = vs[i + 1].split('/')
                while len(vv) < 3: vv.append("")
                v1 = int(vv[0]) - 1
                t1 = int(vv[1]) - 1 if vv[1] != "" else -1
                n1 = int(vv[2]) - 1 if vv[2] != "" else -1
                vv = vs[i + 2].split('/')
                while len(vv) < 3: vv.append("")
                v2 = int(vv[0]) - 1
                t2 = int(vv[1]) - 1 if vv[1] != "" else -1
                n2 = int(vv[2]) - 1 if vv[2] != "" else -1
                mfaces.append(activeMatIdx)
                faces.append([v0, v1, v2])
                tfaces.append([t0, t1, t2])
                nfaces.append([n0, n1, n2])
    assert len(tfaces) == len(faces) and len(nfaces) == len (faces)

    # Create an "uber" material by combining all textures into a larger texture
    if len(used_materials) > 1:
        uber_material, texcoords, tfaces = material.merge_materials(used_materials, texcoords, tfaces, mfaces)
    elif len(used_materials) == 0: uber_material = all_materials[0]
    else: uber_material = used_materials[0]

    vertices = torch.tensor(vertices, dtype=torch.float32, device=device)
    texcoords = torch.tensor(texcoords, dtype=torch.float32, device=device) if len(texcoords) > 0 else None
    normals = torch.tensor(normals, dtype=torch.float32, device=device) if len(normals) > 0 else None

    faces = torch.tensor(faces, dtype=torch.int64, device=device)
    tfaces = torch.tensor(tfaces, dtype=torch.int64, device=device) if texcoords is not None else None
    nfaces = torch.tensor(nfaces, dtype=torch.int64, device=device) if normals is not None else None

    # Read weights and bones if available
    try:
        v_weights = torch.tensor(np.load(os.path.splitext(filename)[0] + ".weights.npy"), dtype=torch.float32, device=device)
        bone_mtx = torch.tensor(np.load(os.path.splitext(filename)[0] + ".bones.npy"), dtype=torch.float32, device=device)
    except:
        v_weights, bone_mtx = None, None

    return Mesh(
      vertices, faces, normals, nfaces, texcoords, tfaces,
      v_weights=v_weights,
      bone_mtx=bone_mtx,
      material=uber_material,
    )

def load_mtl(fn, device="cuda", default_bsdf="diffuse"):
    import re
    mtl_path = os.path.dirname(fn)

    # Read file
    with open(fn) as f:
        lines = f.readlines()

    # Parse materials
    materials = []
    material = None
    for line in lines:
        split_line = re.split(' +|\t+|\n+', line.strip())
        prefix = split_line[0].lower()
        data = split_line[1:]
        if 'newmtl' in prefix:
            material = {'name' : data[0]}
            materials += [material]
        elif len(materials) > 0:
            choices = ["bsdf", "map", "disp", "displacement", "depth", "bump", "refl"]
            if any(t in prefix for t in choices):
                material[prefix] = data[0]

    # Convert everything to textures. Our code expects 'kd' and 'ks' to be texture maps. So replace constants with 1x1 maps
    for mat in materials:
        if not 'bsdf' in mat:
            default = default_bsdf
            print(f"[INFO]: mtl missing bsdf kind, setting to {default}")
            mat['bsdf'] = default

        if 'map_kd' in mat:
            mat['kd'] = torchvision.io.read_image(os.path.join(mtl_path, mat['map_kd'])).to(device)
        else:
            mat['kd'] = torch.tensor([[0.0, 0.0, 0.0]], device=device, dtype=torch.float32)

        if 'map_ks' in mat:
            mat['ks'] = torchvision.io.read_image(os.path.join(mtl_path, mat['map_ks'])).to(device)
        else:
            mat['ks'] = torch.tensor([[0.0, 0.0, 0.0]], device=device, dtype=torch.float32)

        if 'bump' in mat:
            mat['normal'] = torchvision.io.read_image(os.path.join(mtl_path, mat['bump']))
            mat['normal'] = mat['normal'] * 2 - 1

    return materials

