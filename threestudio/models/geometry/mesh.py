import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import threestudio
from threestudio.models.geometry.base import BaseImplicitGeometry, contract_to_unisphere
from threestudio.models.mesh import Mesh as BaseMesh
from threestudio.utils.misc import get_rank
from threestudio.utils.typing import *
import nvdiffrast.torch as dr


@threestudio.register("mesh")
class Mesh(BaseImplicitGeometry):
    @dataclass
    class Config(BaseImplicitGeometry.Config):
      path: str = ""
      """resolution of an image"""
      res: int = 512

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.mesh = BaseMesh.load_from_file(self.cfg.path).unit_mesh()
        self.kd = nn.Parameter(torch.rand(self.cfg.res, self.cfg.res, 3, device="cuda", requires_grad=True))

    def initialize_shape(self) -> None: ...

    def isosurface(self): return self.mesh

    def forward(
        self, points: Float[Tensor, "*N Di"], output_normal: bool = False,
        uv=None,
    ) -> Dict[str, Float[Tensor, "..."]]:
      assert(uv is not None)
      rgb = dr.texture(self.kd[None], uv)
      return {
        "density": torch.ones_like(points),
        "rgb": rgb,
      }

    def forward_field(
        self, points: Float[Tensor, "*N Di"]
    ) -> Tuple[Float[Tensor, "*N 1"], Optional[Float[Tensor, "*N 3"]]]:
        raise NotImplementedError()

    def forward_level(
        self, field: Float[Tensor, "*N 1"], threshold: float
    ) -> Float[Tensor, "*N 1"]:
        raise NotImplementedError()

    def export(self, points: Float[Tensor, "*N Di"], **kwargs) -> Dict[str, Any]:
        torchvision.utils.save_image(self.kd.movedim(-1, 1), "out.png")
        raise NotImplementedError()
