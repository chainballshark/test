import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

# 创建简单的测试张量
means3D = torch.randn((107919, 3), device='cuda').to(torch.float32)
means2D = torch.randn((107919, 2), device='cuda').to(torch.float32)
opacities = torch.randn((107919, 1), device='cuda').to(torch.float32)
sh = torch.randn((107919, 16, 3), device='cuda').to(torch.float32)
scales = torch.randn((107919, 3), device='cuda').to(torch.float32)
rotations = torch.randn((107919, 4), device='cuda').to(torch.float32)
cov3Ds_precomp = torch.Tensor([]).to(torch.float32)

raster_settings = GaussianRasterizationSettings(
    image_height=543,
    image_width=979,
    tanfovx=0.8437573789483421,
    tanfovy=0.4660729587709916,
    bg=torch.tensor([0, 0, 0], device='cuda').to(torch.float32),
    scale_modifier=1.0,
    viewmatrix=torch.eye(4, device='cuda').to(torch.float32),
    projmatrix=torch.eye(4, device='cuda').to(torch.float32),
    sh_degree=0,
    campos=torch.tensor([0, 0, 0], device='cuda').to(torch.float32),
    prefiltered=False,
    debug=False,
)

rasterizer = GaussianRasterizer(raster_settings)

# 调用 rasterizer 并打印输入张量的形状和数据类型
print("Calling rasterizer with the following inputs:")
print("means3D shape:", means3D.shape, "dtype:", means3D.dtype, "device:", means3D.device)
print("means2D shape:", means2D.shape, "dtype:", means2D.dtype, "device:", means2D.device)
print("opacities shape:", opacities.shape, "dtype:", opacities.dtype, "device:", opacities.device)
print("sh shape:", sh.shape, "dtype:", sh.dtype, "device:", sh.device)
print("scales shape:", scales.shape, "dtype:", scales.dtype, "device:", scales.device)
print("rotations shape:", rotations.shape, "dtype:", rotations.dtype, "device:", rotations.device)
print("cov3Ds_precomp shape:", cov3Ds_precomp.shape, "dtype:", cov3Ds_precomp.dtype, "device:", cov3Ds_precomp.device)

# 调用 rasterizer
output = rasterizer(means3D, means2D, opacities, sh, scales=scales, rotations=rotations)
