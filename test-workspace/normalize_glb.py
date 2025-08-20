#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import trimesh
from trimesh.transformations import translation_matrix, scale_matrix

def pca_world_rotation(vertices: np.ndarray) -> np.ndarray:
    """
    对顶点做 PCA，返回把“主轴系”对齐到世界 XYZ 的旋转矩阵（4x4）。
    设特征向量矩阵 E 的列为主轴方向（在世界坐标中表达），
    我们希望把主轴坐标系变成世界坐标系，所以需要应用 R = E^T。
    """
    V = vertices - vertices.mean(axis=0)
    cov = np.cov(V.T)
    w, E = np.linalg.eigh(cov)             # 列向量为特征向量（正交）
    order = np.argsort(w)[::-1]            # 从大到小排序
    E = E[:, order]

    # 保证右手系
    if np.linalg.det(E) < 0:
        E[:, 2] *= -1.0

    R3 = E.T                                # 见注释：对点应用 R = E^T
    R = np.eye(4)
    R[:3, :3] = R3
    return R

def compute_transform(scene: trimesh.Scene, use_pca: bool, target_half_extent: float) -> list:
    """
    计算整体的 T/R/S 变换矩阵列表（按应用顺序返回），
    以便后续对 Scene 顺序 apply_transform（从而避免矩阵次序歧义）。
    """
    # 用合并网格来“测量”，不用于导出
    mesh = scene.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("无法从 glb 中提取有效网格。")

    # 1) 平移到质心为原点
    center = mesh.vertices.mean(axis=0)
    M_T = translation_matrix((-center).tolist())

    # 在“测量网格”上先应用 T，便于后续计算旋转和包围盒
    mesh_T = mesh.copy()
    mesh_T.apply_transform(M_T)

    # 2) （可选）PCA 对齐
    if use_pca:
        M_R = pca_world_rotation(mesh_T.vertices)
        mesh_TR = mesh_T.copy()
        mesh_TR.apply_transform(M_R)
    else:
        M_R = np.eye(4)
        mesh_TR = mesh_T

    # 3) 统一缩放到 [-target, target]
    extents = mesh_TR.extents
    max_extent = float(extents.max())
    if max_extent <= 0:
        raise ValueError("网格尺寸为零，无法缩放。")
    scale = (2.0 * target_half_extent) / max_extent
    M_S = scale_matrix(scale)

    # 返回按应用顺序的矩阵（依次对 scene.apply_transform）
    return [M_T, M_R, M_S]

def normalize_glb_preserve_materials(
    input_path: str,
    output_path: str,
    use_pca: bool = False,
    target_half_extent: float = 1.0
):
    # 以 Scene 读入（保留材质/纹理/法线/节点）
    scene = trimesh.load(input_path, force='scene')

    # 计算统一的 T/R/S
    transforms = compute_transform(scene, use_pca, target_half_extent)

    # 对整个 Scene 依次应用同一套变换（保留所有材质与层级）
    for M in transforms:
        scene.apply_transform(M)

    # 直接导出 Scene（glb）
    scene.export(output_path, file_type="glb")

def main():
    parser = argparse.ArgumentParser(
        description="Normalize a GLB while PRESERVING materials: recenter to origin, optional PCA axis align, and scale to fit within [-1,1]."
    )
    parser.add_argument("input", help="输入 .glb 文件路径")
    parser.add_argument("output", help="输出 .glb 文件路径")
    parser.add_argument("--pca", action="store_true",
                        help="是否启用 PCA 将主轴对齐到世界 XYZ（默认关闭）")
    parser.add_argument("--size", type=float, default=1.0,
                        help="目标半边长（默认 1.0，即让 AABB 落入 [-1,1]）")
    args = parser.parse_args()

    normalize_glb_preserve_materials(
        input_path=args.input,
        output_path=args.output,
        use_pca=args.pca,
        target_half_extent=args.size
    )
    print(f"✅ 已归一化并写出：{args.output}")

if __name__ == "__main__":
    main()

