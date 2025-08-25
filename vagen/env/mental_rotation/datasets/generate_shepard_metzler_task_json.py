# make_rotation_tasks_stepwise.py
# 生成 mental rotation 任务（step-by-step 四元数乘法版本）
# - 从 OBJ_DIR 读取 .glb
# - 对每个对象分别生成 5 组 single-step + 5 组 multi-step（每个“对象×任务”都有独立 RNG）
# - initial 用更细随机度（默认 10°），target 通过逐步四元数左乘得到
# - 输出 JSON 顶层包含 "asset_path": <OBJ_DIR 的最后一级目录名>

import os
import json
import random
from typing import Dict, List, Tuple

# ===== utils：使用你提供的函数 =====
from vagen.env.mental_rotation.utils import euler_xyz_to_quat, quat_multiply  # (x,y,z,degrees=True)->(w,x,y,z); q = q1 ⊗ q2

# =============== 配置区 ===============
OBJ_DIR = "./assets/b10_shepard_metzler_normalized"  # 放 .glb 的文件夹
OUTPUT_SINGLE = "shepard_metzler_single_step_rand_init.json"
OUTPUT_MULTI  = "shepard_metzler_multi_step_rand_init.json"

RANDOM_SEED = 2025
GRANULARITY_DEG = 90                  # 每步粒度（度）
NUM_SINGLE_PAIRS = 5
NUM_MULTI_PAIRS  = 5

# initial orientation 的随机粒度（例如 10°；需整除 360）
INITIAL_RANDOM_STEP = 10

# multi-step：每步角度 = m * GRANULARITY_DEG
MULTI_STEP_M_CHOICES = [1, 2, 3]
MULTI_NUM_STEPS_RANGE = (2, 4)        # 步数范围（含端点）
# ====================================

# 背景 / 相机（随时可改）
BACKGROUND_SPEC = {
    "background": "plane",
    "background_scale": 1.0,
    "background_euler": {"x": 0.0, "y": 0.0, "z": 0.0},
    "background_pos":   {"x": 0.0, "y": 0.0, "z": -1.0},
    "object_scale": 1.0,
    "camera_pos":   {"x": 5.0, "y": -2.0, "z": 2.5},
    "camera_lookat":{"x": 0.0, "y": 0.0, "z": 0.5},
}

AXES = ("x", "y", "z")

# ----------------- 工具函数 -----------------
def find_glb_objects(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Object folder not found: {folder}")
    files = [f for f in os.listdir(folder) if f.lower().endswith(".glb")]
    files.sort()
    return files

def get_asset_path_name(folder: str) -> str:
    """返回 OBJ_DIR 的最后一级目录名（不含路径）。"""
    return os.path.basename(os.path.normpath(folder))

def rand_initial_euler(rng: random.Random) -> Tuple[int, int, int]:
    """initial 欧拉角（度），更细随机度；extrinsic XYZ。"""
    step = INITIAL_RANDOM_STEP
    if 360 % step != 0:
        raise ValueError(f"INITIAL_RANDOM_STEP ({step}) must divide 360.")
    return (
        rng.randrange(0, 360, step),
        rng.randrange(0, 360, step),
        rng.randrange(0, 360, step),
    )

def quat_dict(q: Tuple[float, float, float, float], ndigits: int = 4) -> Dict[str, float]:
    w, x, y, z = q
    return {
        "w": round(float(w), ndigits),
        "x": round(float(x), ndigits),
        "y": round(float(y), ndigits),
        "z": round(float(z), ndigits),
    }

def step_quat(axis: str, angle_deg: int) -> Tuple[float, float, float, float]:
    """单步绕世界轴的旋转四元数（extrinsic XYZ）。左乘到当前姿态上。"""
    if axis == "x":
        return euler_xyz_to_quat(angle_deg, 0, 0, degrees=True)
    elif axis == "y":
        return euler_xyz_to_quat(0, angle_deg, 0, degrees=True)
    elif axis == "z":
        return euler_xyz_to_quat(0, 0, angle_deg, degrees=True)
    else:
        raise ValueError(f"Unknown axis: {axis}")

# ----------------- 单/多步生成（每次只生成 1 对） -----------------
def make_single_pair(rng: random.Random):
    """
    返回 (init_q, target_q, human_desc)
    single-step：仅一步 ±granularity 度，绕单一轴。
    """
    init_euler = rand_initial_euler(rng)
    init_q = euler_xyz_to_quat(*init_euler, degrees=True)

    axis = rng.choice(AXES)
    sign = rng.choice([-1, 1])
    angle = sign * GRANULARITY_DEG

    r = step_quat(axis, angle)            # 该步对应的旋转
    target_q = quat_multiply(r, init_q)   # 左乘：extrinsic（绕世界轴）

    desc = f"Single-step: rotate {abs(angle)}° about {axis.upper()}."
    return init_q, target_q, desc

def make_multi_pair(rng: random.Random):
    """
    返回 (init_q, target_q, human_desc)
    multi-step：≥2 步，至少两个不同轴；每步为 m*granularity。
    """
    init_euler = rand_initial_euler(rng)
    init_q = euler_xyz_to_quat(*init_euler, degrees=True)

    n_steps = rng.randint(*MULTI_NUM_STEPS_RANGE)
    steps = []
    used_axes = set()

    for _ in range(n_steps):
        axis = rng.choice(AXES)
        mult = rng.choice(MULTI_STEP_M_CHOICES)
        sign = rng.choice([-1, 1])
        angle = sign * mult * GRANULARITY_DEG
        steps.append((axis, angle))
        used_axes.add(axis)

    # 约束：至少两个不同轴
    if len(used_axes) < 2:
        extra_axes = [a for a in AXES if a not in used_axes]
        axis = rng.choice(extra_axes) if extra_axes else rng.choice(AXES)
        angle = rng.choice([-1, 1]) * GRANULARITY_DEG
        steps.append((axis, angle))

    # 逐步左乘
    q = init_q
    for axis, angle in steps:
        r = step_quat(axis, angle)
        q = quat_multiply(r, q)
    target_q = q

    step_strs = [f"{('+' if a>0 else '')}{a}° about {ax.upper()}" for ax, a in steps]
    desc = "Multi-step: " + " -> ".join(step_strs)
    return init_q, target_q, desc

# ----------------- 打包 JSON -----------------
def build_task_entry(obj_name: str,
                     init_q: Tuple[float,float,float,float],
                     target_q: Tuple[float,float,float,float],
                     instruction: str) -> Dict:
    task = {
        "object": obj_name,
        "initial_orientation": quat_dict(init_q),
        "target_orientation":  quat_dict(target_q),
        **BACKGROUND_SPEC,
        "instruction": f"Rotate the object to match the target orientation. {instruction}"
    }
    return task

# 用于构造“每对象×每任务”的独立 RNG 种子
def per_task_seed(base_seed: int, obj_index: int, task_index: int, kind: str) -> int:
    """
    生成稳定且分布良好的种子：
    kind: "single" 或 "multi"
    """
    # 两个不同的大素数，减少碰撞
    P1, P2 = 1000003, 1000033
    kind_offset = 17 if kind == "single" else 23
    return base_seed + obj_index * P1 + task_index * P2 + kind_offset

def main():
    obj_files = find_glb_objects(OBJ_DIR)
    if not obj_files:
        raise FileNotFoundError(f"No .glb objects found in: {OBJ_DIR}")

    asset_path_name = get_asset_path_name(OBJ_DIR)

    single_tasks = {"asset_path": asset_path_name, "tasks": []}
    multi_tasks  = {"asset_path": asset_path_name, "tasks": []}

    # 对每个对象分别生成：每个 task 使用不同 RNG
    for obj_idx, obj in enumerate(obj_files):
        # single-step
        for s_idx in range(NUM_SINGLE_PAIRS):
            rng = random.Random(per_task_seed(RANDOM_SEED, obj_idx, s_idx, "single"))
            init_q, targ_q, desc = make_single_pair(rng)
            single_tasks["tasks"].append(
                build_task_entry(obj, init_q, targ_q, f"The target shows a single-step rotation. {desc}")
            )
        # multi-step
        for m_idx in range(NUM_MULTI_PAIRS):
            rng = random.Random(per_task_seed(RANDOM_SEED, obj_idx, m_idx, "multi"))
            init_q, targ_q, desc = make_multi_pair(rng)
            multi_tasks["tasks"].append(
                build_task_entry(obj, init_q, targ_q, f"The target is reachable via multiple steps. {desc}")
            )

    with open(OUTPUT_SINGLE, "w", encoding="utf-8") as f:
        json.dump(single_tasks, f, indent=4)
    with open(OUTPUT_MULTI, "w", encoding="utf-8") as f:
        json.dump(multi_tasks, f, indent=4)

    print(f"[asset_path={asset_path_name}] Wrote {len(single_tasks['tasks'])} single-step tasks -> {OUTPUT_SINGLE}")
    print(f"[asset_path={asset_path_name}] Wrote {len(multi_tasks['tasks'])} multi-step tasks  -> {OUTPUT_MULTI}")

if __name__ == "__main__":
    main()
