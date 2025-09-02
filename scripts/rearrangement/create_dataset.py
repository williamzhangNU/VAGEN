import ai2thor.controller
from ai2thor.platform import CloudRendering
import random
import json
import numpy as np
import os
from PIL import Image
import re

from concurrent.futures import ThreadPoolExecutor
all_scenes = [
    # 厨房场景 (FloorPlan1-30)
    # "FloorPlan1", "FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5",
    # "FloorPlan6", "FloorPlan7", "FloorPlan8", "FloorPlan9", "FloorPlan10",
    # "FloorPlan11", "FloorPlan12", "FloorPlan13", "FloorPlan14", "FloorPlan15",
    # "FloorPlan16", "FloorPlan17", "FloorPlan18", "FloorPlan19", "FloorPlan20",
    # "FloorPlan21", "FloorPlan22", "FloorPlan23", "FloorPlan24", "FloorPlan25",
    # "FloorPlan26", "FloorPlan27", "FloorPlan28", "FloorPlan29", "FloorPlan30",

    # 客厅场景 (FloorPlan201-230)
    # "FloorPlan211",
    "FloorPlan201", "FloorPlan202", "FloorPlan203", "FloorPlan204", "FloorPlan205",
    "FloorPlan206", "FloorPlan207", "FloorPlan208", "FloorPlan209", "FloorPlan210",
    "FloorPlan211", "FloorPlan212", "FloorPlan213", "FloorPlan214", "FloorPlan215",
    "FloorPlan216", "FloorPlan217", "FloorPlan218", "FloorPlan219", "FloorPlan220",
    "FloorPlan221", "FloorPlan222", "FloorPlan223", "FloorPlan224", "FloorPlan225",
    "FloorPlan226", "FloorPlan227", "FloorPlan228", "FloorPlan229", "FloorPlan230",

    # 卧室场景 (FloorPlan301-330)
    # "FloorPlan301", "FloorPlan302", "FloorPlan303", "FloorPlan304", "FloorPlan305",
    # "FloorPlan306", "FloorPlan307", "FloorPlan308", "FloorPlan309", "FloorPlan310",
    # "FloorPlan311", "FloorPlan312", "FloorPlan313", "FloorPlan314", "FloorPlan315",
    # "FloorPlan316", "FloorPlan317", "FloorPlan318", "FloorPlan319", "FloorPlan320",
    # "FloorPlan321", "FloorPlan322", "FloorPlan323", "FloorPlan324", "FloorPlan325",
    # "FloorPlan326", "FloorPlan327", "FloorPlan328", "FloorPlan329", "FloorPlan330",

    # 浴室场景 (FloorPlan401-430)
    # "FloorPlan401", "FloorPlan402", "FloorPlan403", "FloorPlan404", "FloorPlan405",
    # "FloorPlan406", "FloorPlan407", "FloorPlan408", "FloorPlan409", "FloorPlan410",
    # "FloorPlan411", "FloorPlan412", "FloorPlan413", "FloorPlan414", "FloorPlan415",
    # "FloorPlan416", "FloorPlan417", "FloorPlan418", "FloorPlan419", "FloorPlan420",
    # "FloorPlan421", "FloorPlan422", "FloorPlan423", "FloorPlan424", "FloorPlan425",
    # "FloorPlan426", "FloorPlan427", "FloorPlan428", "FloorPlan429", "FloorPlan430"
]
class TaskGenerator:
    def __init__(self, output_dir="generated_tasks", seed=42):
        # Set random seeds for reproducibility
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        # Initialize with a regular scene first, then switch to procedural
        self.controller = ai2thor.controller.Controller(
            agentMode="default",
            visibilityDistance=10,
            gridSize=0.25,
            snapToGrid=False,
            rotateStepDegrees=90,
            renderDepthImage=False,
            renderInstanceSegmentation=False,
            width=960,
            height=540,
            platform=CloudRendering,
            fieldOfView=90,
            randomSeed=seed  # Set AI2Thor's random seed
        )

        # 创建输出目录
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)

        # 只允许地面到地面的移动，移除"on"关系
        self.spatial_relations = ["behind", "in front of", "to the left of", "to the right of"]

    def save_viewpoint_image(self, suffix="before", prefix=None):
        """保存当前视角的图片；支持可选前缀以避免并发文件名冲突。
        例如：<prefix>_FloorPlan201_before.png
        """
        try:
            # 获取当前帧
            event = self.controller.step("Pass")
            if event.metadata["lastActionSuccess"]:
                # 转换为PIL图片
                image = Image.fromarray(event.frame)

                # 基于当前场景名生成文件名
                scene_name = getattr(self, "current_scene", "scene")
                base = f"{scene_name}_{suffix}.png"
                filename = f"{prefix}_{base}" if prefix else base
                filepath = os.path.join(self.images_dir, filename)

                # 保存图片
                image.save(filepath)
                print(f"Saved image: {filepath}")
                return filename
            else:
                print("Failed to capture image for current scene")
                return None
        except Exception as e:
            print(f"Error saving image for current scene: {e}")
            return None

    def is_object_visible(self, obj, percent: float = None, save_filtered_path: str = '/home/zihanhuang/VAGEN/rearrangement_dataset/images'):
        """检查物体是否在当前视角中可见。
        当 percent == 0 时，沿用原有判定：obj["visible"] 且距离小于 visibilityDistance。
        当 percent > 0 时，要求该物体的可见像素占无遮挡像素的比例达到百分之 percent 才视为可见。
        若当前未开启实例分割或获取掩码失败，将回退到原有判定。

        参数:
            obj: AI2-THOR 物体元数据字典
            percent: 百分阈值（0-100），基于可见像素/无遮挡像素的比例
        """
        base_visible = obj.get("visible", False) and (
            obj.get("distance", float("inf")) < self.controller.initialization_parameters["visibilityDistance"]
        )
        if not base_visible:
            return False
        if percent is None:
            return base_visible

        # 需要实例分割来统计像素占比
        if not self.controller.initialization_parameters.get("renderInstanceSegmentation", False):
            return base_visible

        # 获取当前事件（包含分割信息）
        event = self.controller.step("Pass")
        frame = getattr(event, "frame", None)
        if frame is None:
            return base_visible

        # 当前场景下的可见像素
        visible_pixels = 0
        inst_masks = getattr(event, "instance_masks", None)
        obj_id = obj.get("objectId") or obj.get("name")
        if inst_masks and obj_id in inst_masks and inst_masks[obj_id] is not None:
            try:
                visible_pixels = int(np.sum(inst_masks[obj_id]))
            except Exception:
                visible_pixels = 0
        # 若提供保存路径：同时保存原图与掩码过滤图（以 objectType 命名）
        if save_filtered_path and inst_masks and obj_id in inst_masks and inst_masks[obj_id] is not None:
            try:
                obj_type = obj.get("objectType")
                os.makedirs(save_filtered_path, exist_ok=True)
                # 原图
                original_path = os.path.join(save_filtered_path, f"{obj_type}_original.png")
                Image.fromarray(frame).save(original_path)
                # 过滤后图使用随后“仅渲染该物体”的帧保存
                # 在下方渲染后保存 filtered 图像
                pass
            except Exception as e:
                print(f"Failed to save images to {save_filtered_path}: {e}")


        # 仅渲染该物体，计算"无遮挡可见像素数"
        unoccluded_pixels = 0
        if obj_id:
            other_ids = []
            try:
                # 确保使用当前帧的 target_id（避免 name/id 不一致导致误禁用目标）
                target_id = None
                meta_objs = event.metadata.get("objects", []) if hasattr(event, 'metadata') else []
                for o in meta_objs:
                    if o.get("objectId") and (o.get("objectId") == obj.get("objectId") or o.get("name") == obj.get("name")):
                        target_id = o.get("objectId")
                        break
                if target_id is None:
                    target_id = obj_id  # 回退

                # 不禁用结构体，但允许禁用家具（包括父容器），以便去除遮挡
                structural_types = {"floor", "wall", "walls", "ceiling", "ceilings"}
                def is_structural(otype: str) -> bool:
                    return str(otype).lower() in structural_types

                other_ids = []
                for o in meta_objs:
                    oid = o.get("objectId")
                    if not oid or oid == target_id:
                        continue
                    if is_structural(o.get("objectType", "")):
                        continue
                    other_ids.append(oid)

                # 优先使用 SetObjectFilter 仅渲染目标；失败则回退到 HideObject/SetObjectVisibility
                try:
                    # 仅渲染目标物体（不改变物理）
                    self.controller.step(action="SetObjectFilter", objectIds=[target_id])
                    assert self.controller.last_event.metadata["lastActionSuccess"]
                    event_single = self.controller.step("Pass")
                except Exception:
                    # 回退方案：逐个隐藏其他物体（优先 HideObject，不行则 SetObjectVisibility）
                    for oid in other_ids:
                        try:
                            self.controller.step(action="HideObject", objectId=oid)
                        except Exception:
                            try:
                                self.controller.step(action="SetObjectVisibility", objectId=oid, visible=False, forceAction=True)
                            except Exception:
                                pass
                    event_single = self.controller.step("Pass")

                inst_masks_single = getattr(event_single, "instance_masks", None)
                if inst_masks_single and obj_id in inst_masks_single and inst_masks_single[obj_id] is not None:
                    try:
                        unoccluded_pixels = int(np.sum(inst_masks_single[obj_id]))
                    except Exception:
                        unoccluded_pixels = 0
            finally:
                # 恢复所有物体的可见性（渲染层面）
                for oid in other_ids:
                    try:
                        self.controller.step(action="SetObjectVisibility", objectId=oid, visible=True, forceAction=True)
                    except Exception:
                        pass

        # 根据可见像素占无遮挡像素的比例判断
        if unoccluded_pixels > 0:
            visible_ratio_percent = (visible_pixels / float(unoccluded_pixels)) * 100.0
            if save_filtered_path:
                try:
                    obj_type = obj.get("objectType")
                    os.makedirs(save_filtered_path, exist_ok=True)
                    filtered_path = os.path.join(save_filtered_path, f"{obj_type}_filtered.png")
                    Image.fromarray(event_single.frame).save(filtered_path)
                    print(f"Saved filtered image: {filtered_path}")
                except Exception as e:
                    print(f"Failed to save filtered image to {save_filtered_path}: {e}")
            return visible_ratio_percent >= float(percent)
        else:
            return base_visible


    def find_good_viewpoint(self, max_attempts=50):
        """按照0.5米最小间隔、避开碰撞点，筛选可见的地面物体中
        movable 和 reference 都>3的视角；在满足条件的视角中选择地面物体最多的一个"""
        print("Scanning reachable viewpoints (min spacing 0.5m) and filtering by movable/reference > 3 ...")

        # 先获取可达位置（天然避开与物体/障碍的碰撞）
        event = self.controller.step(action="GetReachablePositions")
        if not event.metadata.get("lastActionSuccess", False):
            print("Failed to get reachable positions")
            return None
        reachable = event.metadata.get("actionReturn", []) or []
        if not reachable:
            print("No reachable positions returned")
            return None

        # 以0.5m为最小间隔做下采样，避免过密评估
        min_spacing = 0.5
        selected_positions = []
        for p in reachable:
            px, pz = p.get("x", 0.0), p.get("z", 0.0)
            keep = True
            for q in selected_positions:
                dx = px - q["x"]
                dz = pz - q["z"]
                if dx * dx + dz * dz < (min_spacing - 1e-6) ** 2:
                    keep = False
                    break
            if keep:
                selected_positions.append({"x": px, "y": p.get("y", 0.9), "z": pz})

        print(f"Selected {len(selected_positions)} positions after 0.5m spacing filter (from {len(reachable)} reachable).")

        # 四个主要朝向
        rotations = [0, 90, 180, 270]
        total_candidates = len(selected_positions) * len(rotations)
        print(f"Evaluating {total_candidates} viewpoint candidates ...")

        best_viewpoint = None
        max_objects = -1
        candidate_count = 0

        for pos in selected_positions:
            for rotation_y in rotations:
                candidate_count += 1

                # 尝试传送至候选视角
                event = self.controller.step(
                    action="Teleport",
                    position=pos,
                    rotation={"x": 0, "y": rotation_y, "z": 0}
                )
                if not event.metadata.get("lastActionSuccess", False):
                    continue

                visible_objects, ground_objects = self.get_visible_and_ground_objects()
                # 统计当前视角的对象（仅考虑可移动的地面物体）
                movable_objs = self.get_visible_and_moveable_objects()

                # 仅保留 movable > 3 的视角
                if len(movable_objs) > 0:
                    if len(visible_objects) > max_objects:
                        max_objects = len(visible_objects)
                        best_viewpoint = {
                            "position": {"x": pos["x"], "y": pos["y"], "z": pos["z"]},
                            "rotation": {"x": 0, "y": rotation_y, "z": 0},
                            "total_visible_objects": len(visible_objects),
                            "movable_count": len(movable_objs)
                        }
                        print(
                            f"New valid viewpoint: pos=({pos['x']:.2f},{pos['z']:.2f}), rot={rotation_y}°, "
                            f"movable={len(movable_objs)}"
                        )

                # 进度日志
                if candidate_count % 200 == 0:
                    print(
                        f"Progress: {candidate_count}/{total_candidates}, "
                        f"current best total_ground={max_objects if max_objects>=0 else 0}"
                    )

        if best_viewpoint:
            # 定位到最佳视角（上面已包含统计）
            self.controller.step(
                action="Teleport",
                position=best_viewpoint["position"],
                rotation=best_viewpoint["rotation"]
            )
            print(
                f"Final best viewpoint: pos=({best_viewpoint['position']['x']:.2f},{best_viewpoint['position']['z']:.2f}), "
                f"rotation={best_viewpoint['rotation']['y']}°, movable={best_viewpoint['movable_count']}, "
                f"total_visible_objects={best_viewpoint['total_visible_objects']}"
            )
        else:
            print("No valid viewpoint (movable>3) found under 0.5m spacing.")

        return best_viewpoint

    def plan_ground_path(self, start_pos, goal_pos, target_obj, step_size=0.25, padding=2.0):
        """在地面上为目标物体规划一条无碰撞轨迹（xz 平面）。
        A* 算法 + 4 邻接，网格步长为 step_size（默认 0.25m）。
        每个网格点是否可行通过 PlaceObjectAtPoint 尝试判断（成功=可行），
        尝试后立即将物体放回 original_pos，避免污染场景。
        返回路径点列表（包含起点和终点），每个点为 {x,y,z}；无路则返回 None。
        """
        import heapq
        import math

        target_id = target_obj.get("objectId") or target_obj.get("name")
        ground_y = start_pos.get("y", 0.1)
        base_rot = target_obj["rotation"]


        # 搜索边界（围绕起终点加 padding）
        min_x = min(start_pos["x"], goal_pos["x"]) - padding
        max_x = max(start_pos["x"], goal_pos["x"]) + padding
        min_z = min(start_pos["z"], goal_pos["z"]) - padding
        max_z = max(start_pos["z"], goal_pos["z"]) + padding

        def in_bounds(x, z):
            return (min_x <= x <= max_x) and (min_z <= z <= max_z)

        # 网格映射
        def world_to_grid(x, z):
            gx = int(round((x - min_x) / step_size))
            gz = int(round((z - min_z) / step_size))
            return gx, gz

        def grid_to_world(gx, gz):
            x = min_x + gx * step_size
            z = min_z + gz * step_size
            return x, z

        # 可行性判定缓存
        passable_cache = {}
        def is_passable(gx, gz):
            key = (gx, gz)
            if key in passable_cache:
                return passable_cache[key]
            wx, wz = grid_to_world(gx, gz)
            if not in_bounds(wx, wz):
                passable_cache[key] = False
                return False
            ev = self.controller.step(
                action="PlaceObjectAtPoint",
                objectId=target_id,
                position={"x": wx, "y": ground_y, "z": wz},
                rotation=base_rot,
            )
            ok = bool(ev.metadata.get("lastActionSuccess", False))
            passable_cache[key] = ok
            return ok

        start_g = world_to_grid(start_pos["x"], start_pos["z"])
        goal_g = world_to_grid(goal_pos["x"], goal_pos["z"])

        # 起终点可行性
        if not is_passable(*start_g) or not is_passable(*goal_g):
            return None

        # A*（4邻）
        def h(a, b):
            (x1, z1), (x2, z2) = a, b
            return math.hypot(x1 - x2, z1 - z2)

        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        open_heap = []
        heapq.heappush(open_heap, (0, start_g))
        came_from = {start_g: None}
        g_score = {start_g: 0.0}

        max_nodes = 20000
        expansions = 0

        while open_heap and expansions < max_nodes:
            _, current = heapq.heappop(open_heap)
            if current == goal_g:
                break
            cx, cz = current
            for dx, dz in neighbors:
                nx, nz = cx + dx, cz + dz
                if not is_passable(nx, nz):
                    continue
                tentative_g = g_score[current] + step_size  # 4邻，固定步长
                if (nx, nz) not in g_score or tentative_g < g_score[(nx, nz)]:
                    g_score[(nx, nz)] = tentative_g
                    priority = tentative_g + h((nx, nz), goal_g)
                    heapq.heappush(open_heap, (priority, (nx, nz)))
                    came_from[(nx, nz)] = current
            expansions += 1

        if goal_g not in came_from:
            return None

        # 回溯路径（格点->世界坐标）
        path_g = []
        cur = goal_g
        while cur is not None:
            path_g.append(cur)
            cur = came_from[cur]
        path_g.reverse()

        path = []
        for gx_i, gz_i in path_g:
            wx, wz = grid_to_world(gx_i, gz_i)
            path.append({"x": round(wx, 3), "y": ground_y, "z": round(wz, 3)})

        # 简单压缩共线点
        def is_colinear(p1, p2, p3, eps=1e-6):
            v1x, v1z = p2["x"] - p1["x"], p2["z"] - p1["z"]
            v2x, v2z = p3["x"] - p2["x"], p3["z"] - p2["z"]
            return abs(v1x * v2z - v1z * v2x) < eps
        if len(path) > 2:
            compressed = [path[0]]
            for i in range(1, len(path) - 1):
                if not is_colinear(path[i - 1], path[i], path[i + 1]):
                    compressed.append(path[i])
            compressed.append(path[-1])
            path = compressed

        return path
    def _convert_path_to_view_relative_moves(self, path, min_segment=0.05, quantum=0.25):
        """将世界坐标路径转换为相对于当前相机朝向的前/后/左/右移动序列，并量化到 0.25m。
        - path: [{x,y,z}, ...]，至少包含两点
        - 返回: [{"dir": one of {forward,backward,left,right}, "meters": float}, ...]
        - 会合并连续相同方向的段
        """
        import math
        if not path or len(path) < 2:
            return []
        # 当前相机朝向（弧度），用于定义局部坐标系
        cam_yaw_deg = self.controller.last_event.metadata["agent"]["rotation"]["y"]
        cam_yaw = math.radians(cam_yaw_deg)
        sin_y = math.sin(cam_yaw)
        cos_y = math.cos(cam_yaw)
        def world_to_local(dx, dz):
            # 将世界坐标增量投影到以相机为基的局部坐标
            # forward=+z_local（与相机朝向一致），right=+x_local（相机右手方向）
            # 局部轴：forward = [sin(y), cos(y)], right = [cos(y), -sin(y)]（仅 xz 分量）
            x_local = dx * cos_y - dz * sin_y    # = dot(delta, right)
            z_local = dx * sin_y + dz * cos_y    # = dot(delta, forward)
            return x_local, z_local
        # 生成原子动作：把每一段分解到前/后与左/右两个轴（各自可能为0）
        raw = []
        for i in range(1, len(path)):
            dx = path[i]["x"] - path[i-1]["x"]
            dz = path[i]["z"] - path[i-1]["z"]
            lx, lz = world_to_local(dx, dz)
            # 先处理前/后
            if abs(lz) >= min_segment:
                raw.append({"dir": "forward" if lz >= 0 else "backward", "meters": abs(lz)})
            # 再处理左/右
            if abs(lx) >= min_segment:
                raw.append({"dir": "right" if lx >= 0 else "left", "meters": abs(lx)})
        # 合并同向段
        if not raw:
            return []
        merged = [raw[0]]
        for seg in raw[1:]:
            if seg["dir"] == merged[-1]["dir"]:
                merged[-1]["meters"] += seg["meters"]
            else:
                merged.append(seg)
        # 量化到 0.25m
        def quantize(x: float, q: float) -> float:
            return round(round(x / q) * q, 3)
        for m in merged:
            m["meters"] = quantize(m["meters"], quantum)
        # 去除 0 段
        merged = [m for m in merged if m["meters"] > 0]
        return merged



    def find_far_visible_position(self, target_obj, min_distance: float = 3.0, max_distance: float = 6.0, save_each_move: bool = False):
        """为目标物体寻找一个无遮挡且无碰撞、距离原位置至少 min_distance 的新位置。
        返回 (target_pos, movement_moves) 或 None。
        其中 movement_moves 是相对于“当前视角”的前/后/左/右移动序列，例如：
        [
            {"dir": "forward", "meters": 0.4},
            {"dir": "right", "meters": 0.2},
            ...
        ]
        如果 save_each_move=True，会按 moves 逐步移动“目标物体”（使用 PlaceObjectAtPoint），
        在每步后保存一张图片；Agent 视角不会改变。
        """


        # 获取场景对象
        event = self.controller.step("Pass")
        all_objects = event.metadata.get("objects", [])
        # 基于 receptacle 的可放置点生成候选位置
        ground_y = 0.1
        original_pos = target_obj["position"].copy()
        original_rot = target_obj["rotation"].copy()
        # 只使用 Floor 作为 receptacle（允许 Floor、Floor1 等名字）
        def is_floor(otype: str) -> bool:
            tl = str(otype).lower()
            # 有些场景中地板类型可能命名为 Floor/Floor1 等，做一个宽松匹配
            return tl=="floor"
        receptacles = [o for o in all_objects if is_floor(o.get("objectType", ""))]

        # 为每个 receptacle 获取可放置坐标并当场验证（无碰撞且可见）
        target_id = target_obj.get("objectId")
        for rec in receptacles:
            rec_id = rec.get("objectId")
            if not rec_id:
                continue
            ev = self.controller.step(action="GetSpawnCoordinatesAboveReceptacle", objectId=rec_id, anywhere=False)
            coords = ev.metadata.get("actionReturn") or []
            for p in coords:
                # 距离过滤（仅使用 xz 平面距离）
                dx = p.get("x", 0.0) - original_pos.get("x", 0.0)
                dz = p.get("z", 0.0) - original_pos.get("z", 0.0)
                dist = (dx * dx + dz * dz) ** 0.5
                if dist < float(min_distance):
                    continue
                if max_distance is not None and dist > float(max_distance):
                    continue
                candidate = {"x": p.get("x", 0.0), "y": p.get("y", ground_y), "z": p.get("z", 0.0)}

                # 直接尝试放置（碰撞/可放置判定）
                move_event = self.controller.step(
                    action="PlaceObjectAtPoint",
                    objectId=target_id,
                    position=candidate,
                    rotation=original_rot,
                )
                if not move_event.metadata.get("lastActionSuccess", False):
                    continue

                # 放置成功：可见性用 is_object_visible 直接检查
                check_event = self.controller.step("Pass")
                moved_obj = next((o for o in check_event.metadata.get("objects", [])
                                  if o.get("name") == target_obj.get("name")), None)
                if not (moved_obj and self.is_object_visible(moved_obj)):
                    self.controller.step(action="PlaceObjectAtPoint", objectId=target_id, position=original_pos, rotation=original_rot)
                    continue

                # 可见后再验证是否存在 0.25m / 4邻 A* 可行路径
                start_xy = {"x": original_pos.get("x"), "y": ground_y, "z": original_pos.get("z")}
                path = self.plan_ground_path(start_xy, candidate, target_obj)
                if not path or len(path) < 2:
                    self.controller.step(action="PlaceObjectAtPoint", objectId=target_id, position=original_pos, rotation=original_rot)
                    if not self.controller.last_event.metadata.get("lastActionSuccess"):
                        self.save_viewpoint_image("fail")
                        raise Exception("Failed to restore original position after failed plan_ground_path.")
                    continue
                moves = self._convert_path_to_view_relative_moves(path)

                # 可选：回放 moves 并保存每一步的图像
                if save_each_move and moves:
                    self.controller.step(action="PlaceObjectAtPoint", objectId=target_id, position=original_pos, rotation=original_rot)
                    self._replay_and_save_moves(moves, target_id)

                self.controller.step(
                    action="PlaceObjectAtPoint",
                    objectId=target_id,
                    position=candidate,
                    rotation=original_rot,
                )

                return (candidate, moves)

    def _replay_and_save_moves(self, moves, target_id):
        """根据 moves 逐步移动“指定目标物体”（用 PlaceObjectAtPoint），每步后保存图片。
        Agent 位置与视角不变，仅改变目标物体的位置。
        """
        if not moves:
            return
        # 获取当前场景名作为前缀
        step_idx = 1
        # 将 forward/right 相对位移按相机朝向分解到世界坐标
        import math
        ev = self.controller.step("Pass")
        agent = ev.metadata.get("agent", {})
        rot_y = agent.get("rotation", {}).get("y", 0.0)
        ang = math.radians(rot_y)
        fwd = {"x": math.sin(ang), "z": math.cos(ang)}
        right = {"x": math.cos(ang), "z": -math.sin(ang)}
        offset = {"x": 0.0, "z": 0.0}
        # 从当前帧读取目标物体的位置作为基础
        objs = ev.metadata.get("objects", [])
        base_pos = None
        for o in objs:
            if (o.get("objectId") == target_id) or (o.get("name") == target_id):
                base_pos = o.get("position", {}).copy()
                break
        if base_pos is None:
            return
        for m in moves:
            meters = float(m.get("meters", 0.0))
            if meters <= 0:
                continue
            if m.get("dir") in ("forward", "backward"):
                s = meters if m["dir"] == "forward" else -meters
                offset["x"] += fwd["x"] * s
                offset["z"] += fwd["z"] * s
            elif m.get("dir") in ("right", "left"):
                s = meters if m["dir"] == "right" else -meters
                offset["x"] += right["x"] * s
                offset["z"] += right["z"] * s
            new_pos = {"x": base_pos["x"] + offset["x"], "y": base_pos["y"], "z": base_pos["z"] + offset["z"]}
            self.controller.step(action="PlaceObjectAtPoint", objectId=target_id, position=new_pos)
            # 保存图片
            suffix = f"step{step_idx}_{m['dir']}_{meters:.2f}"
            self.save_viewpoint_image(suffix=suffix)
            step_idx += 1

        return None


    def get_distance_between_objects(self, obj1, obj2):
        pos1 = obj1["position"]
        pos2 = obj2["position"]

        dx = pos1["x"] - pos2["x"]
        dz = pos1["z"] - pos2["z"]

        return (dx**2 + dz**2)**0.5

    def is_object_on_ground(self, obj):
        """检查物体是否在地面上或低矮表面上（更宽松的地面检测）"""
        ground_height = 0.15  # 地面高度
        obj_y = obj["position"]["y"]

        return obj_y <= ground_height

    def get_visible_and_ground_objects(self):
        """获取当前视角中所有在地面上的可见物体（包括可移动和不可移动物体）"""
        event = self.controller.step("Pass")
        objects = event.metadata["objects"]

        ground_objects = []
        visible_objects = []
        for obj in objects:
            if not self.is_object_visible(obj):
                continue
            visible_objects.append(obj)
            # 检查物体是否在地面上
            if not self.is_object_on_ground(obj):
                continue

            ground_objects.append(obj)

        return visible_objects, ground_objects

    def get_visible_and_moveable_objects(self):
        """获取当前视角中可见的可移动/可拾取地面物体列表（先筛可移动，再判可见）"""
        event = self.controller.step("Pass")
        objects = event.metadata["objects"]

        movable = []

        for obj in objects:
            # 仅考虑地面上的物体
            if not self.is_object_on_ground(obj):
                continue

            obj_type = obj["objectType"]

            # 先快速筛掉不可移动的
            is_movable = obj.get("moveable", False) or obj.get("pickupable", False)
            if not is_movable:
                continue

            receptacle = obj.get("receptacleObjectIds")
            if receptacle and len(receptacle) > 0:
                # 如果任何一个receptacle元素包含非'table'，则跳过此物体
                if any('table' not in r.lower() for r in receptacle):
                    continue

            # 过滤掉 sofa/table
            obj_type_lower = obj_type.lower()
            if "sofa" in obj_type_lower or "table" in obj_type_lower:
                continue

            # 再做可见性检查（较慢）
            if not self.is_object_visible(obj):
                continue


            movable.append(obj)

        return movable

    def filter_unique_objects(self, objects):
        """仅保留在该列表中类型出现次数为1的物体；并提前排除 floor/curtain 系列对象（可能带编号）"""
        from collections import Counter
        # 先排除 floor 和 curtain（大小写不敏感，允许编号）
        def _is_excluded(t: str) -> bool:
            tl = str(t).lower()
            return re.match(r"^floor[0-9]*$", tl) is not None or re.match(r"^curtains[0-9]*$", tl) is not None
        filtered = [obj for obj in objects if not _is_excluded(obj.get("objectType", ""))]
        # 仅保留类型唯一者
        types = [obj["objectType"] for obj in filtered]
        counts = Counter(types)
        unique_list = [obj for obj in filtered if counts[obj["objectType"]] == 1]
        return unique_list

    def generate_task_with_validation(self):
        """生成经过验证的任务（假定视角已在外部设置并已 Teleport）。
        需求：
        - 去除 reference 物体，不使用方位关系
        - after = 目标物体可无碰撞移动到的可见位置，且与原位置距离 >= 3m
        - 只需保存移动前/后的两张图（由调用方已处理）
        """
        # 获取可见的地面可移动物体
        movable_objs = self.get_visible_and_moveable_objects()

        if not movable_objs:
            return None

        # 直接按估计体积从小到大选择一个体积较小的目标
        # movable_objs.sort(key=lambda o: self.get_object_size(o))
        for target_obj in movable_objs:
            print(f"  Target Object: {target_obj['objectType']}")

            # 搜索满足 >=3m 且可见的无碰撞位置
            res = self.find_far_visible_position(target_obj, min_distance=3.0, max_distance=6.0)
            if not res:
                print("  No far visible collision-free position found (>=3m).")
                continue
            target_pos, movement_path = res

            # 构造任务（只包含必要信息；before/after 图片由外部保存）
            task_description = f"Move the {target_obj['objectType'].lower()} to a visible collision-free location at least 3m away."
            # 记录 agent 视角位姿
            ev = self.controller.step("Pass")
            agent_meta = ev.metadata.get("agent", {})
            agent_pose = {
                "position": agent_meta.get("position"),
                "rotation": agent_meta.get("rotation"),
            }
            return {
                "description": task_description,
                "agent_view": agent_pose,
                "target_object": {
                    "name": target_obj.get("name"),
                    "type": target_obj.get("objectType"),
                    "original_position": target_obj.get("position"),
                    "final_position": target_pos,
                },
                "movement_path": movement_path,
            }

    def generate_batch(self, num_tasks=10):
        """生成一批高质量的任务，每个任务使用独立的场景。
        要求：场景不能重复；如可选场景数少于 num_tasks 则抛出错误。
        """
        tasks = []

        # 准备不重复场景列表
        all_scenes = [
            "FloorPlan201", "FloorPlan202", "FloorPlan203", "FloorPlan204", "FloorPlan205",
            "FloorPlan206", "FloorPlan207", "FloorPlan208", "FloorPlan209", "FloorPlan210",
            "FloorPlan211", "FloorPlan212", "FloorPlan213", "FloorPlan214", "FloorPlan215",
            "FloorPlan216", "FloorPlan217", "FloorPlan218", "FloorPlan219", "FloorPlan220",
            "FloorPlan221", "FloorPlan222", "FloorPlan223", "FloorPlan224", "FloorPlan225",
            "FloorPlan226", "FloorPlan227", "FloorPlan228", "FloorPlan229", "FloorPlan230",
        ]
        if len(all_scenes) < num_tasks:
            raise ValueError(f"Not enough unique scenes to generate {num_tasks} tasks (available={len(all_scenes)})")

        # 随机抽取不重复的场景
        chosen_scenes = random.sample(all_scenes, num_tasks)

        for task_num, scene in enumerate(chosen_scenes):
            print(f"\nGenerating task {task_num + 1}/{num_tasks} for scene {scene}...")

            # 加载指定场景
            try:
                event = self.controller.reset(scene=scene)
            except Exception as e:
                print(f"  Exception loading scene {scene}: {e}")
                continue
            if not event.metadata.get("lastActionSuccess", False):
                print(f"  Failed to load scene {scene}: {event.metadata.get('errorMessage')}")
                continue
            self.current_scene = scene

            # 在该场景中尝试生成任务（只尝试一次）
            viewpoint = self.find_good_viewpoint()
            if not viewpoint:
                print("  No valid viewpoint found in this selected scene. Skipping...")
                continue
            self.controller.step(
                action="Teleport",
                position=viewpoint["position"],
                rotation=viewpoint["rotation"]
            )
            before_image = self.save_viewpoint_image("before")

            task = self.generate_task_with_validation()
            if task:
                task["scene_id"] = scene

                if before_image:
                    task["before_image"] = before_image

                after_image = self.save_viewpoint_image("after")
                if after_image:
                    task["after_image"] = after_image

                # 移除临时数据
                task.pop("target_obj_data", None)
                task.pop("ref_obj_data", None)

                print(f"  ✅ Successfully generated task: {task['description']}")
            else:
                print("  No valid task found in this selected scene. Returning None for this task.")

            if task:
                tasks.append(task)
                print(f"Generated task {len(tasks)}: {task['description']}")
            else:
                print(f"❌ Task {task_num + 1}: None (no valid scene/view)")

        return tasks

def generate_one_task_threadsafe(output_dir: str, seed: int, index: int, scene: str):
    """在线程中生成一个任务。每个线程内部创建独立的 Controller，避免共享状态。
    该版本接收预分配的唯一 scene，确保多线程多任务时不重复场景。
    返回任务字典或 None。
    """
    gen = None
    try:
        gen = TaskGenerator(output_dir=output_dir, seed=seed)
        print(f"\n[Thread-{index}] Generating task with seed={seed} for scene={scene}...")

        # 加载指定的唯一场景
        try:
            event = gen.controller.reset(scene=scene)
        except Exception as e:
            print(f"[Thread-{index}] Exception loading scene {scene}: {e}")
            return None
        if not event.metadata.get("lastActionSuccess", False):
            print(f"[Thread-{index}] Failed to load scene {scene}: {event.metadata.get('errorMessage')}")
            return None
        gen.current_scene = scene

        viewpoint = gen.find_good_viewpoint()
        if not viewpoint:
            print(f"[Thread-{index}] No valid viewpoint found.")
            return None

        # 定位到最佳视角
        gen.controller.step(
            action="Teleport",
            position=viewpoint["position"],
            rotation=viewpoint["rotation"],
        )

        # 使用唯一前缀避免文件名冲突
        prefix = f"t{index}"
        before_image = gen.save_viewpoint_image("before", prefix=prefix)

        task = gen.generate_task_with_validation()
        if not task:
            print(f"[Thread-{index}] No valid task generated in this scene.")
            return None

        task["scene_id"] = scene
        if before_image:
            task["before_image"] = before_image

        after_image = gen.save_viewpoint_image("after", prefix=prefix)
        if after_image:
            task["after_image"] = after_image

        # 清理临时字段
        task.pop("target_obj_data", None)
        task.pop("ref_obj_data", None)

        print(f"[Thread-{index}] ✅ Task generated: {task['description']}")
        return task
    except Exception as e:
        print(f"[Thread-{index}] Exception: {e}")
        return None
    finally:
        try:
            if gen is not None:
                gen.controller.stop()
        except Exception:
            pass


# 使用示例
if __name__ == "__main__":
    import argparse

    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description="Generate rearrangement tasks dataset")
    parser.add_argument("--output_dir", type=str, default="./rearrangement_dataset",
                       help="Output directory for the dataset")
    parser.add_argument("--num-tasks", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--num_workers", type=int, default=32,
                       help="Number of worker threads to use for parallel generation")


    args = parser.parse_args()

    # 单线程或并行生成
    print("Starting task generation...")
    print(f"Random seed: {args.seed}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of tasks: {args.num_tasks}")
    print(f"Num workers: {args.num_workers}")

    tasks = []

    if args.num_workers and args.num_workers > 1:
        # 并行路径：预分配不重复场景，一一对应到线程任务
        os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)
        print("Running in parallel with ThreadPoolExecutor...")
        
        if len(all_scenes) < args.num_tasks:
            raise ValueError(f"Not enough unique scenes to generate {args.num_tasks} tasks (available={len(all_scenes)})")
        chosen_scenes = random.sample(all_scenes, args.num_tasks)
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = []
            for i, scene in enumerate(chosen_scenes):
                futures.append(executor.submit(
                    generate_one_task_threadsafe,
                    args.output_dir,
                    args.seed + i,
                    i,
                    scene,
                ))
            for fut in futures:
                res = fut.result()
                if res:
                    tasks.append(res)
    else:
        # 保持原有单线程路径
        generator = TaskGenerator(output_dir=args.output_dir, seed=args.seed)
        print(f"Images will be saved to: {generator.images_dir}")
        tasks = generator.generate_batch(args.num_tasks)
        generator.controller.stop()


    # 保存到JSON文件
    output_file = os.path.join(args.output_dir, "tasks.json")
    with open(output_file, "w") as f:
        json.dump(tasks, f, indent=2)

    print(f"\nGenerated {len(tasks)} tasks and saved to {output_file}")
    print(f"Images saved to: {os.path.join(args.output_dir, 'images')}")

    # 打印任务概览
    for i, task in enumerate(tasks, 1):
        before_img = task.get('before_image', 'N/A')
        after_img = task.get('after_image', 'N/A')
        print(f"Task {i}: {task['description']}")
        print(f"  Before: {before_img}, After: {after_img}")

