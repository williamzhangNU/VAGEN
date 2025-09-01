import ai2thor.controller
from ai2thor.platform import CloudRendering
import random
import json
import numpy as np
import os
from PIL import Image
import re

from concurrent.futures import ThreadPoolExecutor, as_completed

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
            renderInstanceSegmentation=True,
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

        # 定义物体类型和空间关系
        # self.movable_objects = ["Vase", "Book", "Mug", "Apple", "Bread", "Plate", "Bowl", "CreditCard", "KeyChain"]
        # self.reference_objects = ["Sofa", "Chair", "ArmChair", "Bed", "Dresser", "Desk", "DiningTable", "CoffeeTable"]
        # 只允许地面到地面的移动，移除"on"关系
        self.spatial_relations = ["behind", "in front of", "to the left of", "to the right of"]

    def use_regular_scene(self):
        """使用常规AI2Thor场景而不是ProcTHOR"""
        # 扩展的场景列表 - 包含厨房、客厅、卧室、浴室场景
        all_scenes = [
            # 厨房场景 (FloorPlan1-30)
            # "FloorPlan1", "FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5",
            # "FloorPlan6", "FloorPlan7", "FloorPlan8", "FloorPlan9", "FloorPlan10",
            # "FloorPlan11", "FloorPlan12", "FloorPlan13", "FloorPlan14", "FloorPlan15",
            # "FloorPlan16", "FloorPlan17", "FloorPlan18", "FloorPlan19", "FloorPlan20",
            # "FloorPlan21", "FloorPlan22", "FloorPlan23", "FloorPlan24", "FloorPlan25",
            # "FloorPlan26", "FloorPlan27", "FloorPlan28", "FloorPlan29", "FloorPlan30",

            # 客厅场景 (FloorPlan201-230)
            # "FloorPlan205",
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

        # 只随机选择一个场景进行尝试
        scene = random.choice(all_scenes)
        try:
            event = self.controller.reset(scene=scene)
            if event.metadata["lastActionSuccess"]:
                print(f"Successfully loaded scene: {scene}")
                self.current_scene = scene
                return True
            else:
                print(f"Failed to load scene {scene}: {event.metadata.get('errorMessage')}")
                return False
        except Exception as e:
            print(f"Exception loading scene {scene}: {e}")
            return False

    def generate_scene(self):
        """生成场景 - 使用常规AI2Thor场景"""
        return self.use_regular_scene()



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

    def _build_complete_object_poses_from_metadata(self, override_poses):
        """构建完整的物体位置列表，参考用户提供的函数"""
        meta_objs = self.controller.last_event.metadata.get("objects", [])
        # Map desired overrides by name
        overrides = {}
        for p in (override_poses or []):
            key = p.get("name") or p.get("objectName")
            if key is not None:
                overrides[key] = {"position": p.get("position", {}), "rotation": p.get("rotation", {})}

        full_list = []
        for obj in meta_objs:
            if obj['pickupable'] or obj['moveable']:
                name = obj.get("name")
                desired = overrides.get(name)
                pos = desired["position"] if desired else obj.get("position", {})
                rot = desired["rotation"] if desired else obj.get("rotation", {})
                full_list.append({
                    "objectName": obj.get("name"),
                    "position": pos,
                    "rotation": rot,
                })

        return full_list

    def find_surface_for_placement(self, target_position):
        """找到目标位置下方的表面物体"""
        # 获取当前场景中的所有物体
        event = self.controller.step("Pass")
        objects = event.metadata["objects"]

        # 寻找目标位置下方的表面
        best_surface = None
        min_distance = float('inf')

        for obj in objects:
            if not obj.get("receptacle", False):
                continue

            obj_pos = obj["position"]
            obj_bounds = self.get_object_bounds(obj)

            if not obj_bounds:
                continue

            # 检查目标位置是否在物体表面上方
            if (obj_bounds["min_x"] <= target_position["x"] <= obj_bounds["max_x"] and
                obj_bounds["min_z"] <= target_position["z"] <= obj_bounds["max_z"] and
                obj_bounds["max_y"] <= target_position["y"]):

                distance = target_position["y"] - obj_bounds["max_y"]
                if distance < min_distance:
                    min_distance = distance
                    best_surface = obj

        # 如果没找到合适的表面，默认是地板
        if best_surface is None:
            return "floor"
        else:
            return best_surface["objectType"].lower()

    def find_best_surface_position(self, ref_obj):
        """找到参照物体的最佳表面位置（如椅子座位）"""
        try:
            # 获取参照物体的详细信息
            obj_type = ref_obj["objectType"].lower()
            ref_bounds = self.get_object_bounds(ref_obj)
            ref_pos = ref_obj["position"]

            if not ref_bounds:
                return None

            # 针对不同类型的物体使用不同的表面检测策略
            if obj_type in ["chair", "armchair"]:
                # 对于椅子，需要更精确地定位座位
                # 椅子的座位通常是最大的水平表面

                # 获取椅子的尺寸
                chair_width = ref_bounds["max_x"] - ref_bounds["min_x"]
                chair_depth = ref_bounds["max_z"] - ref_bounds["min_z"]
                chair_height = ref_bounds["max_y"] - ref_bounds["min_y"]

                # 座位高度通常在椅子总高度的45%-65%之间
                # 这是基于真实椅子的比例
                seat_height_ratio = 0.55  # 座位高度比例
                seat_height = ref_bounds["min_y"] + chair_height * seat_height_ratio

                # 座位位置：椅子中心，但稍微向前（座位通常不在椅子的几何中心）
                seat_x = ref_pos["x"]
                seat_z = ref_pos["z"] + chair_depth * 0.05  # 稍微向前5%

                # 确保座位位置在椅子边界内
                seat_x = max(ref_bounds["min_x"] + chair_width * 0.1,
                           min(ref_bounds["max_x"] - chair_width * 0.1, seat_x))
                seat_z = max(ref_bounds["min_z"] + chair_depth * 0.1,
                           min(ref_bounds["max_z"] - chair_depth * 0.1, seat_z))

                return {
                    "x": seat_x,
                    "y": seat_height,
                    "z": seat_z
                }

            elif obj_type in ["table", "diningtable", "coffeetable", "desk"]:
                # 对于桌子，使用桌面中心
                return {
                    "x": ref_pos["x"],
                    "y": ref_bounds["max_y"],
                    "z": ref_pos["z"]
                }

            elif obj_type in ["bed"]:
                # 对于床，使用床面中心
                return {
                    "x": ref_pos["x"],
                    "y": ref_bounds["max_y"],
                    "z": ref_pos["z"]
                }

            elif obj_type in ["stoveknob", "stoveburner"]:
                # 对于炉灶相关物体，使用顶部中心
                return {
                    "x": ref_pos["x"],
                    "y": ref_bounds["max_y"],
                    "z": ref_pos["z"]
                }

            elif obj_type in ["sinkbasin", "sink"]:
                # 对于水槽，使用边缘位置
                return {
                    "x": ref_pos["x"],
                    "y": ref_bounds["max_y"],
                    "z": ref_pos["z"]
                }

            else:
                # 对于其他物体，使用顶部中心
                return {
                    "x": ref_pos["x"],
                    "y": ref_bounds["max_y"],
                    "z": ref_pos["z"]
                }

        except Exception as e:
            print(f"Error finding best surface position: {e}")
            return None

    def get_camera_direction_vectors(self):
        """获取相机的方向向量（基于当前相机旋转）"""
        import math

        # 获取相机旋转角度（Y轴旋转，单位：度）
        camera_rotation = self.controller.last_event.metadata["agent"]["rotation"]["y"]

        # 转换为弧度
        angle_rad = math.radians(camera_rotation)

        # 计算前方向量（相机朝向）
        forward_x = math.sin(angle_rad)
        forward_z = math.cos(angle_rad)

        # 计算右方向量（相机右侧）
        right_x = math.cos(angle_rad)
        right_z = -math.sin(angle_rad)

        return {
            "forward": {"x": forward_x, "z": forward_z},
            "right": {"x": right_x, "z": right_z},
            "backward": {"x": -forward_x, "z": -forward_z},
            "left": {"x": -right_x, "z": -right_z}
        }

    def calculate_target_position(self, target_obj, ref_obj, relation):
        """根据空间关系计算目标位置：沿关系方向搜索多个offset，
        找到第一个与场景中其他物体无碰撞且与原位置距离>2m的位置；再验证存在一条无碰撞的地面移动轨迹；
        成功时会将物体移动到目标位置并保持，返回 (target_pos, movement_path)。失败返回 None。
        """
        ref_pos = ref_obj.get("position", None)
        ref_bounds = self.get_object_bounds(ref_obj)
        target_original_pos = target_obj.get("position", None)
        if not ref_pos or not ref_bounds or not target_original_pos:
            return None

        # 场景对象用于碰撞/路径规划
        event = self.controller.step("Pass")
        all_objects = event.metadata.get("objects", [])

        # 获取相机方向向量
        directions = self.get_camera_direction_vectors()
        ground_height = 0.1  # 地面高度

        # 选择主方向
        if relation == "behind":
            base_dir = directions["forward"]
        elif relation == "in front of":
            base_dir = directions["backward"]
        elif relation == "to the left of":
            base_dir = directions["left"]
        elif relation == "to the right of":
            base_dir = directions["right"]
        else:
            return None

        # 候选偏移（米）
        offset_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0, 1.2, 1.5]

        # 目标物体原始位置
        orig_pos = target_obj.get("position", {}).copy()

        for d in offset_list:
            cand_x = ref_pos["x"] + base_dir["x"] * d
            cand_z = ref_pos["z"] + base_dir["z"] * d
            target_pos = {"x": cand_x, "y": ground_height, "z": cand_z}

            # 与原始位置的水平距离>2m
            dx = target_pos["x"] - orig_pos.get("x", 0)
            dz = target_pos["z"] - orig_pos.get("z", 0)
            distance_from_orig = (dx * dx + dz * dz) ** 0.5
            if distance_from_orig <= 2.0:
                continue

            # 终点碰撞检查
            if self.check_collision_with_objects(target_pos, target_obj, all_objects):
                print(f"  Target position would cause collision, skipping...")
                continue

            # 规划地面移动轨迹（xz）
            start_xy = {"x": orig_pos.get("x", 0), "y": ground_height, "z": orig_pos.get("z", 0)}
            path = self.plan_ground_path(start_xy, target_pos, target_obj, all_objects)
            if not path or len(path) < 2:
                print("  No collision-free ground path found, skipping...")
                continue

            # 实际移动到候选位置，验证可见性
            original_pos = target_obj["position"].copy()
            original_rot = target_obj["rotation"].copy()
            test_poses = [{
                "objectName": target_obj["name"],
                "position": target_pos,
                "rotation": original_rot,
            }]
            complete_poses = self._build_complete_object_poses_from_metadata(test_poses)
            move_event = self.controller.step(action="SetObjectPoses", objectPoses=complete_poses)
            assert move_event.metadata["lastActionSuccess"]

            check_event = self.controller.step("Pass")
            moved_obj = None
            for obj in check_event.metadata["objects"]:
                if obj.get("name") == target_obj.get("name"):
                    moved_obj = obj
                    break
            is_visible = moved_obj and self.is_object_visible(moved_obj)

            if is_visible:
                return (target_pos, path)
            else:
                # 若不可见，恢复原位
                restore_poses = [{
                    "objectName": target_obj["name"],
                    "position": original_pos,
                    "rotation": original_rot,
                }]
                complete_restore = self._build_complete_object_poses_from_metadata(restore_poses)
                self.controller.step(action="SetObjectPoses", objectPoses=complete_restore)
                print("  Target position is not visible, skipping...")

        return None

    def is_object_visible(self, obj, percent: float = None, save_filtered_path: str = '/home/zihanhuang/VAGEN/rearrangement_dataset/images/filtered.png'):
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

        # 仅渲染该物体，计算"无遮挡可见像素数"
        unoccluded_pixels = 0
        if obj_id:
            other_ids = []
            try:
                # 暂时隐藏除目标外的所有物体，以获得“无遮挡”视图
                meta_objs = event.metadata.get("objects", []) if hasattr(event, 'metadata') else []
                other_ids = [o.get("objectId") for o in meta_objs if o.get("objectId") and o.get("objectId") != obj_id]
                for oid in other_ids:
                    try:
                        self.controller.step(action="DisableObject", objectId=oid)
                    except Exception:
                        pass

                event_single = self.controller.step("Pass")

                # 可选地保存仅渲染该物体的图像（其他物体被隐藏）
                

                inst_masks_single = getattr(event_single, "instance_masks", None)
                if inst_masks_single and obj_id in inst_masks_single and inst_masks_single[obj_id] is not None:
                    try:
                        unoccluded_pixels = int(np.sum(inst_masks_single[obj_id]))
                    except Exception:
                        unoccluded_pixels = 0
            finally:
                # 恢复所有物体的可见性
                for oid in other_ids:
                    try:
                        self.controller.step(action="EnableObject", objectId=oid)
                    except Exception:
                        pass

        # 根据可见像素占无遮挡像素的比例判断
        if unoccluded_pixels > 0:
            visible_ratio_percent = (visible_pixels / float(unoccluded_pixels)) * 100.0
            if visible_ratio_percent < 80:
                print(f"Visible ratio {visible_ratio_percent:.2f}%")
                if save_filtered_path:
                    try:
                        dir_name = os.path.dirname(save_filtered_path)
                        if dir_name:
                            os.makedirs(dir_name, exist_ok=True)
                        img = Image.fromarray(event_single.frame)
                        img.save(save_filtered_path)
                    except Exception as e:
                        print(f"Failed to save filtered image to {save_filtered_path}: {e}")
            return visible_ratio_percent >= float(percent)
        else:
            return base_visible
    def get_object_bounds(self, obj):
        """获取物体的边界框"""
        if "axisAlignedBoundingBox" in obj:
            bbox = obj["axisAlignedBoundingBox"]
            if "cornerPoints" in bbox and bbox["cornerPoints"]:
                corners = bbox["cornerPoints"]
                # Handle different corner point formats
                try:
                    if isinstance(corners[0], dict):
                        # Format: [{"x": 1, "y": 2, "z": 3}, ...]
                        min_x = min(p["x"] for p in corners)
                        max_x = max(p["x"] for p in corners)
                        min_y = min(p["y"] for p in corners)
                        max_y = max(p["y"] for p in corners)
                        min_z = min(p["z"] for p in corners)
                        max_z = max(p["z"] for p in corners)
                    else:
                        # Format: [[x, y, z], ...]
                        min_x = min(p[0] for p in corners)
                        max_x = max(p[0] for p in corners)
                        min_y = min(p[1] for p in corners)
                        max_y = max(p[1] for p in corners)
                        min_z = min(p[2] for p in corners)
                        max_z = max(p[2] for p in corners)
                    return {
                        "min_x": min_x, "max_x": max_x,
                        "min_y": min_y, "max_y": max_y,
                        "min_z": min_z, "max_z": max_z
                    }
                except (KeyError, IndexError, TypeError) as e:
                    print(f"Error parsing bounding box for {obj.get('objectType', 'unknown')}: {e}")

        # Fallback: use object position with default size
        if "position" in obj:
            pos = obj["position"]
            size = 0.5  # Default size
            return {
                "min_x": pos["x"] - size,
                "max_x": pos["x"] + size,
                "min_y": pos["y"],
                "max_y": pos["y"] + size,
                "min_z": pos["z"] - size,
                "max_z": pos["z"] + size
            }
        return None

    def can_place_object(self, target_obj, ref_obj, relation):
        """检查是否可以根据空间关系放置物体"""
        # Check if target object is pickupable
        if not target_obj.get("pickupable", False):
            return False

        ref_bounds = self.get_object_bounds(ref_obj)
        if not ref_bounds:
            return False

        # 根据空间关系检查是否有足够空间
        margin = 0.5  # 安全边距

        if relation in ["behind", "in front of"]:
            return abs(ref_bounds["max_z"] - ref_bounds["min_z"]) > margin
        elif relation in ["to the left of", "to the right of"]:
            return abs(ref_bounds["max_x"] - ref_bounds["min_x"]) > margin

        return True

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
        max_ground_objects = -1
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

                # 统计当前视角的对象
                movable_objs, reference_objs, ground_count = self.get_visible_objects()

                # 仅保留 movable 和 reference 都 > 3 的视角
                if len(movable_objs) > 3 and len(reference_objs) > 3:
                    if ground_count > max_ground_objects:
                        max_ground_objects = ground_count
                        best_viewpoint = {
                            "position": {"x": pos["x"], "y": pos["y"], "z": pos["z"]},
                            "rotation": {"x": 0, "y": rotation_y, "z": 0},
                            "total_ground_objects": ground_count,
                            "movable_count": len(movable_objs),
                            "reference_count": len(reference_objs)
                        }
                        print(
                            f"New valid viewpoint: pos=({pos['x']:.2f},{pos['z']:.2f}), rot={rotation_y}°, "
                            f"movable={len(movable_objs)}, reference={len(reference_objs)}, total_ground={ground_count}"
                        )

                # 进度日志
                if candidate_count % 200 == 0:
                    print(
                        f"Progress: {candidate_count}/{total_candidates}, "
                        f"current best total_ground={max_ground_objects if max_ground_objects>=0 else 0}"
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
                f"reference={best_viewpoint['reference_count']}, total_ground={best_viewpoint['total_ground_objects']}"
            )
        else:
            print("No valid viewpoint (movable>3 and reference>3) found under 0.5m spacing.")

        return best_viewpoint

    def check_collision_with_objects(self, target_position, target_obj, all_objects):
        """检查目标位置是否与其他物体发生碰撞"""
        # 获取目标物体的边界框大小
        target_bounds = self.get_object_bounds(target_obj)
        if not target_bounds:
            return True  # 如果无法获取边界框，保守地认为有碰撞

        # 计算目标物体在新位置的边界框
        target_width = target_bounds["max_x"] - target_bounds["min_x"]
        target_depth = target_bounds["max_z"] - target_bounds["min_z"]
        target_height = target_bounds["max_y"] - target_bounds["min_y"]

        # 目标物体在新位置的边界框
        new_target_bounds = {
            "min_x": target_position["x"] - target_width / 2,
            "max_x": target_position["x"] + target_width / 2,
            "min_z": target_position["z"] - target_depth / 2,
            "max_z": target_position["z"] + target_depth / 2,
            "min_y": target_position["y"],
            "max_y": target_position["y"] + target_height
        }

        # 安全边距
        safety_margin = 0.1  # 10厘米安全距离

        # 检查与所有其他物体的碰撞
        for obj in all_objects:
            # 跳过目标物体本身（以 name 作为唯一凭证）
            if obj.get("name") and target_obj.get("name") and obj["name"] == target_obj["name"]:
                continue

            # 只检查可见的物体
            if not self.is_object_visible(obj):
                continue

            # 忽略地板（objectType 可能形如 Floor555），与之不进行碰撞计算
            obj_type_lower = str(obj.get("objectType", "")).lower()
            if re.match(r"^floor[0-9]*$", obj_type_lower):
                continue

            obj_bounds = self.get_object_bounds(obj)
            if not obj_bounds:
                continue

            # 检查3D边界框重叠（加上安全边距）
            if (new_target_bounds["max_x"] + safety_margin > obj_bounds["min_x"] and
                new_target_bounds["min_x"] - safety_margin < obj_bounds["max_x"] and
                new_target_bounds["max_z"] + safety_margin > obj_bounds["min_z"] and
                new_target_bounds["min_z"] - safety_margin < obj_bounds["max_z"] and
                new_target_bounds["max_y"] + safety_margin > obj_bounds["min_y"] and
                new_target_bounds["min_y"] - safety_margin < obj_bounds["max_y"]):

                # print(f"  Collision detected with {obj['objectType']} at position ({obj['position']['x']:.2f}, {obj['position']['z']:.2f})")
                return True  # 发现碰撞

        return False  # 没有碰撞

    def plan_ground_path(self, start_pos, goal_pos, target_obj, all_objects, grid_size=0.2, padding=2.0):
        """在地面上为目标物体规划一条无碰撞轨迹（xz 平面上的路径）。
        - 使用简单的二维A*（8邻接）
        - 使用目标物体的占地尺寸考虑碰撞（不添加额外安全边距）
        - 返回路径点列表（包含起点和终点），每个点为 {x,y,z}
        - 返回 None 表示无可行路径
        """
        import heapq
        import math

        # 估算目标物体在 xz 平面的占地尺寸（半宽半深）
        tb = self.get_object_bounds(target_obj) or {"min_x": 0, "max_x": 0, "min_z": 0, "max_z": 0}

        # 收集障碍（不包含目标自身、也不把地板当障碍），不做额外膨胀
        obstacles = []
        for obj in (all_objects or []):
            if obj.get("name") == target_obj.get("name"):
                continue
            obj_type_lower = str(obj.get("objectType", "")).lower()
            if re.match(r"^floor[0-9]*$", obj_type_lower):
                continue
            ob = self.get_object_bounds(obj)
            if not ob:
                continue
            # 直接使用障碍物自身边界（不膨胀）
            obstacles.append({
                "min_x": ob["min_x"],
                "max_x": ob["max_x"],
                "min_z": ob["min_z"],
                "max_z": ob["max_z"],
            })

        # 规划区域边界（围绕起终点加 padding）
        min_x = min(start_pos["x"], goal_pos["x"]) - padding
        max_x = max(start_pos["x"], goal_pos["x"]) + padding
        min_z = min(start_pos["z"], goal_pos["z"]) - padding
        max_z = max(start_pos["z"], goal_pos["z"]) + padding

        def in_bounds(x, z):
            return (min_x <= x <= max_x) and (min_z <= z <= max_z)

        def blocked(x, z):
            # 判断点是否落在任一膨胀障碍之内
            for ob in obstacles:
                if ob["min_x"] <= x <= ob["max_x"] and ob["min_z"] <= z <= ob["max_z"]:
                    return True
            return False

        # 网格坐标映射
        def world_to_grid(x, z):
            gx = int(round((x - min_x) / grid_size))
            gz = int(round((z - min_z) / grid_size))
            return gx, gz

        def grid_to_world(gx, gz):
            x = min_x + gx * grid_size
            z = min_z + gz * grid_size
            return x, z

        start_g = world_to_grid(start_pos["x"], start_pos["z"])
        goal_g = world_to_grid(goal_pos["x"], goal_pos["z"])

        # 起终点可行性初检
        sx, sz = grid_to_world(*start_g)
        gx, gz = grid_to_world(*goal_g)
        if not in_bounds(sx, sz) or not in_bounds(gx, gz):
            return None
        if blocked(sx, sz) or blocked(gx, gz):
            return None

        # A* 搜索
        def h(a, b):
            (x1, z1), (x2, z2) = a, b
            return math.hypot(x1 - x2, z1 - z2)

        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),  # 4邻
                     (-1, -1), (-1, 1), (1, -1), (1, 1)]  # 对角 8邻

        open_heap = []
        heapq.heappush(open_heap, (0, start_g))
        came_from = {start_g: None}
        g_score = {start_g: 0.0}

        max_nodes = 20000  # 防止无限膨胀
        expansions = 0

        while open_heap and expansions < max_nodes:
            _, current = heapq.heappop(open_heap)
            if current == goal_g:
                break
            cx, cz = current
            for dx, dz in neighbors:
                nx, nz = cx + dx, cz + dz
                wx, wz = grid_to_world(nx, nz)
                if not in_bounds(wx, wz) or blocked(wx, wz):
                    continue
                step_cost = math.hypot(dx, dz) * grid_size
                tentative_g = g_score[current] + step_cost
                if (nx, nz) not in g_score or tentative_g < g_score[(nx, nz)]:
                    g_score[(nx, nz)] = tentative_g
                    priority = tentative_g + h((nx, nz), goal_g)
                    heapq.heappush(open_heap, (priority, (nx, nz)))
                    came_from[(nx, nz)] = current
            expansions += 1

        if goal_g not in came_from:
            return None

        # 回溯路径
        path_g = []
        cur = goal_g
        while cur is not None:
            path_g.append(cur)
            cur = came_from[cur]
        path_g.reverse()

        # 转换为世界坐标并设置到地面高度
        ground_y = 0.1
        path = []
        for gx_i, gz_i in path_g:
            wx, wz = grid_to_world(gx_i, gz_i)
            path.append({"x": wx, "y": ground_y, "z": wz})

        # 可选：路径压缩（保留转折点）
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


    def get_object_size(self, obj):
        """估算物体的大小（基于边界框）"""
        bounds = self.get_object_bounds(obj)
        if not bounds:
            return float('inf')  # 如果无法获取边界框，认为很大

        # 计算物体的体积（长x宽x高的近似）
        width = bounds["max_x"] - bounds["min_x"]
        depth = bounds["max_z"] - bounds["min_z"]

        # 使用面积作为大小的度量
        size = width * depth
        return size

    def get_distance_between_objects(self, obj1, obj2):
        """计算两个物体之间的距离"""
        pos1 = obj1["position"]
        pos2 = obj2["position"]

        dx = pos1["x"] - pos2["x"]
        dz = pos1["z"] - pos2["z"]

        return (dx**2 + dz**2)**0.5

    def is_object_on_ground(self, obj):
        """检查物体是否在地面上或低矮表面上（更宽松的地面检测）"""
        ground_height = 0.1  # 地面高度
        obj_y = obj["position"]["y"]

        # 考虑物体的边界框，检查物体底部是否接近地面
        bounds = self.get_object_bounds(obj)
        if bounds:
            bottom_y = bounds["min_y"]
            # 检查物体底部是否在地面附近（包括低矮表面）
            is_on_ground = bottom_y <= (ground_height)
            if obj.get("moveable", False):  # 对于可移动物体，打印调试信息
                print(f"    {obj['objectType']}: bottom_y={bottom_y:.2f}, ground={ground_height}, on_ground={is_on_ground}")
            return is_on_ground
        else:
            # 如果无法获取边界，使用物体中心位置判断
            is_on_ground = obj_y <= (ground_height)
            if obj.get("moveable", False):  # 对于可移动物体，打印调试信息
                print(f"    {obj['objectType']}: center_y={obj_y:.2f}, ground={ground_height}, on_ground={is_on_ground}")
            return is_on_ground

    def get_all_ground_objects(self):
        """获取当前视角中所有在地面上的可见物体（包括可移动和不可移动物体）"""
        event = self.controller.step("Pass")
        objects = event.metadata["objects"]

        ground_objects = []
        print("Checking all objects on ground (including furniture and items):")

        for obj in objects:
            if not self.is_object_visible(obj):
                continue

            # 检查物体是否在地面上
            if not self.is_object_on_ground(obj):
                continue

            ground_objects.append(obj)
            obj_type = obj["objectType"]
            pickupable = obj.get("pickupable", False)
            print(f"  Found ground object: {obj_type} (pickupable: {pickupable})")

        print(f"Total ground objects: {len(ground_objects)}")
        return ground_objects

    def get_visible_objects(self):
        """获取当前视角中可见的物体，只选择地面上的物体"""
        event = self.controller.step("Pass")
        objects = event.metadata["objects"]

        # 选择可移动的物体和参照物体，只选择地面上的
        movable = []
        all_ground_objects = []
        ground_objects_count = 0

        print("Checking objects on ground:")
        for obj in objects:
            # 检查物体是否在地面上
            if not self.is_object_on_ground(obj):
                continue

            ground_objects_count += 1
            obj_type = obj["objectType"]
            all_ground_objects.append(obj)

            if not self.is_object_visible(obj, 30):
                continue

            
            # Check for movable objects (包含可移动或可拾取的物体，且在地面上)
            # 仍然排除包含 sofa/table 的类型名
            is_movable = obj.get("moveable", False) or obj.get("pickupable", False)
            if is_movable:
                obj_type_lower = obj_type.lower()
                if "sofa" not in obj_type_lower and "table" not in obj_type_lower:
                    movable.append(obj)
                    print(f"  Added movable: {obj_type}")
                else:
                    print(f"  Skipped movable (contains sofa/table): {obj_type}")
            else:
                print(f"  Found ground object: {obj_type}")

        # 所有地面物体都可以作为参照物体（除了目标物体本身）
        # 这个筛选会在任务生成时进行，这里先保留所有地面物体
        reference = all_ground_objects.copy()

        # 随机选择参照物体，不按大小排序
        if reference:
            import random
            # 随机打乱参照物体列表
            random.shuffle(reference)
            # 保留所有参照物体，或者限制数量以避免过多
            reference = reference[:min(15, len(reference))]  # 最多保留15个参照物体

        # Debug output
        print(f"Total ground objects: {ground_objects_count}")
        # print(f"Found {len(movable)} movable objects on ground: {[obj['objectType'] for obj in movable]}")
        # print(f"Found {len(reference)} reference objects on ground: {[obj['objectType'] for obj in reference]}")

        return movable, reference, ground_objects_count

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
        """生成经过验证的任务（假定视角已在外部设置并已 Teleport）"""
        # 直接基于当前相机视角获取可见物体
        movable_objs, reference_objs, _ = self.get_visible_objects()
        if not movable_objs or not reference_objs:
            return None

        # 过滤出唯一类型的物体（确保每种类型只有一个）
        unique_movable_objs = self.filter_unique_objects(movable_objs)
        unique_reference_objs = self.filter_unique_objects(reference_objs)

        print(f"unique movable objects: {len(unique_movable_objs)} {[obj['objectType'] for obj in unique_movable_objs]}")
        print(f"unique reference objects: {len(unique_reference_objs)} {[obj['objectType'] for obj in unique_reference_objs]}")

        # 智能重试机制：先尝试不同方位词，再更换参照物体
        if not unique_movable_objs or not unique_reference_objs:
            return None

        # 先选择目标物体：按估计体积从小到大排序，优先尝试体积小的
        unique_movable_objs.sort(key=lambda o: self.get_object_size(o))
        target_obj = unique_movable_objs[0]
        print(f"  Selected target object (smallest): {target_obj['objectType']}")

        # 选择与目标物体有一定距离的参照物体
        suitable_references = []
        min_distance = 1.0  # 最小距离1米
        max_distance = 6.0  # 最大距离3米

        for ref_obj in unique_reference_objs:
            distance = self.get_distance_between_objects(target_obj, ref_obj)
            if min_distance <= distance <= max_distance:
                suitable_references.append(ref_obj)

        # 如果没有合适距离的物体，放宽条件
        if not suitable_references:
            suitable_references = [obj for obj in unique_reference_objs
                                 if self.get_distance_between_objects(target_obj, obj) >= 0.5]

        if not suitable_references:
            print(f"  No suitable reference objects within distance for {target_obj['objectType']}")
            return None

        # 随机打乱参照物体和方位词的顺序
        random.shuffle(suitable_references)
        spatial_relations = list(self.spatial_relations)
        random.shuffle(spatial_relations)

        print(f"  Found {len(suitable_references)} suitable reference objects")

        # 尝试所有参照物体和方位词的组合
        for ref_obj in suitable_references:
            print(f"    Trying reference object: {ref_obj['objectType']}")

            for relation in spatial_relations:
                print(f"      Trying spatial relation: {relation}")

                # 验证空间关系是否合理（包括碰撞检测与路径）；成功则保持移动
                result = self.calculate_target_position(target_obj, ref_obj, relation)
                if result:
                    target_pos, movement_path = result
                    print(f"      ✅ Success with {relation}!")

                    # 生成任务描述 - 不包含颜色和大小描述
                    task_description = f"Place the {target_obj['objectType'].lower()} {relation} the {ref_obj['objectType'].lower()}."

                    return {
                        "description": task_description,
                        "target_object": {
                            "name": target_obj.get("name"),
                            "type": target_obj.get("objectType"),
                            "position": target_obj.get("position")
                        },
                        "reference_object": {
                            "name": ref_obj.get("name"),
                            "type": ref_obj.get("objectType"),
                            "position": ref_obj.get("position")
                        },
                        "spatial_relation": relation,
                        "scene_metadata": {
                            "visible_movable_objects": len(movable_objs),
                            "visible_reference_objects": len(reference_objs)
                        },
                        "movement_path": movement_path,  # 地面上的无碰撞轨迹（含起终点）
                        "target_obj_data": target_obj,  # 保存完整的目标物体数据（以 name 作为唯一凭证）
                        "ref_obj_data": ref_obj  # 保存完整的参照物体数据（以 name 作为唯一凭证）
                    }
                else:
                    print(f"      ❌ Failed with {relation} (collision, visibility, path, or other issue)")

            print(f"      All spatial relations failed for {ref_obj['objectType']}")

        print(f"    All reference objects and spatial relations failed for {target_obj['objectType']}")
        return None

    def generate_batch(self, num_tasks=10):
        """生成一批高质量的任务，每个任务使用独立的场景"""
        tasks = []

        for task_num in range(num_tasks):
            print(f"\nGenerating task {task_num + 1}/{num_tasks}...")

            # 每个任务只随机选择一个场景，若该场景无有效任务则返回 None
            task = None

            print("  Selecting a single random scene...")
            success = self.generate_scene()

            if not success:
                print("  Failed to load the randomly selected scene. Returning None for this task.")
            else:
                # 在该场景中尝试生成任务（只尝试一次）
                # 先寻找最佳视角，并显式 Teleport 到该视角，再保存移动前的图片
                viewpoint = self.find_good_viewpoint()
                if not viewpoint:
                    print("  No valid viewpoint found in this selected scene. Returning None for this task.")
                    continue
                self.controller.step(
                    action="Teleport",
                    position=viewpoint["position"],
                    rotation=viewpoint["rotation"]
                )
                before_image = self.save_viewpoint_image("before")

                task = self.generate_task_with_validation()
                if task:
                    task["scene_id"] = f"scene_{task_num + 1}"

                    # 如果有 before 图片则记录
                    if before_image:
                        task["before_image"] = before_image

                    # generate_task_with_validation 内部已完成最终移动与验证
                    # 此处仅保存移动后图片（以 FloorPlan 名称命名）
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

def generate_one_task_threadsafe(output_dir: str, seed: int, index: int):
    """在线程中生成一个任务。每个线程内部创建独立的 Controller，避免共享状态。
    返回任务字典或 None。
    """
    gen = None
    try:
        gen = TaskGenerator(output_dir=output_dir, seed=seed)
        print(f"\n[Thread-{index}] Generating task with seed={seed}...")

        task = None
        success = gen.generate_scene()
        if not success:
            print(f"[Thread-{index}] Failed to load a scene.")
            return None

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

        task["scene_id"] = f"scene_{index + 1}"
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
    parser.add_argument("--num_tasks", type=int, default=1)
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
        # 并行路径：每个线程独立创建 TaskGenerator 和 Controller
        os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)
        print("Running in parallel with ThreadPoolExecutor...")
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = []
            for i in range(args.num_tasks):
                futures.append(executor.submit(
                    generate_one_task_threadsafe,
                    args.output_dir,
                    args.seed + i,
                    i
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

