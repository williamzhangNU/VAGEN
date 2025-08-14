from dataclasses import dataclass, fields, field
from vagen.env.base.base_env_config import BaseEnvConfig
import genesis as gs

@dataclass
class MentalRotationEnvConfig(BaseEnvConfig):
    env_name: str = "mental-rotation"
    render_mode: str = "vision" 
    image_placeholder: str = "<image>"
    target_image_placeholder: str = "<target_image>"
    max_actions_per_step: int = 1
    prompt_format: str = "no_think"  # TODO

    n_parallel_envs: int = 1
    parallel_env_spacing: float = 10.0

    max_steps: int = 5
    rotate_granularity: int = 90

    resolution: tuple[int, int] = (1280, 960)
    device: str = "cpu" # "cpu" or "cuda"
    fov: int = 30

    renderer: gs.renderers.RendererOptions = field(default_factory=gs.renderers.Rasterizer)
    options: str = "fast"

    format_reward: float = 0.2
    success_reward: float = 1.0

    def config_id(self) -> str:
        id_fields = [
            "device",
        ]
        id_str = ",".join(
            [f"{field.name}={getattr(self, field.name)}" for field in fields(self) if field.name in id_fields]
        )
        return f"MentalRotationEnvConfig({id_str})"



viewer_options = gs.options.ViewerOptions(
    res           = (1280, 960),
    camera_pos    = (3.5, 0.0, 2.5),
    camera_lookat = (0.0, 0.0, 0.5),
    camera_fov    = 40,
    max_FPS       = 60,
)

vis_options = gs.options.VisOptions(
    show_world_frame = True, # visualize the coordinate frame of `world` at its origin
    world_frame_size = 1.5, # length of the world frame in meter
    show_link_frame  = True, # do not visualize coordinate frames of entity links
    show_cameras     = False, # do not visualize mesh and frustum of the cameras added
    # plane_reflection = True, # turn on plane reflection
    ambient_light    = (0.1, 0.1, 0.1), # ambient light setting
)

rigid_opts = gs.options.RigidOptions(
    dt=None,  # Inherit from SimOptions; set to 0.0 in SimOptions if needed for static scenes.
    gravity=None,  # Inherit; no gravity needed for static renders.
    enable_collision=False,  # Disable to skip collision kernels.
    enable_joint_limit=False,  # Disable if no joints.
    enable_self_collision=False,  # Already default, but explicit for clarity.
    max_collision_pairs=1,  # Minimal value since collisions are disabled.
    integrator=gs.integrator.Euler,  # Simplest integrator for faster compile.
    IK_max_targets=1,  # Minimal targets to reduce memory/kernels.
    constraint_solver=gs.constraint_solver.CG,  # Default, efficient for minimal use.
    iterations=1,  # Minimal iterations to speed solver kernels.
    tolerance=1e-3,  # Looser tolerance for quicker convergence.
    ls_iterations=1,  # Minimal line search.
    ls_tolerance=0.1,  # Looser for speed.
    sparse_solve=False,  # Disable sparsity exploitation.
    contact_resolve_time=None,  # No need if collisions disabled.
    use_contact_island=False,  # Disable for simplicity.
    use_hibernation=False,  # Disable to avoid extra checks.
    hibernation_thresh_vel=0.001,  # Irrelevant but default.
    hibernation_thresh_acc=0.01   # Irrelevant but default.
)

options_map = {
    "fast": {
        "viewer_options": viewer_options,
        "vis_options": vis_options,
        "rigid_options": rigid_opts,
    },
    "physics": {
        "viewer_options": viewer_options,
        "vis_options": viewer_options,
        "rigid_options": None,
    }
}

if __name__ == "__main__":
    config = MentalRotationEnvConfig()
    print(config.config_id())