def system_prompt(**kwargs):
    return (
        "You are tasked with a 3D mental rotation challenge. "
        "You will be shown an image of a 3D object and need to rotate it to match a target orientation.\n\n"
        "Available actions:\n"
        "- Rotate around X-axis: x90, x180, x270, x-90, x-180, x-270\n"
        "- Rotate around Y-axis: y90, y180, y270, y-90, y-180, y-270\n"
        "- Rotate around Z-axis: z90, z180, z270, z-90, z-180, z-270\n\n"
        "The numbers represent degrees (positive = clockwise, negative = counterclockwise).\n"
        "Respond with your chosen action enclosed in <answer>action</answer> tags.\n"
        "For example: <answer>x90</answer> or <answer>y-180</answer>"
    )


def init_observation_template(**kwargs):
    observation = kwargs.get("img_str", "[an image of a 3D object]")
    target_observation = kwargs.get("target_img_str", "[target image]")
    valid_actions = kwargs.get("valid_actions", [])
    
    actions_str = ", ".join(valid_actions) if valid_actions else "x90, x180, x270, x-90, x-180, x-270, y90, y180, y270, y-90, y-180, y-270, z90, z180, z270, z-90, z-180, z-270"
    
    return f"""[Mental Rotation Task - Initial State]

Current Object: {observation}
Target Object: {target_observation}

Your goal is to rotate the object from its current orientation to match the target orientation shown in the target image.

Available Actions: {actions_str}

Compare the current and target images carefully, then decide your first rotation action."""


def action_template(**kwargs):
    observation = kwargs.get("img_str", "[the updated image after your rotation]")
    target_observation = kwargs.get("target_img_str", "[target image]")
    last_action = kwargs.get("last_action", "Unknown")
    step_count = kwargs.get("step_count", 0)
    
    return f"""[Step {step_count} Result]

Your last action: {last_action}
Current Object: {observation}
Target Object: {target_observation}

Compare your current result with the target image. Continue rotating to match the target, or you may have already reached it.
Decide your next action."""


if __name__ == "__main__":
    # 测试prompt模板
    print("=== System Prompt ===")
    print(system_prompt())
    print("\n=== Initial Observation ===")
    print(init_observation_template(
        img_str="<image>",
        target_img_str="<target_image>",
        valid_actions=["x90", "x180", "x270", "y90", "y180", "y270", "z90", "z180", "z270"]
    ))
    print("\n=== Action Template ===")
    print(action_template(
        img_str="<image>",
        target_img_str="<target_image>",
        last_action="x90",
        step_count=1
    )) 