# Prompts for rearrangement two-phase workflow

# Format configurations for different reasoning styles
FORMAT_CONFIGS = {
    "free_think": {
        "description": "You should first give your thought process, and then your answer.",
        "format": "<think>...</think><answer>...</answer>",
        "example": """<think>I can see a box on the table that needs to be moved. I should first approach the table, then pick up the box and move it to the target location. After placing it correctly, I can finish the task.</think><answer>moveahead{action_sep}moveahead{action_sep}pickup{action_sep}moveright{action_sep}putdown{action_sep}done</answer>"""
    },
    "no_think": {
        "description": "You should provide only your answer.",
        "format": "<answer>...</answer>",
        "example": """<answer>moveahead{action_sep}moveahead{action_sep}pickup{action_sep}moveright{action_sep}putdown{action_sep}done</answer>"""
    },
    "grounding": {
        "description": "You should first give your thought process with your observation and reasoning, and finally your answer.\nThe observation should be described in detail about what you see in the environment.",
        "format": "<think><observation>...</observation><reasoning>...</reasoning></think><answer>...</answer>",
        "example": """<think><observation>I am in a living room. I can see a red box on the coffee table in front of me, and there's an empty shelf to my right where the box should be placed according to the target state.</observation><reasoning>I need to pick up the red box from the coffee table and place it on the shelf. First, I'll move forward to get closer to the table, then pick up the box, turn right toward the shelf, and place it there.</reasoning></think><answer>moveahead{action_sep}pickup{action_sep}rotateright{action_sep}moveahead{action_sep}putdown</answer>"""
    },
    "worldmodeling": {
        "description": "You should first give your thought process with reasoning and prediction of next state, then your answer.\nThe prediction should describe what you expect to see after your actions are executed.",
        "format": "<think><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>",
        "example": """<think><reasoning>I need to move the vase from the table to the shelf. I'll approach the table, pick up the vase, and move to the shelf.</reasoning><prediction>After executing these actions, I will be standing next to the shelf with the vase placed on it. The table will be empty, and the vase will be in its target position on the shelf.</prediction></think><answer>moveahead{action_sep}pickup{action_sep}rotateright{action_sep}moveahead{action_sep}putdown</answer>"""
    },
    "grounding_worldmodeling": {
        "description": "You should first give your thought process with your observation, reasoning, and prediction of next state, then your answer.\nBoth the observation and prediction should describe what you see or expect to see in the environment.",
        "format": "<think><observation>...</observation><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>",
        "example": """<think><observation>I am in a bedroom. There's a lamp on the nightstand that appears to be in the wrong position. According to my memory of the target state, this lamp should be on the desk near the window.</observation><reasoning>I need to pick up the lamp from the nightstand and move it to the desk. I'll approach the nightstand, carefully pick up the lamp, then navigate to the desk and place it there.</reasoning><prediction>After completing these actions, the lamp will be positioned on the desk near the window as it should be in the target state. The nightstand will be empty, and I'll be standing next to the desk.</prediction></think><answer>moveahead{action_sep}pickup{action_sep}rotateleft{action_sep}moveahead{action_sep}moveahead{action_sep}putdown</answer>"""
    }
}

WALKTHROUGH_SYSTEM_PROMPT = """You are a home robot performing the first phase of an object rearrangement task (Target State Observation).
In this phase, you need to carefully observe and record the target state of all movable objects in the environment.
Please explore the scene without modifying anything, and record detailed information about key movable/pickupable/openable objects:
- Object name (complete objectName)
- Object type (e.g., Box, Vase, Laptop, etc.)
- Precise position coordinates (x,y,z)
- Orientation angles (especially y-axis rotation)
- Openness state (if applicable)

Actions you can take to explore the environment:
moveahead: Move forward by 0.5 meter
moveback: Move backward by 0.5 meter
moveright: Move rightward by 0.5 meter
moveleft: Move leftward by 0.5 meter
rotateright: Rotate to the right by 90 degrees
rotateleft: Rotate to the left by 90 degrees
lookup: Tilt the camera upward by 30 degrees
lookdown: Tilt the camera downward by 30 degrees
done: Finish the walkthrough phase

After completing your observation, use the "done" action to finish the walkthrough phase:

Hints:
1. You can take multiple actions at a time. If you need to explore different areas, you can chain movement actions.
2. Use rotation and camera tilt to get a complete view of the environment.
3. Pay special attention to objects that can be moved, picked up, or opened/closed.
4. Record the exact positions and orientations as you will need this information in the next phase."""

UNSHUFFLE_SYSTEM_PROMPT = """You are a home robot performing the second phase of an object rearrangement task (Object Restoration).
Based on the target state memory recorded in the first phase, you need to restore objects that have been moved to their correct positions.
The objects in the current environment have been shuffled, and you need to:
1. Observe current object positions
2. Compare with the target state memory
3. Move objects one by one to their correct positions

Actions you can take:
moveahead: Move forward by 0.5 meter
moveback: Move backward by 0.5 meter
moveright: Move rightward by 0.5 meter
moveleft: Move leftward by 0.5 meter
rotateright: Rotate to the right by 90 degrees
rotateleft: Rotate to the left by 90 degrees
lookup: Tilt the camera upward by 30 degrees
lookdown: Tilt the camera downward by 30 degrees
pickup: Pick up the nearest pickupable object
putdown: Put down the currently held object
open: Open the nearest openable object
close: Close the nearest closeable object
done: Finish the unshuffle phase

Please execute step by step and explain your operations. After completing all object restoration, use the "done" action to finish the unshuffle phase.

Hints:
1. You can take multiple actions at a time for efficient movement.
2. When picking up objects, make sure you're close enough to them.
3. When putting down objects, ensure you're at the correct target location.
4. Use rotation and camera movement to locate objects and target positions.
5. If you get stuck, try different viewing angles or movement directions."""

# Format prompt function (adapted from navigation)
def format_prompt(**kwargs):
    """Format prompt based on the specified format type."""
    selected_format = kwargs.get("format", "grounding_worldmodeling")

    if selected_format in FORMAT_CONFIGS:
        config = FORMAT_CONFIGS[selected_format]
        return config["description"]
    else:
        return "Please provide your reasoning and answer."

# System prompt function
def system_prompt(**kwargs):
    selected_format = kwargs.get("format", "grounding_worldmodeling")
    phase = kwargs.get("phase", "walkthrough")

    if phase == "walkthrough":
        base_prompt = WALKTHROUGH_SYSTEM_PROMPT
        if selected_format == "grounding_worldmodeling":
            example = """
Example:
Round 1:
image_1
<think><observation>I am in a living room. I can see a coffee table in the center with a red box on it. There's a bookshelf to my right with some books and a vase. On the left side, there's a couch with a laptop on it. I need to explore more to see all movable objects.</observation><reasoning>I should move around to get a complete view of all objects in the room. Let me start by moving forward to get closer to the coffee table, then rotate to see other areas.</reasoning><prediction>After moving forward, I'll have a better view of the coffee table and the objects on it. Then I can rotate to see the bookshelf and couch more clearly.</prediction></think>
<answer>moveahead, moveahead, rotateright</answer>

Round 2:
Env_feedback: Last action is executed successfully.
image_2
<think><observation>Now I'm closer to the coffee table and can see the red box clearly. It's positioned at approximately the center of the table. After rotating right, I can see the bookshelf better. There's a blue vase on the second shelf and several books. I should continue exploring to see all areas.</observation><reasoning>I need to continue my exploration to record all movable objects. Let me rotate more to see the couch area and check if there are any other objects I missed.</reasoning><prediction>After rotating more, I'll be able to see the couch with the laptop and potentially other objects in the room.</prediction></think>
<answer>rotateright, rotateright</answer>

Round 3:
Env_feedback: Last action is executed successfully.
image_3
<think><observation>Perfect! Now I can see the couch clearly with a silver laptop on it. I've completed my exploration and can see all the main movable objects: a red box on the coffee table, a blue vase on the bookshelf, and a silver laptop on the couch. These appear to be the key objects I need to remember for the rearrangement task.</observation><reasoning>I have observed all the movable objects in their target positions. Now I should record this information and complete the walkthrough phase. I can use the done action to finish this phase efficiently.</reasoning><prediction>After using the done action, I will transition to the unshuffle phase where objects will be moved and I'll need to restore them based on my observations.</prediction></think>
<answer>done</answer>"""
        else:
            example = f"""
Format: {FORMAT_CONFIGS[selected_format]['format']}
Example: {FORMAT_CONFIGS[selected_format]['example']}
"""
    else:
        base_prompt = UNSHUFFLE_SYSTEM_PROMPT
        if selected_format == "grounding_worldmodeling":
            example = """
Example:
Round 1:
image_1
<think><observation>I can see the room has been rearranged. The red box that should be on the coffee table is now on the floor near the couch. The blue vase that should be on the bookshelf is on the coffee table. The laptop appears to be in the correct position on the couch. I need to restore the box and vase to their correct positions.</observation><reasoning>Based on my memory, the red box should be on the coffee table and the blue vase should be on the bookshelf. I should start by picking up the red box from the floor and placing it on the coffee table.</reasoning><prediction>After moving to the red box and picking it up, I'll be holding the box and can then move to the coffee table to place it in the correct position.</prediction></think>
<answer>moveright, moveright, pickup</answer>

Round 2:
Env_feedback: Last action is executed successfully.
image_2
<think><observation>Good! I've successfully picked up the red box. Now I'm holding it and need to move to the coffee table to place it in the correct position. I can see the coffee table ahead of me with the blue vase on it (which is in the wrong place).</observation><reasoning>I need to move to the coffee table and put down the red box. The coffee table currently has the blue vase on it, but there should be enough space for both objects temporarily.</reasoning><prediction>After moving to the coffee table and putting down the red box, I'll have the box in its correct position. Then I'll need to pick up the blue vase and move it to the bookshelf.</prediction></think>
<answer>moveleft, moveahead, putdown</answer>

Round 3:
Env_feedback: Last action is executed successfully.
image_3
<think><observation>Excellent! The red box is now on the coffee table in its correct position. I can see the blue vase is still on the coffee table next to the box. Now I need to pick up the blue vase and move it to the bookshelf where it belongs.</observation><reasoning>Now I need to pick up the blue vase and take it to the bookshelf. I should pick it up first, then navigate to the bookshelf.</reasoning><prediction>After picking up the vase and moving to the bookshelf, I'll be able to place the vase in its correct position, completing the rearrangement task.</prediction></think>
<answer>pickup, rotateright, moveahead, putdown</answer>

Round 4:
Env_feedback: Last action is executed successfully.
image_4
<think><observation>Perfect! I have successfully restored all objects to their correct positions. The red box is on the coffee table, the blue vase is on the bookshelf, and the laptop remains on the couch. All objects are now in their target positions as recorded during the walkthrough phase.</observation><reasoning>I have completed the rearrangement task successfully. All movable objects are now in their correct positions matching the target state from my memory. I can now finish the task using the done action.</reasoning><prediction>The task is complete. All objects are properly arranged according to the target state.</prediction></think>
<answer>done</answer>"""
        else:
            example = f"""
Format: {FORMAT_CONFIGS[selected_format]['format']}
Example: {FORMAT_CONFIGS[selected_format]['example']}
"""

    return f"{base_prompt}\n\n{example}"

# Observation templates
def init_observation_template(**kwargs):
    observation = kwargs.get("observation", "<image>")
    instruction = kwargs.get("instruction", "Observe the target state, then rearrange to match.")
    phase = kwargs.get("phase", "walkthrough")

    if phase == "walkthrough":
        phase_instruction = "Carefully observe the environment and record the target state of all movable objects."
    else:
        phase_instruction = "Restore objects to their correct positions based on your memory."

    return f"""[Initial Observation - {phase.upper()}]:
{observation}

Task: {instruction}
Phase Instructions: {phase_instruction}

Decide your next action(s)."""

def action_template(**kwargs):
    observation = kwargs.get("observation", "<image>")
    instruction = kwargs.get("instruction", "")
    valid_action = kwargs.get("valid_action", [])
    env_feedback = kwargs.get("env_feedback", "")
    reward = kwargs.get("reward", 0.0)
    done = kwargs.get("done", False)
    phase = kwargs.get("phase", "walkthrough")

    feedback_text = f"After your answer, the extracted valid action is {valid_action}.\nThe environment feedback is: {env_feedback}\nreward: {reward}\ndone: {done}"

    if phase == "walkthrough":
        phase_reminder = "Continue exploring and recording object states. Reply WALKTHROUGH_DONE when finished."
    else:
        phase_reminder = "Continue restoring objects to target positions. Reply UNSHUFFLE_DONE when finished."

    return f"""{feedback_text}
After that, the observation is:
{observation}

Phase: {phase.upper()}
Task: {instruction}
Reminder: {phase_reminder}

Decide your next action(s)."""

# format_prompt_generator function, similar to navigation
def format_prompt_generator(format_type):
    """
    Generates a prompt function for the specified rearrangement format type.
    This returned function creates the per-turn instruction for the LLM.
    """
    def prompt_function(**kwargs):
        """
        Generate a prompt for the specified format for the rearrangement task.

        Args:
            max_actions_per_step (int): Max actions. Defaults to 5.
            action_sep (str): Separator. Defaults to ','.
            add_example (bool): Whether to add an example. Defaults to True.

        Returns:
            str: The formatted prompt.
        """
        # Defaults suitable for the rearrangement task
        max_actions_per_step = kwargs.get("max_actions_per_step", 5)
        action_sep = kwargs.get("action_sep", ",")
        add_example = kwargs.get("add_example", True)

        if format_type not in FORMAT_CONFIGS:
            raise ValueError(f"Unknown format_type: {format_type}")
        config = FORMAT_CONFIGS[format_type]

        base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by '{action_sep}'.
{config["description"]}"""

        if "additional_info" in config:
            base_prompt += f"\n{config['additional_info']}"

        base_prompt += f"""
Your response should be in the format of:
{config["format"]}"""

        if add_example:
            example_text = config["example"].format(action_sep=action_sep)
            return base_prompt + '\n' + f"e.g. {example_text}"

        return base_prompt

    return prompt_function

# format_prompt dictionary
format_prompt = {
    ft: format_prompt_generator(ft)
    for ft in FORMAT_CONFIGS
}