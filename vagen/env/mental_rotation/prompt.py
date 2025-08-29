FORMAT_CONFIGS = {
    "free_think": {
        "description": "You should first give your thought process, and then your answer.",
        "format": "<think>...</think><answer>...</answer>",
        "example": """<think>I can see the front of the shape which indicated by the small blocks is facing to the position direction of x-axis in original figure, is rotated to the right in the target figure. To achieve this transformation, I need to rotate the shape around z axis for 90 degrees. </think><answer>z90</answer>"""
    },
    "no_think": {
        "description": "You should provide only your answer.",
        "format": "<answer>...</answer>",
        "example": """<answer>z90</answer>"""
    },
    # "grounding": {
    #     "description": "You should first give your thought process with your observation and reasoning, and finally your answer.\nThe observation should be described in detail about what you see in the environment.",
    #     "format": "<think><observation>...</observation><reasoning>...</reasoning></think><answer>...</answer>",
    #     "example": """<think><observation>I am in a living room. There is a couch to my left, a TV in front of me, and a doorway to the kitchen on my right. The target object, a vase, appears to be on a shelf near the kitchen doorway.</observation><reasoning>I need to move toward the kitchen doorway to reach the vase. I'll move forward to get closer to the center of the room, then turn right and move toward the kitchen.</reasoning></think><answer>moveahead{action_sep}moveahead{action_sep}rotateright{action_sep}moveahead{action_sep}moveahead</answer>"""
    # },
    # "worldmodeling": {
    #     "description": "You should first give your thought process with reasoning and prediction of next state,  then your answer.\nThe prediction should describe what you expect to see after your actions are executed.",
    #     "format": "<think><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>",
    #     "example": """<think><reasoning>I can see the kitchen doorway to my right, and I need to go there to find the refrigerator. I'll turn right and move forward.</reasoning><prediction>I am now in the kitchen doorway. In front of me is the kitchen counter with a sink. To the left I can see a refrigerator against the wall. There's a kitchen island in the center of the room.</prediction></think><answer>rotateright{action_sep}moveahead{action_sep}moveahead</answer>"""
    # },
    # "grounding_worldmodeling": {
    #     "description": "You should first give your thought process with the your observation, reasoning, and prediction of next state, then your answer.\nBoth the observation and prediction should describe what you see or expect to see in the environment.",
    #     "format": "<think><observation>...</observation><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>",
    #     "example": """<think><observation>I am at the entrance of a bedroom. There is a bed to the left, a desk with a lamp on the right, and a closet straight ahead. The target object, a book, appears to be on the desk.</observation><reasoning>I need to move toward the desk to reach the book. I'll turn right and move forward.</reasoning><prediction>I am now standing in front of the desk. The desk has a lamp, a computer, and several books on it. The target book is within reach on the right side of the desk.</prediction></think><answer>rotateright{action_sep}moveahead{action_sep}moveahead</answer>"""
    # }
}

# format_prompt_generator function, similar to your first (FrozenLake) example
def format_prompt_generator(format_type):
    """
    Generates a prompt function for the specified robot navigation format type.
    This returned function creates the per-turn instruction for the LLM.
    """
    def prompt_function(**kwargs):
        """
        Generate a prompt for the specified format for the robot navigation task.
        
        Args:
            max_actions_per_step (int): Max actions. Defaults to 5 (common for robot).
            action_sep (str): Separator. Defaults to ',' (common for robot).
            add_example (bool): Whether to add an example. Defaults to True.
            
        Returns:
            str: The formatted prompt.
        """

        add_example = kwargs.get("add_example", True)
        
        if format_type not in FORMAT_CONFIGS:
            raise ValueError(f"Unknown format_type: {format_type}")
        config = FORMAT_CONFIGS[format_type]
        
        base_prompt = f"""{config["description"]}"""
        
        if "additional_info" in config: # In case it's added to FORMAT_CONFIGS later
            base_prompt += f"\n{config['additional_info']}"
        
        base_prompt += f"""
Your response should be in the format of:
{config["format"]}"""
        
        if add_example:
            # The 'e.g.' is already part of the example string in this FORMAT_CONFIGS
            example_text = config["example"].format()
            return base_prompt + '\n' + f"e.g. {example_text}"
        
        return base_prompt
    
    return prompt_function


def system_prompt(**kwargs):
    example = "" # Default empty example
    # Internally uses kwargs.get("format"), as in your original code
    selected_format = kwargs.get("format", "free_think")
        
    base_prompt_text = """You are tasked with a 3D mental rotation challenge. You will be shown an image of a 3D object and a target orientation of the same object. Your goal is to rotate the object from its current orientation to match the target orientation shown in the target image.

Conventions:
1) Axes are indicated by colored arrows in the figure; the arrow direction is the positive direction.  
   - Red → +x  
   - Green → +y  
   - Blue → +z
2) The coordinate origin is located at the object’s centroid.
3) The object undergoes rotation about the axis that passes through its centroid.
4) Rotations follow the right-hand rule: when viewed from the positive direction of an axis, counterclockwise is considered positive.

Available actions:
- Rotate around X-axis: x90, x180, x270, x-90, x-180, x-270
- Rotate around Y-axis: y90, y180, y270, y-90, y-180, y-270
- Rotate around Z-axis: z90, z180, z270, z-90, z-180, z-270

The numbers represent degrees (positive = clockwise, negative = counterclockwise).
"""
    return base_prompt_text + '\n' + example

def init_observation_template(**kwargs):
    observation = kwargs.get("img_str", "[an image of a 3D object]")
    target_observation = kwargs.get("target_img_str", "[target image]")
    valid_actions = kwargs.get("valid_actions", [])
    
    actions_str = ", ".join(valid_actions) if valid_actions else "x90, x180, x270, x-90, x-180, x-270, y90, y180, y270, y-90, y-180, y-270, z90, z180, z270, z-90, z-180, z-270"
    
    return f"""[Initial State]
Current Orientation: {observation}
Target Orientation: {target_observation}
Instruction: Your goal is to rotate the object from its current orientation to match the target orientation shown in the target image.
Available Actions: {actions_str}
Decide your next action."""


def action_template(**kwargs):
    observation = kwargs.get("img_str", "No observation provided.")
    target_observation = kwargs.get("target_img_str", "No target image provided.")
    last_action = kwargs.get("last_action", "No valid action provided.")
    # reward = kwargs.get("reward", "No reward provided.")
    # done = kwargs.get("done", "No done status provided.")
    
    return f"""After your answer, the extracted valid action is {last_action}.
Your last action: {last_action}
After that, the observation is: {observation}
Target Orientation: {target_observation}
Instruction: Your goal is to rotate the object from its current orientation to match the target orientation shown in the target image.
Decide your next action."""

format_prompt = {
    ft: format_prompt_generator(ft) 
    for ft in FORMAT_CONFIGS  # Iterate directly over keys in FORMAT_CONFIGS
}

if __name__ == "__main__":
    print(system_prompt(
        format="free_think"
    ))
    for key, func in format_prompt.items():
        print(f"{key} format prompt:")
        print(func())
        print("\n" + "="*50 + "\n")