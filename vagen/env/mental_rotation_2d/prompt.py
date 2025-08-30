def system_prompt(**kwargs):
    """System prompt for mental rotation 2D task."""
    return """You are tasked with a 2D mental rotation challenge. You will be shown an image containing both the current shape and the target shape side by side. Your goal is to rotate the current shape to match the target shape's orientation.

The image layout:
- Left side: The current shape in its current orientation
- Right side: The same shape in the desired target orientation

Rules:
1. You can only rotate the shape in 2D (clockwise or counterclockwise).
2. Determine the rotation angle needed to transform the current shape (left) to match the target shape (right).
3. Provide your answer as a rotation angle in degrees.

Available actions:
- Positive angles for clockwise rotation (e.g., 15, 30, 45, 90, 180)
- Negative angles for counterclockwise rotation (e.g., -15, -30, -45, -90, -180)
- The available rotation angles depend on the task configuration

Study both shapes carefully and determine the rotation needed to match the target orientation.
"""

def init_observation_template(**kwargs):
    """Template for initial observation."""
    observation = kwargs.get("observation", "<image>")
    valid_actions = kwargs.get("valid_actions", [])
    
    actions_str = ", ".join(map(str, valid_actions)) if valid_actions else "rotation angles"
    
    return f"""[Initial State]
Current Shape: {observation}
Instruction: Your goal is to rotate the shape from its current orientation to match the target orientation shown in the target image.
Available Actions: {actions_str}
Decide your next action."""

def action_template(**kwargs):
    """Template for action feedback."""
    valid_action = kwargs.get("valid_action", "0")
    observation = kwargs.get("observation", "<image>")
    
    return f"""After your answer, the extracted valid action is {valid_action} degrees.
Your last action: {valid_action} degrees
After that, the observation is: {observation}
Instruction: Your goal is to rotate the shape from its current orientation to match the target orientation shown in the target image.
Decide your next action."""

# Format configurations for different prompt styles
FORMAT_CONFIGS = {
    "free_think": {
        "format": "<think>...</think><answer>...</answer>",
        "description": "You should first give your reasoning, and then your answer.",
        "example": "<think>Looking at the current shape, I can see it has a specific orientation. Comparing it with the target shape, I need to determine the rotation angle to match the target orientation. Based on the visual comparison, this requires a 90-degree clockwise rotation.</think><answer>90</answer>"
    },
    
    "no_think": {
        "format": "<answer>...</answer>",
        "description": "You should provide only your answer.",
        "example": "<answer>90</answer>"
    },
    
    "grounding": {
        "format": "<think><observation>...</observation><reasoning>...</reasoning></think><answer>...</answer>",
        "description": "You should first describe what you observe in both shapes, then your reasoning, and finally your answer.",
        "example": "<think><observation>In the current shape, I see a triangle with its apex pointing upward. In the target shape, I see the same triangle with its apex pointing to the right.</observation><reasoning>To rotate the triangle from its current orientation to match the target orientation, I need to rotate it 90 degrees clockwise.</reasoning></think><answer>90</answer>",
        "additional_info": "Inside the <observation> tags, describe the orientation and key features of both the current and target shapes."
    },
    
    "worldmodeling": {
        "format": "<think><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>",
        "description": "You should first give your reasoning, then predict the result after rotation, and finally your answer.",
        "example": "<think><reasoning>The current shape needs to be rotated to match the target orientation. Based on visual comparison, this requires a 90-degree clockwise rotation.</reasoning><prediction>After a 90-degree clockwise rotation, the shape will match the target orientation exactly.</prediction></think><answer>90</answer>",
        "additional_info": "Inside the <prediction> tags, describe what the shape will look like after your proposed rotation."
    },
    
    "grounding_worldmodeling": {
        "format": "<think><observation>...</observation><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>",
        "description": "You should first describe the observation, then your reasoning, then predict the result, and finally your answer.",
        "additional_info": "Describe both the current state of the shapes and predict the outcome of your rotation.",
        "example": "<think><observation>The current shape shows a triangle pointing upward, while the target shape shows the same triangle pointing to the right.</observation><reasoning>To transform the current triangle orientation to match the target orientation, I need to rotate it 90 degrees clockwise.</reasoning><prediction>After rotating 90 degrees clockwise, the triangle will point to the right, matching the target shape exactly.</prediction></think><answer>90</answer>"
    }
}

def format_prompt_generator(format_type):
    """
    Generates a prompt function for the specified format type.
    
    Args:
        format_type (str): The format type to generate a prompt function for
        
    Returns:
        function: A function that generates a prompt for the specified format
    """
    def prompt_function(**kwargs):
        """
        Generate a prompt for the specified format.
        
        Args:
            max_actions_per_step (int): Maximum number of actions allowed per step
            action_sep (str): Separator between actions (usually not needed for single rotation)
            add_example (bool): Whether to add an example
            
        Returns:
            str: The formatted prompt
        """
        max_actions_per_step = kwargs.get("max_actions_per_step", 1)
        action_sep = kwargs.get("action_sep", ",")
        add_example = kwargs.get("add_example", False)
        config = FORMAT_CONFIGS[format_type]
        
        # Build the base prompt text
        base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time.
{config["description"]}"""
        
        # Add additional information if available
        if "additional_info" in config:
            base_prompt += f"\n{config['additional_info']}"
        
        # Add response format instruction
        base_prompt += f"""
Your response should be in the format of:
{config["format"]}"""
        
        # Add example if requested
        if add_example:
            example = config["example"].format(action_sep=action_sep)
            return base_prompt + '\n' + f"e.g. {example}"
        
        return base_prompt
    
    return prompt_function

# Generate the format prompt dictionary using the generator
format_prompt = {format_type: format_prompt_generator(format_type) 
                for format_type in FORMAT_CONFIGS}

if __name__ == "__main__":
    # Example usage
    max_actions_per_step = 1
    action_sep = ","
    
    print("System Prompt:")
    print(system_prompt())
    print("\n" + "="*50 + "\n")
    
    for key, func in format_prompt.items():
        print(f"{key} format prompt:")
        print(func(max_actions_per_step=max_actions_per_step, action_sep=action_sep, add_example=True))
        print("\n" + "="*50 + "\n")
