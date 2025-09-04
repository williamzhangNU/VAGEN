
FORMAT_PROMPT = "Always output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text."

# Initial instruction on reset: show BEFORE/AFTER images and list three action types
INIT_PROMPT = """\
One object changes its position. You should try to find which object moves and move it from the initial position to the target position. 
Initial observation: {image_placeholder}. 
Target observation: {image_placeholder}.

Available actions:
- Select(Object): Select the object that you think needs to be moved from candidates: {candidate_types}.
- Move(direction, meters): Move the selected object according to current view. Directions: ahead|back|left|right. Unit is meters.
- Term(): Terminate the episode when you think the selected object reaches the target position.

Answer format: <action_1>, <action_2>, ..., <action_n>

Rules:
- Use the fewest possible actions and avoid collisions with other objects.
- You will receive a new observation after each step. If unsure, take smaller moves and continue in the next step.
- Do not move the object out of current view.

Examples:
Step 1: Select(Chair)
Step 2: Move(ahead, 0.25), Move(left, 1.0)
Step 3: Move(ahead, 0.6), Term()

You have a maximum of {step} exploration steps.
"""

# Simple observation after actions
STEP_PROMPT = (
    "After executing these actions, you observe {image_placeholder}."
)
