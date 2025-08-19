

ACTIVE_INSTRUCTION = """\
# Spatial Exploration Task

Goal: Build a global understanding of the whole scene: resolve spatial relationships for EVERY object pair across ALL rooms. Stop immediately once complete.

Facing: forward/backward/right/left. When facing north: forward=north, back=south, right=east, left=west.

Observation: For visible objects you receive (direction, signed degree, distance).
- direction uses <vertical>-<horizontal> with front|back|same and left|right|same and
- degree is clockwise from your facing; distance is Euclidean
- You may ignore degree/distance for the stopping condition

Multi-room: The scene may have multiple rectangular rooms connected by gates/doors on vertical (N–S) or horizontal (E–W) walls. Stand at a door and use GoThroughDoor(name) to traverse.

Rules:
- Achieve complete coverage with the fewest steps; continue only while any pair is unknown
- Prefer actions that reveal many unknowns; avoid redundancy
- FOV is 90°
- Track your current and initial pose

## Room Layout
{room_info}

{cogmap_instruction}

## Action Instructions
{exp_instructions}

After exploration, you will return to your starting position facing north.
"""

PASSIVE_INSTRUCTION = """\
# Spatial Understanding Task

You will be given a multi-room layout and a tour (you return to start). Then answer the question.

## Facing
- forward, backward, right, left. When facing north: forward=north, back=south, right=east, left=west.

## Observation Format (in tour)
(direction, degree, distance); direction uses <vertical>-<horizontal>.

## Room Layout
{room_info}

{cogmap_instruction}

{exp_history}
"""

# NOTE: COGNITION_MAP_INSTRUCTION has been moved to CognitiveMap class for flexible formatting
# The dynamic instruction is now provided by CognitiveMap.get_json_format_instruction()

EVALUATION_INSTRUCTION = "NOTE: Now you return to your starting position and facing north.\n{eval_question}"
SHORT_EXPLORATION_PROMPT = "Please respond with valid actions to explore the rooms."
SHORT_EVALUATION_PROMPT = "Please respond with a valid answer to the question."