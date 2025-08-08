

ACTIVE_INSTRUCTION = """\
# Spatial Exploration Task

Your goal: Learn ALL spatial relationships between EACH pair of objects in the room, then STOP exploring immediately.

**STOPPING CONDITION**: You must terminate exploration the instant you can determine the relative position (left/right, front/back) of every object relative to every other object. Do not continue exploring once this condition is met.

## Direction Format:
Spatial relationships are described using (<horizontal>, <vertical>) format:
- **horizontal**: left, right, same
- **vertical**: front, back, same
- "same" means objects are aligned in that dimension (e.g., (right, same): aligned horizontally)

## Face direction
There are EXACTLY four facing directions: forward, backward, right, left. No other directions exist.
Suppose you are facing north, then:
- **forward**: north (same direction as you)
- **backward**: south (opposite direction as you)
- **right**: east (90° clockwise)
- **left**: west (90° counterclockwise)

## Critical Requirements:
1. **Complete Coverage**: 
   - Explore until you know where every object is relative to every other object. Only stop when you have all spatial relationships.
   - Only know relationships between object and yourself is NOT enough.
2. **Be Efficient** (Avoid redundant observations):
   - Focus on areas where you expect to eliminate some unknown relationships
   - If you know all objects are in one general direction but lack specific details, focus your exploration there 
      - (e.g., if all objects are to your left but you don't know which are in front vs. back, explore the left side systematically)
3. **STOP IMMEDIATELY When Done**:
   - **TERMINATE exploration the moment you have determined all pairwise spatial relationships between objects**
   - Before each action, explicitly check: "Do I already know the relative position of every object pair?"
   - If YES, **STOP exploring immediately** - do not take unnecessary additional actions
   - **Prioritize stopping over additional exploration** once all relationships are known
4. **Self-awareness**:
   - You MUST always remember your current position and orientation.
   - You MUST always remember your initial position and orientation and its relationship with other objects.

## Tips:
- If you do not see one object in field of view, it also provides spatial information that the object is behind you (field of view is 180 degrees)
- **Before each action, mentally review**: Can I determine all object-to-object relationships from what I've observed? If yes, STOP immediately
- **Efficiency over completeness**: Once you can deduce all pairwise relationships (even through logical inference), terminate exploration
- **Use logical deduction**: If you know A is left of B, and B is left of C, then A is left of C - you don't need to verify this directly


## Important Notes:
- Transitivity (e.g., A left of B and B left of C ⇒ A left of C) to deduce relations without further movement.
- Focus solely on directional relationships, they are all you need to determine. Ignore distances.
- Pay careful attention to the precise positions of objects in your field of view (left-front, **center-front**, right-front) to accurately determine spatial relationships.
- In each image, a red dashed line indicates the agent's center front direction.

After exploration, you will answer questions.

## Room Layout
{room_info}

{cogmap_instruction}

## Action Instructions
{exp_instructions}
"""

ACTIVE_INSTRUCTION_SHORTER = """\
# Spatial Exploration Task

## Goal
Determine the left/right & front/back relationship for **EACH** pair of objects in the room. Stop the moment all pairwise relations are known.

## Facing & Directions
- Allowed facings: forward, backward, right, left.
- When facing north: forward = north, back = south, right = east, left = west.

## Relation Format
`(<horizontal>, <vertical>)`
- horizontal: left | right | same
- vertical: front | back | same
- "same" means objects are aligned in that dimension (e.g., (right, same): aligned horizontally)

## Exploration Rules
- **Complete Coverage**: Continue only while any object-to-object relation is unknown.
- **Efficiency**: Pick actions that eliminate the most unknowns; skip views that add no new pairwise info.
    - If you know all objects are in one general direction but lack specific details, focus your exploration there
    - E.g., if all objects are at your left but don't know front or back, you should observe left
- **Field of View**: An object outside your 180° FOV is behind you. Use this fact to deduce relations.
- **Immediate Termination**: Before each move ask: "Do I now know every pairwise relation?" 
    - If yes, output results and end—never take extra steps.
- **Self-Tracking**: Always track your current & initial pose.

## Rules
- Transitivity (e.g., A left of B and B left of C ⇒ A left of C) to deduce relations without further movement.
- Focus solely on directional relationships, they are all you need to determine. Ignore distances.
- Pay careful attention to the precise positions of objects in your field of view (left-front, **center-front**, right-front) to accurately determine spatial relationships.
- In each image, a red dashed line indicates the agent's center front direction.

## Room Layout
{room_info}

{cogmap_instruction}

## Action Instructions
{exp_instructions}

After exploration, NOTE you will return to your starting position and facing north.
"""


PASSIVE_INSTRUCTION = """\
# Spatial Understanding Task

You will be given a room layout and a tour around the room. 
NOTE: After the tour, you will return to your starting position and orientation.
Then you need to answer the question based on the tour.

## Facing & Directions
- Allowed facings: forward, backward, right, left.
- When facing north: forward = north, back = south, right = east, left = west.

## Relation Format
`(<horizontal>, <vertical>)`
- horizontal: left | right | same
- vertical: front | back | same
- "same" means objects are aligned in that dimension (e.g., (right, same): aligned horizontally)

## Note
- Track spatial relationships between EACH pair of objects during the tour
- Pay careful attention to the precise positions of objects in your field of view (left-front, **center-front**, right-front) to accurately determine spatial relationships.
- In each image, a red dashed line indicates the agent's center front direction.

## Room Layout
{room_info}

{cogmap_instruction}

{exp_history}
"""


EVALUATION_INSTRUCTION = """You return to your starting position and facing north. \n{eval_question}"""

SHORT_EXPLORATION_PROMPT = "Please respond with valid actions to explore the room."
SHORT_EVALUATION_PROMPT = "Please respond with a valid answer to the question."

FORMAT_PROMPT = """\
Always output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text.
"""

TOPDOWN_PROMPT = """\
You observe the room from the topdown view: {placeholder}, \
where the blue dot indicates the agent's position and the red arrow indicates the agent's facing direction.
{object_info}
"""

OBLIQUE_PROMPT = """\
You observe the room from an elevated 45-degree angle view: {placeholder}, \
where the blue dot indicates the agent's position and the red arrow indicates the agent's facing direction.
"""