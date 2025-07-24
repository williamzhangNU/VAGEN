ACTIVE_INSTRUCTION = """\
# Spatial Exploration Task

Your goal: Learn ALL spatial relationships between EACH pair of objects in the room.

## Direction Format:
Spatial relationships are described using (<horizontal>, <vertical>) format:
- **horizontal**: left, right, same
- **vertical**: front, back, same
- "same" means objects are aligned in that dimension (e.g., (front, same) means center front)

## Face direction
There are EXACTLY four facing directions: north, south, east, west. No other directions exist.
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
3. **Stop When Done**: End exploration as soon as you have all spatial relationships

## Important Notes:
- Focus on **directional relationships** between objects, not exact distances
- Pay careful attention to the precise positions of objects in your field of view (left-front, **center-front**, right-front) to accurately determine spatial relationships.
- In each image, a red dashed line indicates the agent's center front direction.

After exploration, you return to starting position to answer questions.

## Room Layout
{room_info}

{exp_instructions}
"""

PASSIVE_INSTRUCTION = """\
# Spatial Understanding Task

You will be given a room layout and a tour around the room. 
NOTE: After the tour, you will return to your starting position and orientation.
Then you need to answer the question based on the tour.

## Direction Format:
Spatial relationships are described using (<horizontal>, <vertical>) format:
- **horizontal**: left, right, same
- **vertical**: front, back, same
- "same" means objects are aligned in that dimension (e.g., (front, same) means center front)

## Face direction
There are EXACTLY four facing directions: north, south, east, west. No other directions exist.
Suppose you are facing north, then:
- **forward**: north (same direction as you)
- **backward**: south (opposite direction as you)
- **right**: east (90° clockwise)
- **left**: west (90° counterclockwise)

## Note
- Track spatial relationships between EACH pair of objects during the tour
- Pay careful attention to the precise positions of objects in your field of view (left-front, **center-front**, right-front) to accurately determine spatial relationships.
- In each image, a red dashed line indicates the agent's center front direction.

## Room Layout
{room_info}

{exp_history}
"""

EVALUATION_INSTRUCTION = """\
You return to your starting position and orientation.
{eval_question}
"""

SHORT_EXPLORATION_PROMPT = """\
Please respond with valid actions to explore the room.
"""

SHORT_EVALUATION_PROMPT = """\
Please respond with a valid answer to the question.
"""

FORMAT_PROMPT = """\
Always output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text.
"""

TOPDOWN_PROMPT = """\
You observe the room from the topdown view: {placeholder}, \
where the blue dot indicates the agent's position and the red arrow indicates the agent's facing direction.
"""