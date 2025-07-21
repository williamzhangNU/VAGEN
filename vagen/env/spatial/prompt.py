ACTIVE_INSTRUCTION = """\
# Spatial Exploration Task

Your goal: Learn ALL spatial relationships between EACH pair of objects in the room.

## Direction Format:
Spatial relationships are described using (<horizontal>, <vertical>) format:
- **horizontal**: left, right, same, unknown
- **vertical**: front, back, same, unknown
- "same" means objects are aligned in that dimension (e.g., same horizontal line)

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

## Tips:
- If you do not see one object in field of view, it also provides spatial information that the object is behind you (field of view is 180 degrees)


## Important Notes:
- Focus on **directional relationships** between objects, not exact distances

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
- **horizontal**: left, right, same, unknown
- **vertical**: front, back, same, unknown
- "same" means objects are aligned in that dimension (e.g., same horizontal line)

## Face direction
There are EXACTLY four facing directions: north, south, east, west. No other directions exist.
Suppose you are facing north, then:
- **forward**: north (same direction as you)
- **backward**: south (opposite direction as you)
- **right**: east (90° clockwise)
- **left**: west (90° counterclockwise)

## Note
- Track spatial relationships between EACH pair of objects during the tour
- Track precise positions where objects appear in your field of view (e.g., left-front, direct front, right-front) to build accurate spatial relationships.

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