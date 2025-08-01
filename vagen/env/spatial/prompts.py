import numpy as np
from vagen.env.spatial.Base.tos_base import ActionSequence
from vagen.env.spatial.utils.generate_history import AutoExplore
from vagen.env.spatial.Base.tos_base import Room


class Prompter:
    """A class to generate prompts for the SpatialGym environment."""

    _ACTIVE_INSTRUCTION = """\
# Spatial Exploration Task

Your goal: Learn ALL spatial relationships between EACH pair of objects in the room, then STOP exploring immediately.

**STOPPING CONDITION**: You must terminate exploration the instant you can determine the relative position (left/right, front/back) of every object relative to every other object. Do not continue exploring once this condition is met.

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
3. **STOP IMMEDIATELY When Done**:
   - **TERMINATE exploration the moment you have determined all pairwise spatial relationships between objects**
   - Before each action, explicitly check: "Do I already know the relative position of every object pair?"
   - If YES, **STOP exploring immediately** - do not take unnecessary additional actions
   - **Prioritize stopping over additional exploration** once all relationships are known

## Tips:
- If you do not see one object in field of view, it also provides spatial information that the object is behind you (field of view is 180 degrees)
- **Before each action, mentally review**: Can I determine all object-to-object relationships from what I've observed? If yes, STOP immediately
- **Efficiency over completeness**: Once you can deduce all pairwise relationships (even through logical inference), terminate exploration
- **Use logical deduction**: If you know A is left of B, and B is left of C, then A is left of C - you don't need to verify this directly


## Important Notes:
- Focus on **directional relationships** between objects, not exact distances
- Pay careful attention to the precise positions of objects in your field of view (left-front, **center-front**, right-front) to accurately determine spatial relationships.
- In each image, a red dashed line indicates the agent's center front direction.

After exploration, you will answer questions.

## Room Layout
{room_info}

{exp_instructions}
"""

    _PASSIVE_INSTRUCTION = """\
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

    _COGNITION_MAP_INSTRUCTION = """\
## Cognitive Map Creation

**YOU MUST ALWAYS OUTPUT A JSON COGNITIVE MAP IN YOUR REASONING SECTION BEFORE ANSWERING ANY QUESTION.**

### Coordinate System:
- Use a 5x5 grid with YOU at the center position [0,0]
- X-axis (horizontal): -2 (far left) to +2 (far right)
- Y-axis (vertical): -2 (far back) to +2 (far front)
- Grid directions:
  * +Y = forward/north (towards you when facing north)
  * -Y = backward/south (behind you when facing north)
  * +X = right/east (to your right when facing north)
  * -X = left/west (to your left when facing north)

### Step-by-Step Process:
1. **Identify all objects** mentioned in the observations
2. **Determine each object's position** relative to your starting point [0,0]
3. **Assign coordinates** based on their spatial relationships
4. **Include object orientation** if mentioned (north/south/east/west)
5. **OUTPUT THE JSON MAP** - This step is NOT optional

### REQUIRED JSON OUTPUT FORMAT:
**You MUST include this exact JSON structure in your reasoning:**
```json
{{
  "object_name_1": {{"position": [x, y], "facing": "direction"}},
  "object_name_2": {{"position": [x, y], "facing": "direction"}}
}}
```

### Example (MUST follow this format):
If a table is front right of you and a chair is in front of you:
```json
{{
  "table": {{"position": [1, 1], "facing": "north"}},
  "chair": {{"position": [0, 1], "facing": "north"}}
}}
```

**CRITICAL**: Your response will be considered incomplete without the JSON cognitive map. Always include it in your reasoning before providing your final answer.
"""

    _EVALUATION_INSTRUCTION = """You return to your starting position and facing north. \n{eval_question}"""

    SHORT_EXPLORATION_PROMPT = "Please respond with valid actions to explore the room."
    SHORT_EVALUATION_PROMPT = "Please respond with a valid answer to the question."

    FORMAT_PROMPT = """\
Always output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text.
"""

    _TOPDOWN_PROMPT = """\
You observe the room from the topdown view: {placeholder}, \
where the blue dot indicates the agent's position and the red arrow indicates the agent's facing direction.
{object_info}
"""

    _OBLIQUE_PROMPT = """\
You observe the room from an elevated 45-degree angle view: {placeholder}, \
where the blue dot indicates the agent's position and the red arrow indicates the agent's facing direction.
"""


    def __init__(self, config, image_handler, np_random):
        self.config = config
        self.image_handler = image_handler
        self.np_random = np_random

    def _get_topdown_prompt(self, prompt_template: str, room: Room) -> str:
        obj_info = "Each object in the room is labeled with a numerical marker for easy identification."
        for idx, obj in enumerate(room.objects):
            obj_info += f"\nObject {idx + 1}: {obj.name}"
        return prompt_template.format(placeholder=self.config.image_placeholder, object_info=obj_info)

    def get_initial_observation_prompt(self, room: Room) -> dict:
        """
        Generates the initial observation prompt based on the exploration type.
        """
        room_desc = room.get_room_description()
        if self.config.prompt_with_topdown:
            room_desc += self._get_topdown_prompt(self._TOPDOWN_PROMPT, room)
        if self.config.prompt_with_oblique:
            room_desc += self._get_topdown_prompt(self._OBLIQUE_PROMPT, room)
            
        if self.config.exp_type == 'active':
            exp_instructions = f"## Action Instructions\n{ActionSequence.get_usage_instructions()}\n\nYou have a maximum of {self.config.max_exp_steps} exploration steps."
            obs_str = self._ACTIVE_INSTRUCTION.format(
                room_info=room_desc,
                exp_instructions=exp_instructions
            )
            result = {'obs_str': obs_str + "\n" + self.FORMAT_PROMPT}
            if self.config.prompt_with_topdown:
                result['multi_modal_data'] = {self.config.image_placeholder: [self.image_handler.get_image('topdown')]}
            return result
        else:
            exp_history = ""
            images = []
            if not self.config.prompt_with_topdown and not self.config.prompt_with_oblique:
                exp_history_obs = AutoExplore(room, self.np_random, self.image_handler).gen_exp_history()
                exp_history = f"## Exploration History\n{exp_history_obs['obs_str']}"
                images.extend(exp_history_obs['multi_modal_data'][self.config.image_placeholder])
            elif self.config.prompt_with_topdown:
                images.append(self.image_handler.get_image('topdown'))
            elif self.config.prompt_with_oblique:
                images.append(self.image_handler.get_image('oblique'))
                
            obs_str = self._PASSIVE_INSTRUCTION.format(
                room_info=room_desc,
                exp_history=exp_history
            )
            result = {'obs_str': obs_str + "\n" + self.FORMAT_PROMPT}
            if images:
                result['multi_modal_data'] = {self.config.image_placeholder: images}
            return result

    def get_evaluation_prompt(self, eval_question: str) -> str:
        """
        Generates the evaluation prompt, optionally including the cognitive map instructions.
        """
        if self.config.prompt_with_cogmap:
            return f"{self._COGNITION_MAP_INSTRUCTION}\n{self._EVALUATION_INSTRUCTION.format(eval_question=eval_question)}"
        return self._EVALUATION_INSTRUCTION.format(eval_question=eval_question)
