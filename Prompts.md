Role: Spatial Reasoner in a 2D N×M grid.
Goal: Build a **COMPLETE AND ACCURATE MAP** of the environment with **MINIMAL TOTAL COST**.

## Environment Rules
- Vision: 90° FOV. Confined to current room. Doors block vision unless you are standing in the doorway (allows seeing both rooms).
- Coords: Integer (x, y). 0° = Front. + = Clockwise, - = Counterclockwise.
- Bins (Input Interpretation):
    - Ego (0°=Front): [-22.5, 22.5]: front. ±[22.5, 45]: front-slight-R/L. ±[45, 90]: front-R/L. Else: beyond-fov.
    - Cardinal (North-relative): 45° sectors (North, NE, E, SE, S, SW, W, NW).
    - Distance: 0=Same, (0,2]=Near, (2,4]=Mid, (4,8]=Slightly far, (8,16]=Far, (16,32]=Very far, >32=Extremely far.

## Actions & Grammar
- Constraint: Max 20 steps.Format: Actions: [ <Move>*, <Final> ] (List of Moves followed by exactly one Final action).
- Movement Actions (<Move>):
    - JumpTo(OBJ): Jump to the same position as the object/door OBJ. Constraint: Must be visible & previously observed. No JumpTo on step 1. Orientation unchanged. Use object/door names only.
    - Rotate(DEG): Relative rotation. Valid DEG: [-270, -180, -90, 0, 90, 180, 270].
- Final Actions (<Final> - Only one per turn):
    - Observe(): Cost: 1. Reports objects and their relationships relative to you in FOV from current pose.
    - Query(OBJ): Cost: 2. Returns exact coordinates of OBJ. Only use when necessary to eliminate ambiguities.
    - Term(): End task. Constraint: Must be the only action in the list (Actions: [Term()]).
- Examples:
    - Valid: Actions: [JumpTo(red door), Rotate(90), JumpTo(table), Observe()]
    - Valid: Actions: [Observe()] or [Query(table)]
    - Invalid (no <Final>): Actions: [JumpTo(table)]
    - Invalid (multiple <Final>): Actions: [Observe(), Rotate(90), Observe()]
    - Invalid (Term not alone): Actions: [JumpTo(table), Term()]
- Rules:
    - Observe only reports from your current pose. If you jump several times, the last Observe() shows the view from your final pose.
    - Actions execute in order.


## Current Context:
{room_info}
Unless otherwise specified, treat the starting position as origin (0, 0), facing North (+y axis).
You have a maximum of {max_steps} steps.


## Output Format
THINK:
[Reasoning for next step. Track pose/coverage.]
FINAL ANSWER:
Actions: [ ... ]