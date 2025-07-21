"""
Utility functions for converting action results to text observations.
"""
from typing import List
from vagen.env.spatial.Base.tos_base import ActionResult, ObserveAction


def action_results_to_text(
    action_results: List[ActionResult],
    image_placeholder: str = "<image>",
    use_text_obs: bool = False,
) -> str:
    """Convert list of ActionResults to text observation.
    
    Args:
        action_results: List of ActionResult objects from action execution
        image_placeholder: Placeholder text for observe actions (e.g., "<image>")
        use_text_obs: Whether to use text observation for observe actions
    
    Returns:
        Text observation string
    """
    if not action_results:
        return "No actions executed"
    
    messages = []
    for result in action_results:
        if result.success:
            if result.action_type == 'observe' and not use_text_obs:
                # messages.append(result.message)
                messages.append(f"You observe: {image_placeholder}")
            else:
                messages.append(result.message)
        else:
            messages.append(result.message)
    
    return " ".join(messages) 