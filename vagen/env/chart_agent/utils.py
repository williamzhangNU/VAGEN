import re
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image, ImageDraw


def apply_bounding_boxes(
    image: Image.Image,
    bounding_boxes: List[Tuple[int, int, int, int]],
    return_cropped: bool = False,
    highlight_boxes: bool = False,
) -> Tuple[Image.Image, List[Image.Image]]:
    """
    Apply bounding boxes to a PIL Image, with optional highlighting and cropping.
    
    Args:
        image: Input PIL Image.
        bounding_boxes: List of bounding boxes in (x1, y1, x2, y2) format.
        return_cropped: If True, returns cropped images from bounding boxes.
        highlight_boxes: If True, highlights bounding box areas and darkens the rest.
    
    Returns:
        A tuple containing:
        - The image with bounding boxes drawn.
        - A list of cropped images (empty if return_cropped is False).

    TODO:
    - Change font size and bounding box width according to image size
    - Change way to highlight bounding box area
    """
    if not bounding_boxes:
        return image, []
        
    output_image = image.copy()
    width, height = image.size

    if highlight_boxes:
        output_image = output_image.point(lambda p: int(p * 0.5))
        # Sort boxes by area (descending) so smaller boxes are processed last and drawn on top.
        indexed_boxes = sorted(
            enumerate(bounding_boxes),
            key=lambda item: (item[1][2] - item[1][0]) * (item[1][3] - item[1][1]),
            reverse=True,
        )

        for _, (x1, y1, x2, y2) in indexed_boxes:
            box = (max(0, x1), max(0, y1), min(width, x2), min(height, y2))
            
            region = image.crop(box)
            if region.mode != "RGBA":
                region = region.convert("RGBA")

            yellow_overlay = Image.new("RGBA", region.size, (255, 255, 0, 80))
            highlighted_region = Image.alpha_composite(region, yellow_overlay)

            if image.mode == "RGB":
                highlighted_region = highlighted_region.convert("RGB")

            output_image.paste(highlighted_region, (box[0], box[1]))

    draw = ImageDraw.Draw(output_image)
    cropped_regions = []

    for i, (x1, y1, x2, y2) in enumerate(bounding_boxes):
        box = (max(0, x1), max(0, y1), min(width, x2), min(height, y2))
        
        draw.rectangle(box, outline="red", width=5)
        draw.text((box[0], box[1] - 15), f"Box {i+1}", fill="red")
        
        if return_cropped:
            cropped_regions.append(image.crop(box))
            
    return output_image, cropped_regions


def response_extraction(llm_raw_response: str) -> Optional[Dict[str, Any]]:
    """
    Extract answer and bounding boxes from the LLM's response.
    
    Args:
        llm_raw_response: The raw response from the LLM.
        
    Returns:
        Dictionary containing:
        - answer (str): The extracted final answer, empty if not found
        - reasoning (str): The reasoning/thinking process
        - bounding_boxes (List[Tuple[int, int, int, int]]): Extracted bounding boxes
        
    Returns None if parsing fails completely.

    TODO modify here
    """
    if not llm_raw_response or not isinstance(llm_raw_response, str):
        return None
    
    response = llm_raw_response.strip()
    
    # Initialize result
    result = {
        "answer": "",
        "reasoning": "",
        "bounding_boxes": []
    }
    
    # Extract reasoning/thinking content
    think_pattern = r'<think>(.*?)</think>'
    think_match = re.search(think_pattern, response, re.DOTALL)
    if think_match:
        result["reasoning"] = think_match.group(1).strip()
    
    # Extract final answer
    answer_pattern = r'<answer>(.*?)</answer>'
    answer_match = re.search(answer_pattern, response, re.DOTALL)
    if answer_match:
        result["answer"] = answer_match.group(1).strip()
    
    # Extract bounding boxes - look for coordinate patterns
    # Pattern for coordinates like (x1, y1, x2, y2) or [x1, y1, x2, y2]
    bbox_pattern = r'[\(\[]\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*[\)\]]'
    bbox_matches = re.findall(bbox_pattern, response)
    
    if bbox_matches:
        result["bounding_boxes"] = [(int(x1), int(y1), int(x2), int(y2)) 
                                   for x1, y1, x2, y2 in bbox_matches]
    
    # If no structured format found, treat entire response as reasoning
    if not result["answer"] and not result["reasoning"] and not result["bounding_boxes"]:
        result["reasoning"] = response
        # For simple text responses, assume it might be an answer
        if len(response.split()) <= 10:  # Short responses likely to be answers
            result["answer"] = response
    
    return result


if __name__ == "__main__":
    # Test the response extraction
    test_responses = [
        "<think>I need to analyze the chart</think><answer>42</answer>",
        "The answer is 100 based on the chart data at coordinates [50, 50, 100, 100]",
        "<answer>The value is 75</answer>",
        "Simple answer without formatting"
    ]
    
    for i, response in enumerate(test_responses):
        print(f"Test {i+1}: {response}")
        result = response_extraction(response)
        print(f"Result: {result}")
        print("-" * 50)

    # # test cases for bouding box 
    # image = Image.open("vagen/env/chart_agent/ChartAgent/images/test.png")
    # bounding_boxes = [
    #     (360, 180, 440, 260),
    #     (500, 200, 650, 400),
    #     (370, 400, 430, 500),
    #     # (0, 0, 1000, 1000),
    # ]

    # img_with_boxes, cropped_regions = apply_bounding_boxes(
    #     image, bounding_boxes, return_cropped=True, highlight_boxes=True
    # )
    # img_with_boxes.save("vagen/env/chart_agent/ChartAgent/images/test_with_boxes.png")

    # for i, cropped in enumerate(cropped_regions):
    #     cropped.save(f"vagen/env/chart_agent/ChartAgent/images/test_cropped_{i}.png")

    # 