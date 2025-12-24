#!/usr/bin/env python3
"""
Qwen VL Model Utilities

Image preprocessing and coordinate scaling for Qwen VL models.
Based on vendor OSWorld implementation.
"""

import base64
import math
import re
from io import BytesIO
from typing import Tuple, Optional

from PIL import Image


def round_by_factor(number: int, factor: int) -> int:
    """Return the nearest integer divisible by factor."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Return the smallest integer >= number that is divisible by factor."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Return the largest integer <= number that is divisible by factor."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = 32,
    min_pixels: int = 56 * 56,
    max_pixels: int = 16 * 16 * 4 * 12800,
    max_long_side: int = 8192,
) -> Tuple[int, int]:
    """
    Resize dimensions while:
    1. Making height/width divisible by factor
    2. Keeping total pixels within [min_pixels, max_pixels]
    3. Limiting longest side to max_long_side
    4. Preserving aspect ratio

    Returns:
        Tuple of (new_height, new_width)
    """
    if height < 2 or width < 2:
        raise ValueError(f"height:{height} or width:{width} must be >= 2")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(f"aspect ratio must be < 200, got {height}/{width}")

    # Limit longest side
    if max(height, width) > max_long_side:
        beta = max(height, width) / max_long_side
        height, width = int(height / beta), int(width / beta)

    # Round to factor
    h_bar = round_by_factor(height, factor)
    w_bar = round_by_factor(width, factor)

    # Adjust if outside pixel bounds
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(int(height / beta), factor)
        w_bar = floor_by_factor(int(width / beta), factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(int(height * beta), factor)
        w_bar = ceil_by_factor(int(width * beta), factor)

    return h_bar, w_bar


def preprocess_image_for_qwen(
    image_bytes: bytes,
    factor: int = 32,
    max_pixels: int = 16 * 16 * 4 * 12800,
) -> Tuple[str, int, int, int, int]:
    """
    Preprocess an image for Qwen VL models.

    Args:
        image_bytes: Raw image bytes
        factor: Resize factor (32 for Qwen3-VL)
        max_pixels: Maximum pixel count

    Returns:
        Tuple of (base64_image, original_width, original_height, processed_width, processed_height)
    """
    image = Image.open(BytesIO(image_bytes))
    original_width, original_height = image.size

    # Calculate new dimensions preserving aspect ratio
    resized_height, resized_width = smart_resize(
        height=original_height,
        width=original_width,
        factor=factor,
        max_pixels=max_pixels,
    )

    # Resize the image
    image = image.resize((resized_width, resized_height))

    # Convert to base64
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    processed_bytes = buffer.getvalue()
    base64_image = base64.b64encode(processed_bytes).decode("utf-8")

    return base64_image, original_width, original_height, resized_width, resized_height


def scale_coordinates_from_processed(
    x: float,
    y: float,
    original_width: int,
    original_height: int,
    processed_width: int,
    processed_height: int,
) -> Tuple[int, int]:
    """
    Scale coordinates from processed image dimensions back to original screen dimensions.

    Args:
        x, y: Coordinates in processed image space
        original_width, original_height: Original screen dimensions
        processed_width, processed_height: Processed image dimensions

    Returns:
        Tuple of (scaled_x, scaled_y) in original screen space
    """
    x_scale = original_width / processed_width
    y_scale = original_height / processed_height
    return int(x * x_scale), int(y * y_scale)


def scale_qwen_response_coordinates(
    response: str,
    original_width: int,
    original_height: int,
    processed_width: int,
    processed_height: int,
) -> str:
    """
    Scale coordinates in a Qwen response from processed dimensions to original screen dimensions.

    Handles pyautogui format: pyautogui.click(x, y), pyautogui.rightClick(x, y), etc.

    Args:
        response: Raw response string containing pyautogui commands
        original_width, original_height: Original screen dimensions (e.g., 1920, 1080)
        processed_width, processed_height: Processed image dimensions (e.g., 672, 384)

    Returns:
        Response with scaled coordinates
    """
    if not response:
        return response

    x_scale = original_width / processed_width
    y_scale = original_height / processed_height

    def scale_pyautogui_call(match):
        full_match = match.group(0)
        func_name = match.group(1)
        args_str = match.group(2)

        # Pattern 1: x=N, y=N (both keyword)
        both_kw = re.search(r'x\s*=\s*(\d+(?:\.\d+)?)\s*,\s*y\s*=\s*(\d+(?:\.\d+)?)', args_str)
        if both_kw:
            x_val = float(both_kw.group(1))
            y_val = float(both_kw.group(2))
            new_x = int(round(x_val * x_scale))
            new_y = int(round(y_val * y_scale))
            new_args = re.sub(
                r'x\s*=\s*\d+(?:\.\d+)?\s*,\s*y\s*=\s*\d+(?:\.\d+)?',
                f'x={new_x}, y={new_y}',
                args_str
            )
            return f'pyautogui.{func_name}({new_args})'

        # Pattern 2: x=N, N (mixed - x keyword, y positional)
        mixed_kw = re.search(r'x\s*=\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)', args_str)
        if mixed_kw:
            x_val = float(mixed_kw.group(1))
            y_val = float(mixed_kw.group(2))
            new_x = int(round(x_val * x_scale))
            new_y = int(round(y_val * y_scale))
            new_args = re.sub(
                r'x\s*=\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?',
                f'x={new_x}, y={new_y}',
                args_str
            )
            return f'pyautogui.{func_name}({new_args})'

        # Pattern 3: N, N (both positional)
        pos_match = re.match(r'^(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)(.*)', args_str)
        if pos_match:
            x_val = float(pos_match.group(1))
            y_val = float(pos_match.group(2))
            rest = pos_match.group(3)
            new_x = int(round(x_val * x_scale))
            new_y = int(round(y_val * y_scale))
            return f'pyautogui.{func_name}({new_x}, {new_y}{rest})'

        return full_match

    # Match pyautogui function calls with arguments
    pattern = r'pyautogui\.(\w+)\(([^)]+)\)'
    return re.sub(pattern, scale_pyautogui_call, response)


def parse_qwen_tool_call(
    response: str,
    original_width: int = 1920,
    original_height: int = 1080,
) -> Tuple[str, str]:
    """
    Parse Qwen3-VL tool-call response format and convert to pyautogui code.

    Qwen3-VL outputs in format:
    <tool_call>
    {"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [500, 300]}}
    </tool_call>

    This function:
    1. Extracts the JSON from <tool_call> tags
    2. Scales coordinates from 0-999 to actual screen dimensions
    3. Converts to pyautogui code

    Args:
        response: Raw response from Qwen3-VL
        original_width: Original screen width (e.g., 1920)
        original_height: Original screen height (e.g., 1080)

    Returns:
        Tuple of (pyautogui_code, action_description)
    """
    import json
    import logging

    logger = logging.getLogger("white_agent.qwen_utils")

    pyautogui_code = ""
    action_description = ""

    if not response:
        return pyautogui_code, action_description

    # Extract action description (text before <tool_call>)
    action_match = re.search(r'Action:\s*(.+?)(?=<tool_call>|$)', response, re.DOTALL)
    if action_match:
        action_description = action_match.group(1).strip()

    # Extract tool_call JSON
    tool_call_match = re.search(r'<tool_call>\s*(.*?)\s*</tool_call>', response, re.DOTALL)
    if not tool_call_match:
        # No tool_call found - check for DONE/FAIL/WAIT
        response_upper = response.upper().strip()
        if 'TERMINATE' in response_upper or 'DONE' in response_upper:
            return "DONE", action_description
        elif 'FAIL' in response_upper:
            return "FAIL", action_description
        elif 'WAIT' in response_upper:
            return "WAIT", action_description
        logger.warning(f"No <tool_call> found in response: {response[:200]}")
        return pyautogui_code, action_description

    json_str = tool_call_match.group(1).strip()

    try:
        tool_call = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse tool_call JSON: {e}, content: {json_str[:200]}")
        return pyautogui_code, action_description

    if tool_call.get("name") != "computer_use":
        logger.warning(f"Unknown tool name: {tool_call.get('name')}")
        return pyautogui_code, action_description

    args = tool_call.get("arguments", {})
    action = args.get("action", "")

    # Scale coordinates from 0-999 to actual screen dimensions
    def scale_coord(x: float, y: float) -> Tuple[int, int]:
        # Qwen outputs in 0-999 range, scale to actual screen
        scaled_x = int(round(x * original_width / 999))
        scaled_y = int(round(y * original_height / 999))
        return scaled_x, scaled_y

    if action == "left_click":
        if "coordinate" in args:
            x, y = args["coordinate"]
            sx, sy = scale_coord(x, y)
            pyautogui_code = f"pyautogui.click({sx}, {sy})"
            logger.info(f"Qwen tool_call: left_click({x}, {y}) -> click({sx}, {sy})")
        else:
            pyautogui_code = "pyautogui.click()"

    elif action == "right_click":
        if "coordinate" in args:
            x, y = args["coordinate"]
            sx, sy = scale_coord(x, y)
            pyautogui_code = f"pyautogui.rightClick({sx}, {sy})"
            logger.info(f"Qwen tool_call: right_click({x}, {y}) -> rightClick({sx}, {sy})")
        else:
            pyautogui_code = "pyautogui.rightClick()"

    elif action == "middle_click":
        if "coordinate" in args:
            x, y = args["coordinate"]
            sx, sy = scale_coord(x, y)
            pyautogui_code = f"pyautogui.middleClick({sx}, {sy})"
        else:
            pyautogui_code = "pyautogui.middleClick()"

    elif action == "double_click":
        if "coordinate" in args:
            x, y = args["coordinate"]
            sx, sy = scale_coord(x, y)
            pyautogui_code = f"pyautogui.doubleClick({sx}, {sy})"
            logger.info(f"Qwen tool_call: double_click({x}, {y}) -> doubleClick({sx}, {sy})")
        else:
            pyautogui_code = "pyautogui.doubleClick()"

    elif action == "mouse_move":
        if "coordinate" in args:
            x, y = args["coordinate"]
            sx, sy = scale_coord(x, y)
            pyautogui_code = f"pyautogui.moveTo({sx}, {sy})"
        else:
            pyautogui_code = "pyautogui.moveTo(0, 0)"

    elif action == "left_click_drag":
        if "coordinate" in args:
            x, y = args["coordinate"]
            sx, sy = scale_coord(x, y)
            # Drag to the coordinate from current position
            pyautogui_code = f"pyautogui.drag({sx}, {sy}, duration=0.5)"

    elif action == "type":
        text = args.get("text", "")
        # Escape single quotes in text
        text_escaped = text.replace("'", "\\'")
        pyautogui_code = f"pyautogui.typewrite('{text_escaped}')"

    elif action == "key":
        keys = args.get("keys", [])
        if isinstance(keys, list) and len(keys) > 0:
            keys_str = ", ".join([f"'{k}'" for k in keys])
            if len(keys) > 1:
                pyautogui_code = f"pyautogui.hotkey({keys_str})"
            else:
                pyautogui_code = f"pyautogui.press({keys_str})"

    elif action == "scroll":
        pixels = args.get("pixels", 0)
        pyautogui_code = f"pyautogui.scroll({int(pixels)})"

    elif action == "wait":
        time_secs = args.get("time", 1)
        pyautogui_code = "WAIT"

    elif action == "terminate":
        status = args.get("status", "success")
        if status == "success":
            pyautogui_code = "DONE"
        else:
            pyautogui_code = "FAIL"

    else:
        logger.warning(f"Unknown action: {action}")

    return pyautogui_code, action_description
