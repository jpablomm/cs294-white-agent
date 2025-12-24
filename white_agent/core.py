#!/usr/bin/env python3
"""
White Agent Core - Shared utilities for A2A and REST implementations.

This module contains:
- Action parsing (pyautogui -> OSWorld action format)
- Observation parsing (base64 screenshots, accessibility trees)
- URL building utilities
"""

import base64
import json
import logging
import re
from typing import Dict, Any

from white_agent.config import get_agent_url

logger = logging.getLogger(__name__)


def parse_observation(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse observation from request data.

    Supports multiple formats:
    - Nested: {"observation": {"image_png_b64": ..., "instruction": ...}}
    - Flat: {"image_png_b64": ..., "instruction": ...}

    Returns:
        Dict with 'screenshot' (bytes), 'instruction' (str), 'frame_id' (int),
        'done' (bool), and optionally 'accessibility_tree' (str)
    """
    # Support both nested and flat formats
    if "observation" in data:
        obs_data = data["observation"]
    else:
        obs_data = data

    # Decode base64 screenshot
    image_b64 = obs_data.get("image_png_b64", "")
    if not image_b64:
        raise ValueError("Observation must have image_png_b64")

    screenshot_bytes = base64.b64decode(image_b64)

    result = {
        "frame_id": obs_data.get("frame_id", 0),
        "screenshot": screenshot_bytes,
        "instruction": obs_data.get("instruction", obs_data.get("message", "")),
        "done": obs_data.get("done", False)
    }

    # Include accessibility tree if provided (XML string from OSWorld VM)
    if "accessibility_tree" in obs_data and obs_data["accessibility_tree"]:
        result["accessibility_tree"] = obs_data["accessibility_tree"]

    return result


def parse_actions(actions_str: str) -> Dict[str, Any]:
    """
    Parse pyautogui action string into OSWorld action format.

    Converts from:
        "pyautogui.click(100, 200)"
    To:
        {"op": "click", "args": {"x": 100, "y": 200}}

    Also handles:
    - JSON format: '{"op": "click", "args": {"x": 100, "y": 200}}'
    - Multi-line code blocks
    - DONE/FAIL markers
    """
    actions_str = actions_str.strip()

    # Strip markdown code blocks (```json ... ``` or ```python ... ```)
    # First, extract only the FIRST code block if multiple exist
    if '```' in actions_str:
        # Find the first code block
        code_block_match = re.search(r'```(?:python|json)?\s*\n?(.*?)```', actions_str, re.DOTALL)
        if code_block_match:
            actions_str = code_block_match.group(1).strip()
        elif actions_str.startswith('```'):
            # Fallback: Remove opening fence (```json, ```python, or just ```)
            lines = actions_str.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]  # Remove first line with ```
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]  # Remove last line with ```
            actions_str = '\n'.join(lines).strip()

    # Check for JSON format (GPT-4o and Qwen sometimes return this)
    if actions_str.startswith('{') and actions_str.endswith('}'):
        try:
            action_dict = json.loads(actions_str)
            if "op" in action_dict:
                # Handle screenshot action (convert to wait since screenshots are automatic)
                if action_dict["op"] == "screenshot":
                    logger.info("Model requested screenshot - converting to wait")
                    return {"op": "wait", "args": {"duration": 0.5}}

                # Handle Qwen format where coordinates are packed as array in "x" field
                # e.g., {"op": "right_click", "args": {"x": [14, 124]}}
                args = action_dict.get("args", {})
                if "x" in args and isinstance(args["x"], list) and len(args["x"]) >= 2:
                    # Extract x, y from the array
                    coords = args["x"]
                    action_dict["args"] = {"x": coords[0], "y": coords[1]}
                    logger.info(f"Extracted coordinates from array: x={coords[0]}, y={coords[1]}")

                return action_dict
        except json.JSONDecodeError:
            pass

    # Extract first command from multi-line code blocks BEFORE checking for DONE/FAIL
    # This prevents false positives from verbose model responses
    if '\n' in actions_str or 'import' in actions_str:
        lines = actions_str.split('\n')
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('import') or line.startswith('time.'):
                continue
            if line.startswith('pyautogui.') or any(line.startswith(f'{cmd}(') for cmd in
                ['click', 'type_text', 'hotkey', 'scroll', 'doubleClick', 'rightClick',
                 'moveTo', 'moveRel', 'drag', 'dragTo', 'mouseDown', 'mouseUp', 'press']):
                actions_str = line
                break
        else:
            # No pyautogui command found - NOW check for standalone DONE/FAIL
            actions_str_upper = actions_str.upper()
            if actions_str_upper.strip() in ['DONE', 'FAIL'] or \
               actions_str_upper.strip().startswith('DONE') or \
               actions_str_upper.strip().startswith('FAIL'):
                return {"op": "done", "args": {}}
            logger.warning("No command found in code block, defaulting to wait")
            return {"op": "wait", "args": {"duration": 1.0}}

    # Check for standalone DONE/FAIL (only for simple single-line responses)
    actions_str_upper = actions_str.upper().strip()
    if actions_str_upper in ['DONE', 'FAIL', 'WAIT']:
        if actions_str_upper == 'WAIT':
            return {"op": "wait", "args": {"duration": 1.0}}
        return {"op": "done", "args": {}}

    # Strip comments
    if '#' in actions_str:
        actions_str = actions_str.split('#')[0].strip()

    # Parse move actions
    if match := re.match(r'pyautogui\.moveRel\((-?\d+),\s*(-?\d+)\)', actions_str):
        logger.info("moveRel detected - treating as wait")
        return {"op": "wait", "args": {"duration": 0.5}}

    # Parse moveTo action
    if match := re.match(r'(?:pyautogui\.)?moveTo\((?:x=)?(\d+),\s*(?:y=)?(\d+)(?:,\s*duration=[\d.]+)?\)', actions_str):
        return {"op": "move", "args": {"x": int(match.group(1)), "y": int(match.group(2))}}

    # Parse click with button parameter (must be before simple click pattern)
    # Handles: pyautogui.click(x, y, button='right') or click(x, y, button='left')
    if match := re.match(r'(?:pyautogui\.)?click\((?:x=)?(\d+),\s*(?:y=)?(\d+),\s*button=["\'](\w+)["\']', actions_str):
        x, y, button = int(match.group(1)), int(match.group(2)), match.group(3)
        if button == 'right':
            return {"op": "right_click", "args": {"x": x, "y": y}}
        elif button == 'middle':
            return {"op": "click", "args": {"x": x, "y": y, "button": "middle"}}
        else:  # left or default
            return {"op": "click", "args": {"x": x, "y": y}}

    # Parse simple click actions
    if match := re.match(r'(?:pyautogui\.)?click\((?:x=)?(\d+),\s*(?:y=)?(\d+)\)', actions_str):
        return {"op": "click", "args": {"x": int(match.group(1)), "y": int(match.group(2))}}

    if match := re.match(r'(?:pyautogui\.)?click\(\)', actions_str):
        return {"op": "click", "args": {}}

    # Parse double click
    if match := re.match(r'(?:pyautogui\.)?doubleClick\((?:x=)?(\d+),\s*(?:y=)?(\d+)\)', actions_str):
        return {"op": "double_click", "args": {"x": int(match.group(1)), "y": int(match.group(2))}}

    # Parse right click (explicit rightClick function)
    if match := re.match(r'(?:pyautogui\.)?rightClick\((?:x=)?(\d+),\s*(?:y=)?(\d+)\)', actions_str):
        return {"op": "right_click", "args": {"x": int(match.group(1)), "y": int(match.group(2))}}

    # Parse type/write actions (output type_text to match green agent tool name)
    if match := re.match(r'(?:pyautogui\.)?(?:typewrite|write|type_text)\(["\'](.+?)["\']\)', actions_str):
        return {"op": "type_text", "args": {"text": match.group(1)}}

    # Parse hotkey actions
    if match := re.match(r'(?:pyautogui\.)?hotkey\(["\'](.+?)["\'],\s*["\'](.+?)["\']\)', actions_str):
        return {"op": "hotkey", "args": {"keys": [match.group(1), match.group(2)]}}

    # Parse hotkey with array syntax
    if match := re.match(r'(?:pyautogui\.)?hotkey\(\[([^\]]+)\]\)', actions_str):
        keys = [k.strip().strip("'\"") for k in match.group(1).split(',')]
        return {"op": "hotkey", "args": {"keys": keys}}

    # Parse press actions
    if match := re.match(r'(?:pyautogui\.)?press\(["\'](.+?)["\']', actions_str):
        return {"op": "hotkey", "args": {"keys": [match.group(1)]}}

    # Parse scroll
    if match := re.match(r'(?:pyautogui\.)?scroll\((-?\d+)\)', actions_str):
        return {"op": "scroll", "args": {"amount": int(match.group(1))}}

    # Parse drag operations (convert to execute_python since VM doesn't have native drag)
    if match := re.match(r'(?:pyautogui\.)?drag\((-?\d+),\s*(-?\d+)', actions_str):
        dx, dy = int(match.group(1)), int(match.group(2))
        logger.info(f"drag detected - converting to execute_python")
        return {"op": "execute_python", "args": {"code": f"import pyautogui; pyautogui.drag({dx}, {dy}, duration=0.5)"}}

    if match := re.match(r'(?:pyautogui\.)?dragTo\((\d+),\s*(\d+)', actions_str):
        x, y = int(match.group(1)), int(match.group(2))
        logger.info(f"dragTo detected - converting to execute_python")
        return {"op": "execute_python", "args": {"code": f"import pyautogui; pyautogui.dragTo({x}, {y}, duration=0.5)"}}

    # Parse mouseDown/mouseUp (part of drag sequences - convert to wait with warning)
    if re.match(r'(?:pyautogui\.)?mouseDown\(', actions_str):
        logger.warning("mouseDown detected - VM doesn't support partial drag, use execute_python for drag sequences")
        return {"op": "wait", "args": {"duration": 0.5}}

    if re.match(r'(?:pyautogui\.)?mouseUp\(', actions_str):
        logger.warning("mouseUp detected - VM doesn't support partial drag, use execute_python for drag sequences")
        return {"op": "wait", "args": {"duration": 0.5}}

    # Default: wait
    logger.warning(f"Unknown action format: {actions_str}, defaulting to wait")
    return {"op": "wait", "args": {"duration": 1.0}}


def build_agent_url() -> str:
    """Build agent URL from environment variables."""
    return get_agent_url()
