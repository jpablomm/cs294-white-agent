"""
PromptAgent - Multi-model vision-language agent for desktop automation.

Forked from OSWorld's mm_agents/agent.py and adapted for the white_agent package.
Supports GPT-4V, Claude, Gemini, Qwen, and other vision-language models.
"""

import base64
import json
import logging
import os
import re
import tempfile
import time
import xml.etree.ElementTree as ET
from http import HTTPStatus
from io import BytesIO
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

from white_agent.config import (
    get_azure_openai_api_key,
    get_azure_openai_endpoint,
    get_openai_base_url,
    get_reasoning_effort,
    get_genai_api_key,
    get_groq_api_key,
    get_qwen3_vl_endpoint_url,
)

import backoff
import requests
import tiktoken
from PIL import Image
from requests.exceptions import SSLError

# Optional imports - these are only needed for specific providers
try:
    import dashscope
except ImportError:
    dashscope = None

try:
    import google.generativeai as genai
    from google.api_core.exceptions import InvalidArgument, ResourceExhausted, InternalServerError, BadRequest
except ImportError:
    genai = None
    InvalidArgument = ResourceExhausted = InternalServerError = BadRequest = Exception

try:
    import openai
except ImportError:
    openai = None

try:
    from groq import Groq
except ImportError:
    Groq = None

# LangChain with DeepAgents support
try:
    from langchain.chat_models import init_chat_model
    from langchain_core.messages import HumanMessage, SystemMessage
    langchain_available = True
except ImportError:
    langchain_available = False

try:
    from deepagents import create_deep_agent
    deepagents_available = True
except ImportError:
    deepagents_available = False

try:
    from tavily import TavilyClient
    tavily_available = True
except ImportError:
    tavily_available = False

# Local imports - now using white_agent package paths
from white_agent.accessibility_tree_wrap.heuristic_retrieve import filter_nodes, draw_bounding_boxes
from white_agent.prompts import (
    SYS_PROMPT_IN_SCREENSHOT_OUT_CODE,
    SYS_PROMPT_IN_SCREENSHOT_OUT_ACTION,
    SYS_PROMPT_IN_A11Y_OUT_CODE,
    SYS_PROMPT_IN_A11Y_OUT_ACTION,
    SYS_PROMPT_IN_BOTH_OUT_CODE,
    SYS_PROMPT_IN_BOTH_OUT_ACTION,
    SYS_PROMPT_IN_SOM_OUT_TAG,
    QWEN3_VL_SYSTEM_PROMPT,
)
from white_agent.qwen_utils import (
    preprocess_image_for_qwen,
    parse_qwen_tool_call,
)

logger = logging.getLogger("white_agent.prompt_agent")

pure_text_settings = ['a11y_tree']

# Groq model configurations (updated December 2025)
# See https://console.groq.com/docs/models for latest models
#
# Only including VISION-CAPABLE models for screenshot-based tasks
GROQ_MODELS = {
    # Llama 4 multimodal models (support images)
    "meta-llama/llama-4-maverick-17b-128e-instruct",  # Llama 4 Maverick - 600 t/s, vision, $0.20/$0.60
    "meta-llama/llama-4-scout-17b-16e-instruct",      # Llama 4 Scout - 750 t/s, vision, $0.11/$0.34
}

# User-friendly name mappings (for groq-* prefix usage)
GROQ_MODEL_MAPPING = {
    "llama4-maverick": "meta-llama/llama-4-maverick-17b-128e-instruct",
    "llama4-scout": "meta-llama/llama-4-scout-17b-16e-instruct",
    # Short aliases
    "llama4": "meta-llama/llama-4-scout-17b-16e-instruct",  # Default to Scout (faster)
}

# Qwen3-VL models on Vertex AI (vision + GUI automation)
# Deploy with: python scripts/vertex-ai/deploy_qwen3_vl.py --project YOUR_PROJECT
QWEN3_VL_MODELS = {
    # 30B models (faster, cheaper)
    "qwen3-vl-30b-instruct",
    "qwen3-vl-30b-thinking",
    # 235B models (more capable)
    "qwen3-vl-235b-instruct",
    "qwen3-vl-235b-thinking",
}

# Map user-friendly names to Vertex AI model names
QWEN3_VL_MODEL_MAPPING = {
    "qwen3-vl-30b-instruct": "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "qwen3-vl-30b-thinking": "Qwen/Qwen3-VL-30B-A3B-Thinking",
    "qwen3-vl-235b-instruct": "Qwen/Qwen3-VL-235B-A22B-Instruct",
    "qwen3-vl-235b-thinking": "Qwen/Qwen3-VL-235B-A22B-Thinking",
    # Short aliases
    "qwen3-vl": "Qwen/Qwen3-VL-235B-A22B-Instruct",  # Default to 235B instruct
}

attributes_ns_ubuntu = "https://accessibility.windows.example.org/ns/attributes"
attributes_ns_windows = "https://accessibility.windows.example.org/ns/attributes"
state_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/state"
state_ns_windows = "https://accessibility.windows.example.org/ns/state"
component_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/component"
component_ns_windows = "https://accessibility.windows.example.org/ns/component"
value_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/value"
value_ns_windows = "https://accessibility.windows.example.org/ns/value"
class_ns_windows = "https://accessibility.windows.example.org/ns/class"
# More namespaces defined in OSWorld, please check desktop_env/server/main.py


# Function to encode the image
def encode_image(image_content):
    return base64.b64encode(image_content).decode('utf-8')


def encoded_img_to_pil_img(data_str):
    base64_str = data_str.replace("data:image/png;base64,", "")
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))

    return image


def save_to_tmp_img_file(data_str):
    base64_str = data_str.replace("data:image/png;base64,", "")
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))

    tmp_img_path = os.path.join(tempfile.mkdtemp(), "tmp_img.png")
    image.save(tmp_img_path)

    return tmp_img_path


def scale_qwen_coordinates(response: str, screen_width: int = 1920, screen_height: int = 1080) -> str:
    """
    Scale Qwen VL model coordinates from 0-999 normalized space to actual screen pixels.

    Qwen VL models are trained to output coordinates in a 1000x1000 normalized grid.
    This function parses pyautogui code and scales x,y values to actual screen resolution.

    Handles multiple coordinate formats:
    - pyautogui.click(x=17, y=134)    # Both keyword args
    - pyautogui.click(17, 134)         # Both positional args
    - pyautogui.click(x=17, 134)       # Mixed: x keyword, y positional

    Args:
        response: The raw response from Qwen VL containing pyautogui code
        screen_width: Actual screen width in pixels (default 1920)
        screen_height: Actual screen height in pixels (default 1080)

    Returns:
        Response with coordinates scaled to actual screen resolution
    """
    if not response:
        return response

    # Scale factor from 0-999 grid to actual screen
    x_scale = screen_width / 999.0
    y_scale = screen_height / 999.0

    def scale_pyautogui_call(match):
        """Scale coordinates in a pyautogui function call."""
        full_match = match.group(0)
        func_name = match.group(1)
        args_str = match.group(2)

        # Try different coordinate patterns:

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

        # Pattern 2: x=N, N (x keyword, y positional) - Qwen's mixed format
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

        # Pattern 3: N, N (both positional) - check it's at start of args
        pos_match = re.match(r'^(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)(.*)', args_str)
        if pos_match:
            x_val = float(pos_match.group(1))
            y_val = float(pos_match.group(2))
            rest = pos_match.group(3)
            new_x = int(round(x_val * x_scale))
            new_y = int(round(y_val * y_scale))
            return f'pyautogui.{func_name}({new_x}, {new_y}{rest})'

        # No coordinate pattern matched, return unchanged
        return full_match

    # Match any pyautogui function call with arguments
    pattern = r'pyautogui\.(\w+)\(([^)]+)\)'
    scaled_response = re.sub(pattern, scale_pyautogui_call, response)

    return scaled_response


def linearize_accessibility_tree(accessibility_tree, platform="ubuntu"):

    if platform == "ubuntu":
        _attributes_ns = attributes_ns_ubuntu
        _state_ns = state_ns_ubuntu
        _component_ns = component_ns_ubuntu
        _value_ns = value_ns_ubuntu
    elif platform == "windows":
        _attributes_ns = attributes_ns_windows
        _state_ns = state_ns_windows
        _component_ns = component_ns_windows
        _value_ns = value_ns_windows
    else:
        raise ValueError("Invalid platform, must be 'ubuntu' or 'windows'")

    filtered_nodes = filter_nodes(ET.fromstring(accessibility_tree), platform)
    linearized_accessibility_tree = ["tag\tname\ttext\tclass\tdescription\tposition (top-left x&y)\tsize (w&h)"]

    # Linearize the accessibility tree nodes into a table format
    for node in filtered_nodes:
        if node.text:
            text = (
                node.text if '"' not in node.text \
                    else '"{:}"'.format(node.text.replace('"', '""'))
            )

        elif node.get("{{{:}}}class".format(class_ns_windows), "").endswith("EditWrapper") \
                and node.get("{{{:}}}value".format(_value_ns)):
            node_text = node.get("{{{:}}}value".format(_value_ns), "")
            text = (node_text if '"' not in node_text \
                        else '"{:}"'.format(node_text.replace('"', '""'))
                    )
        else:
            text = '""'

        linearized_accessibility_tree.append(
            "{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}".format(
                node.tag, node.get("name", ""),
                text,
                node.get("{{{:}}}class".format(_attributes_ns), "") if platform == "ubuntu" else node.get("{{{:}}}class".format(class_ns_windows), ""),
                node.get("{{{:}}}description".format(_attributes_ns), ""),
                node.get('{{{:}}}screencoord'.format(_component_ns), ""),
                node.get('{{{:}}}size'.format(_component_ns), "")
            )
        )

    return "\n".join(linearized_accessibility_tree)


def tag_screenshot(screenshot, accessibility_tree, platform="ubuntu"):
    nodes = filter_nodes(ET.fromstring(accessibility_tree), platform=platform, check_image=True)
    # Make tag screenshot
    marks, drew_nodes, element_list, tagged_screenshot = draw_bounding_boxes(nodes, screenshot)

    return marks, drew_nodes, tagged_screenshot, element_list


def parse_actions_from_string(input_string):
    if input_string.strip() in ['WAIT', 'DONE', 'FAIL']:
        return [input_string.strip()]
    # Search for a JSON string within the input string
    actions = []
    matches = re.findall(r'```json\s+(.*?)\s+```', input_string, re.DOTALL)
    if matches:
        # Assuming there's only one match, parse the JSON string into a dictionary
        try:
            for match in matches:
                action_dict = json.loads(match)
                actions.append(action_dict)
            return actions
        except json.JSONDecodeError as e:
            return f"Failed to parse JSON: {e}"
    else:
        matches = re.findall(r'```\s+(.*?)\s+```', input_string, re.DOTALL)
        if matches:
            # Assuming there's only one match, parse the JSON string into a dictionary
            try:
                for match in matches:
                    action_dict = json.loads(match)
                    actions.append(action_dict)
                return actions
            except json.JSONDecodeError as e:
                return f"Failed to parse JSON: {e}"
        else:
            try:
                action_dict = json.loads(input_string)
                return [action_dict]
            except json.JSONDecodeError:
                raise ValueError("Invalid response format: " + input_string)


def parse_code_from_string(input_string):
    input_string = "\n".join([line.strip() for line in input_string.split(';') if line.strip()])
    if input_string.strip() in ['WAIT', 'DONE', 'FAIL']:
        return [input_string.strip()]

    # This regular expression will match both ```code``` and ```python code```
    # and capture the `code` part. It uses a non-greedy match for the content inside.
    pattern = r"```(?:\w+\s+)?(.*?)```"
    # Find all non-overlapping matches in the string
    matches = re.findall(pattern, input_string, re.DOTALL)

    # The regex above captures the content inside the triple backticks.
    # The `re.DOTALL` flag allows the dot `.` to match newline characters as well,
    # so the code inside backticks can span multiple lines.

    # matches now contains all the captured code snippets

    codes = []

    for match in matches:
        match = match.strip()
        commands = ['WAIT', 'DONE', 'FAIL']  # fixme: updates this part when we have more commands

        if match in commands:
            codes.append(match.strip())
        elif match.split('\n')[-1] in commands:
            if len(match.split('\n')) > 1:
                codes.append("\n".join(match.split('\n')[:-1]))
            codes.append(match.split('\n')[-1])
        else:
            codes.append(match)

    return codes


def parse_code_from_som_string(input_string, masks):
    # parse the output string by masks
    tag_vars = ""
    for i, mask in enumerate(masks):
        x, y, w, h = mask
        tag_vars += "tag_" + str(i + 1) + "=" + "({}, {})".format(int(x + w // 2), int(y + h // 2))
        tag_vars += "\n"

    actions = parse_code_from_string(input_string)

    for i, action in enumerate(actions):
        if action.strip() in ['WAIT', 'DONE', 'FAIL']:
            pass
        else:
            action = tag_vars + action
            actions[i] = action

    return actions


def trim_accessibility_tree(linearized_accessibility_tree, max_tokens):
    enc = tiktoken.encoding_for_model("gpt-4")
    tokens = enc.encode(linearized_accessibility_tree)
    if len(tokens) > max_tokens:
        linearized_accessibility_tree = enc.decode(tokens[:max_tokens])
        linearized_accessibility_tree += "[...]\n"
    return linearized_accessibility_tree


class PromptAgent:
    def __init__(
            self,
            platform="ubuntu",
            model="gpt-4-vision-preview",
            max_tokens=1500,
            top_p=0.9,
            temperature=0.5,
            action_space="computer_13",
            observation_type="screenshot_a11y_tree",
            # observation_type can be in ["screenshot", "a11y_tree", "screenshot_a11y_tree", "som"]
            max_trajectory_length=3,
            a11y_tree_max_tokens=10000,
            client_password="password"
    ):
        self.platform = platform
        self.model = model
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.action_space = action_space
        self.observation_type = observation_type
        self.max_trajectory_length = max_trajectory_length
        self.a11y_tree_max_tokens = a11y_tree_max_tokens
        self.client_password = client_password

        self.thoughts = []
        self.actions = []
        self.observations = []

        # Track if this is a Qwen3-VL model (needs image preprocessing)
        self.is_qwen3_vl = model.lower().startswith("qwen3-vl") or "qwen3-vl" in model.lower()
        # Dimension tracking for Qwen3-VL coordinate scaling
        self._qwen_original_width = None
        self._qwen_original_height = None
        self._qwen_processed_width = None
        self._qwen_processed_height = None

        if observation_type == "screenshot":
            if action_space == "computer_13":
                self.system_message = SYS_PROMPT_IN_SCREENSHOT_OUT_ACTION
            elif action_space == "pyautogui":
                self.system_message = SYS_PROMPT_IN_SCREENSHOT_OUT_CODE
            else:
                raise ValueError("Invalid action space: " + action_space)
        elif observation_type == "a11y_tree":
            if action_space == "computer_13":
                self.system_message = SYS_PROMPT_IN_A11Y_OUT_ACTION
            elif action_space == "pyautogui":
                self.system_message = SYS_PROMPT_IN_A11Y_OUT_CODE
            else:
                raise ValueError("Invalid action space: " + action_space)
        elif observation_type == "screenshot_a11y_tree":
            if action_space == "computer_13":
                self.system_message = SYS_PROMPT_IN_BOTH_OUT_ACTION
            elif action_space == "pyautogui":
                self.system_message = SYS_PROMPT_IN_BOTH_OUT_CODE
            else:
                raise ValueError("Invalid action space: " + action_space)
        elif observation_type == "som":
            if action_space == "computer_13":
                raise ValueError("Invalid action space: " + action_space)
            elif action_space == "pyautogui":
                self.system_message = SYS_PROMPT_IN_SOM_OUT_TAG
            else:
                raise ValueError("Invalid action space: " + action_space)
        else:
            raise ValueError("Invalid experiment type: " + observation_type)
        
        self.system_message = self.system_message.format(CLIENT_PASSWORD=self.client_password)

        # For Qwen3-VL models, use a simplified OSWorld-style approach:
        # - Tell model screen is 1000x1000 (normalized coordinate space)
        # - Model outputs coordinates in 0-999 range
        # - We always scale from 0-999 to actual screen dimensions
        # This is simpler and more reliable than trying to detect coordinate space
        if self.is_qwen3_vl:
            logger.info(f"Qwen3-VL model detected: {model} - using 1000x1000 normalized coord space")
            # Use the simplified prompt with 1000x1000 normalized space
            self.system_message = QWEN3_VL_SYSTEM_PROMPT.format(CLIENT_PASSWORD=self.client_password)

    def predict(self, instruction: str, obs: Dict) -> List:
        """
        Predict the next action(s) based on the current observation.
        """
        # Detect if stuck feedback is present in the instruction
        if "STUCK LOOP DETECTED" in instruction:
            logger.warning("[LoopDetection] === PROCESSING STUCK INSTRUCTION ===")
            logger.info("[LoopDetection] LLM will receive recovery guidance with coordinate suggestions")
        elif "COORDINATE ANALYSIS" in instruction:
            logger.info("[LoopDetection] Instruction contains coordinate guidance from a11y tree")

        # For Qwen3-VL, preprocess image to maintain aspect ratio for better model performance
        # Model outputs in 0-999 normalized space, we scale to actual screen dimensions
        qwen_base64_image = None
        if self.is_qwen3_vl and self.observation_type in ["screenshot", "screenshot_a11y_tree"]:
            (
                qwen_base64_image,
                self._qwen_original_width,
                self._qwen_original_height,
                self._qwen_processed_width,
                self._qwen_processed_height
            ) = preprocess_image_for_qwen(obs["screenshot"])
            logger.info(
                f"Qwen3-VL image preprocessed: {self._qwen_original_width}x{self._qwen_original_height} -> "
                f"{self._qwen_processed_width}x{self._qwen_processed_height} (coords will be 0-999 -> screen)"
            )

        # Build system message with instruction
        system_message = self.system_message + "\nYou are asked to complete the following task: {}".format(instruction)

        # Prepare the payload for the API call
        messages = []
        masks = None

        messages.append({
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_message
                },
            ]
        })

        # Append trajectory
        assert len(self.observations) == len(self.actions) and len(self.actions) == len(self.thoughts) \
            , "The number of observations and actions should be the same."

        if len(self.observations) > self.max_trajectory_length:
            if self.max_trajectory_length == 0:
                _observations = []
                _actions = []
                _thoughts = []
            else:
                _observations = self.observations[-self.max_trajectory_length:]
                _actions = self.actions[-self.max_trajectory_length:]
                _thoughts = self.thoughts[-self.max_trajectory_length:]
        else:
            _observations = self.observations
            _actions = self.actions
            _thoughts = self.thoughts

        for previous_obs, previous_action, previous_thought in zip(_observations, _actions, _thoughts):

            # {{{1
            # NOTE: For stateless operation, historical observations may have screenshot=None
            # In that case, we only include the accessibility tree (text-only history)
            _screenshot = previous_obs.get("screenshot")
            _linearized_accessibility_tree = previous_obs.get("accessibility_tree")

            if self.observation_type == "screenshot_a11y_tree":
                # If screenshot is None (stateless mode), use text-only history
                if _screenshot is None:
                    if _linearized_accessibility_tree:
                        messages.append({
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Given the info from accessibility tree as below:\n{}\nWhat's the next step that you will do to help with the task?".format(
                                        _linearized_accessibility_tree)
                                }
                            ]
                        })
                    else:
                        messages.append({
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "What's the next step that you will do to help with the task?"
                                }
                            ]
                        })
                else:
                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Given the screenshot and info from accessibility tree as below:\n{}\nWhat's the next step that you will do to help with the task?".format(
                                    _linearized_accessibility_tree)
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{_screenshot}",
                                    "detail": "high"
                                }
                            }
                        ]
                    })
            elif self.observation_type in ["som"]:
                # If screenshot is None (stateless mode), skip image
                if _screenshot is None:
                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "What's the next step that you will do to help with the task?"
                            }
                        ]
                    })
                else:
                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Given the tagged screenshot as below. What's the next step that you will do to help with the task?"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{_screenshot}",
                                    "detail": "high"
                                }
                            }
                        ]
                    })
            elif self.observation_type == "screenshot":
                # If screenshot is None (stateless mode), skip image
                if _screenshot is None:
                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "What's the next step that you will do to help with the task?"
                            }
                        ]
                    })
                else:
                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Given the screenshot as below. What's the next step that you will do to help with the task?"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{_screenshot}",
                                    "detail": "high"
                                }
                            }
                        ]
                    })
            elif self.observation_type == "a11y_tree":
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Given the info from accessibility tree as below:\n{}\nWhat's the next step that you will do to help with the task?".format(
                                _linearized_accessibility_tree)
                        }
                    ]
                })
            else:
                raise ValueError("Invalid observation_type type: " + self.observation_type)  # 1}}}

            messages.append({
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": previous_thought.strip() if len(previous_thought) > 0 else "No valid action"
                    },
                ]
            })

        # {{{1
        if self.observation_type in ["screenshot", "screenshot_a11y_tree"]:
            # For Qwen3-VL, use the already-preprocessed image from earlier
            if self.is_qwen3_vl and qwen_base64_image is not None:
                base64_image = qwen_base64_image
            else:
                base64_image = encode_image(obs["screenshot"])

            # Handle accessibility tree - may be missing if green agent doesn't provide it
            linearized_accessibility_tree = None
            if self.observation_type == "screenshot_a11y_tree":
                if "accessibility_tree" in obs and obs["accessibility_tree"]:
                    linearized_accessibility_tree = linearize_accessibility_tree(
                        accessibility_tree=obs["accessibility_tree"],
                        platform=self.platform
                    )
                    logger.info(f"Linearized accessibility tree: {len(linearized_accessibility_tree)} chars")
                    # Log first 10 lines and any menu-related items for debugging
                    lines = linearized_accessibility_tree.split('\n')
                    logger.info(f"A11Y first 5 lines: {lines[:5]}")
                    for line in lines:
                        line_lower = line.lower()
                        if 'menu' in line_lower or 'remove' in line_lower or 'favorite' in line_lower or 'new window' in line_lower:
                            logger.info(f"A11Y MENU: {line}")
                else:
                    logger.warning(
                        "observation_type is 'screenshot_a11y_tree' but no accessibility_tree provided. "
                        "Falling back to screenshot-only mode for this step."
                    )
            else:
                logger.info(f"Skipping accessibility tree (observation_type={self.observation_type})")

            logger.debug("LINEAR AT: %s", linearized_accessibility_tree)

            if linearized_accessibility_tree:
                linearized_accessibility_tree = trim_accessibility_tree(linearized_accessibility_tree,
                                                                        self.a11y_tree_max_tokens)

            if self.observation_type == "screenshot_a11y_tree":
                self.observations.append({
                    "screenshot": base64_image,
                    "accessibility_tree": linearized_accessibility_tree
                })
            else:
                self.observations.append({
                    "screenshot": base64_image,
                    "accessibility_tree": None
                })

            # Build message text based on whether accessibility tree is available
            if linearized_accessibility_tree:
                message_text = "Given the screenshot and info from accessibility tree as below:\n{}\nWhat's the next step that you will do to help with the task?".format(
                    linearized_accessibility_tree)
                logger.info(f"Message includes accessibility tree ({len(linearized_accessibility_tree)} chars)")
            else:
                message_text = "Given the screenshot as below. What's the next step that you will do to help with the task?"
                logger.info("Message does NOT include accessibility tree")

            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": message_text
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            })
        elif self.observation_type == "a11y_tree":
            linearized_accessibility_tree = linearize_accessibility_tree(accessibility_tree=obs["accessibility_tree"],
                                                                         platform=self.platform)
            logger.debug("LINEAR AT: %s", linearized_accessibility_tree)

            if linearized_accessibility_tree:
                linearized_accessibility_tree = trim_accessibility_tree(linearized_accessibility_tree,
                                                                        self.a11y_tree_max_tokens)

            self.observations.append({
                "screenshot": None,
                "accessibility_tree": linearized_accessibility_tree
            })

            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Given the info from accessibility tree as below:\n{}\nWhat's the next step that you will do to help with the task?".format(
                            linearized_accessibility_tree)
                    }
                ]
            })
        elif self.observation_type == "som":
            # Add som to the screenshot
            masks, drew_nodes, tagged_screenshot, linearized_accessibility_tree = tag_screenshot(obs["screenshot"], obs[
                "accessibility_tree"], self.platform)
            base64_image = encode_image(tagged_screenshot)
            logger.debug("LINEAR AT: %s", linearized_accessibility_tree)

            if linearized_accessibility_tree:
                linearized_accessibility_tree = trim_accessibility_tree(linearized_accessibility_tree,
                                                                        self.a11y_tree_max_tokens)

            self.observations.append({
                "screenshot": base64_image,
                "accessibility_tree": linearized_accessibility_tree
            })

            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Given the tagged screenshot and info from accessibility tree as below:\n{}\nWhat's the next step that you will do to help with the task?".format(
                            linearized_accessibility_tree)
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            })
        else:
            raise ValueError("Invalid observation_type type: " + self.observation_type)  # 1}}}

        # with open("messages.json", "w") as f:
        #     f.write(json.dumps(messages, indent=4))

        # logger.info("PROMPT: %s", messages)

        try:
            response = self.call_llm({
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
                "temperature": self.temperature
            })
        except Exception as e:
            logger.error("Failed to call" + self.model + ", Error: " + str(e))
            response = ""

        logger.info("RESPONSE: %s", response)

        try:
            actions = self.parse_actions(response, masks)
            self.thoughts.append(response)
        except ValueError as e:
            print("Failed to parse action from response", e)
            actions = None
            self.thoughts.append("")

        return response, actions

    @backoff.on_exception(
        backoff.constant,
        # here you should add more model exceptions as you want,
        # but you are forbidden to add "Exception", that is, a common type of exception
        # because we want to catch this kind of Exception in the outside to ensure each example won't exceed the time limit
        (
                # General exceptions
                SSLError,

                # OpenAI exceptions
                openai.RateLimitError,
                openai.BadRequestError,
                openai.InternalServerError,

                # Google exceptions
                InvalidArgument,
                ResourceExhausted,
                InternalServerError,
                BadRequest,

                # Groq exceptions
                # todo: check
        ),
        interval=30,
        max_tries=10
    )
    def call_llm(self, payload):

        if payload['model'].startswith("azure-gpt-4o"):


            #.env config example :
            # AZURE_OPENAI_API_BASE=YOUR_API_BASE
            # AZURE_OPENAI_DEPLOYMENT=YOUR_DEPLOYMENT
            # AZURE_OPENAI_API_VERSION=YOUR_API_VERSION
            # AZURE_OPENAI_MODEL=gpt-4o-mini
            # AZURE_OPENAI_API_KEY={{YOUR_API_KEY}}
            # AZURE_OPENAI_ENDPOINT=${AZURE_OPENAI_API_BASE}/openai/deployments/${AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version=${AZURE_OPENAI_API_VERSION}


            # Load environment variables
            load_dotenv()
            api_key = get_azure_openai_api_key()
            openai_endpoint = get_azure_openai_endpoint()
            #logger.info("Openai endpoint: %s", openai_endpoint)

            headers = {
                "Content-Type": "application/json",
                "api-key": api_key
            }
            logger.info("Generating content with GPT model: %s", payload['model'])
            response = requests.post(
                openai_endpoint,
                headers=headers,
                json=payload
            )

            if response.status_code != 200:
                if response.json()['error']['code'] == "context_length_exceeded":
                    logger.error("Context length exceeded. Retrying with a smaller context.")
                    payload["messages"] = [payload["messages"][0]] + payload["messages"][-1:]
                    retry_response = requests.post(
                        openai_endpoint,
                        headers=headers,
                        json=payload
                    )
                    if retry_response.status_code != 200:
                         logger.error(
                            "Failed to call LLM even after attempt on shortening the history: " + retry_response.text)
                         return ""

                logger.error("Failed to call LLM: " + response.text)
                time.sleep(5)
                return ""
            else:
                return response.json()['choices'][0]['message']['content']
        elif self.model.startswith("gpt"):
            # Support custom OpenAI base URL via environment variable
            base_url = get_openai_base_url()
            # Smart handling: avoid duplicate /v1 if base_url already ends with /v1
            api_url = f"{base_url}/chat/completions" if base_url.endswith('/v1') else f"{base_url}/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
            }
            logger.info("Generating content with GPT model: %s", self.model)

            # GPT-5 models have different parameter requirements
            if self.model.startswith("gpt-5"):
                # max_tokens -> max_completion_tokens
                if "max_tokens" in payload:
                    payload["max_completion_tokens"] = payload.pop("max_tokens")

                # GPT-5.1 defaults to reasoning_effort="none", which supports temperature/top_p
                # Other reasoning levels (low/medium/high) do NOT support temperature/top_p
                reasoning_effort = get_reasoning_effort()
                payload["reasoning_effort"] = reasoning_effort

                # temperature and top_p are only supported with reasoning_effort="none"
                if reasoning_effort != "none":
                    payload.pop("temperature", None)
                    payload.pop("top_p", None)

            response = requests.post(
                api_url,
                headers=headers,
                json=payload
            )

            if response.status_code != 200:
                if response.json()['error']['code'] == "context_length_exceeded":
                    logger.error("Context length exceeded. Retrying with a smaller context.")
                    payload["messages"] = [payload["messages"][0]] + payload["messages"][-1:]
                    retry_response = requests.post(
                        api_url,
                        headers=headers,
                        json=payload
                    )
                    if retry_response.status_code != 200:
                        logger.error(
                            "Failed to call LLM even after attempt on shortening the history: " + retry_response.text)
                        return ""

                logger.error("Failed to call LLM: " + response.text)
                time.sleep(5)
                return ""
            else:
                return response.json()['choices'][0]['message']['content']

        elif self.model.startswith("claude"):
            messages = payload["messages"]
            max_tokens = payload["max_tokens"]
            top_p = payload["top_p"]
            temperature = payload["temperature"]

            claude_messages = []

            for i, message in enumerate(messages):
                claude_message = {
                    "role": message["role"],
                    "content": []
                }
                assert len(message["content"]) in [1, 2], "One text, or one text with one image"
                for part in message["content"]:

                    if part['type'] == "image_url":
                        image_source = {}
                        image_source["type"] = "base64"
                        image_source["media_type"] = "image/png"
                        image_source["data"] = part['image_url']['url'].replace("data:image/png;base64,", "")
                        claude_message['content'].append({"type": "image", "source": image_source})

                    if part['type'] == "text":
                        claude_message['content'].append({"type": "text", "text": part['text']})

                claude_messages.append(claude_message)

            # the claude not support system message in our endpoint, so we concatenate it at the first user message
            if claude_messages[0]['role'] == "system":
                claude_system_message_item = claude_messages[0]['content'][0]
                claude_messages[1]['content'].insert(0, claude_system_message_item)
                claude_messages.pop(0)

            logger.debug("CLAUDE MESSAGE: %s", repr(claude_messages))

            headers = {
                "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }

            # Claude API doesn't allow both temperature and top_p for newer models
            # Use only temperature for consistency
            payload = {
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": claude_messages,
                "temperature": temperature,
            }

            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload
            )

            if response.status_code != 200:

                logger.error("Failed to call LLM: " + response.text)
                time.sleep(5)
                return ""
            else:
                return response.json()['content'][0]['text']

        elif self.model.startswith("mistral"):
            messages = payload["messages"]
            max_tokens = payload["max_tokens"]
            top_p = payload["top_p"]
            temperature = payload["temperature"]

            assert self.observation_type in pure_text_settings, f"The model {self.model} can only support text-based input, please consider change based model or settings"

            mistral_messages = []

            for i, message in enumerate(messages):
                mistral_message = {
                    "role": message["role"],
                    "content": ""
                }

                for part in message["content"]:
                    mistral_message['content'] = part['text'] if part['type'] == "text" else ""

                mistral_messages.append(mistral_message)

            from openai import OpenAI

            client = OpenAI(api_key=os.environ["TOGETHER_API_KEY"],
                            base_url='https://api.together.xyz',
                            )

            flag = 0
            while True:
                try:
                    if flag > 20:
                        break
                    logger.info("Generating content with model: %s", self.model)
                    response = client.chat.completions.create(
                        messages=mistral_messages,
                        model=self.model,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        temperature=temperature
                    )
                    break
                except:
                    if flag == 0:
                        mistral_messages = [mistral_messages[0]] + mistral_messages[-1:]
                    else:
                        mistral_messages[-1]["content"] = ' '.join(mistral_messages[-1]["content"].split()[:-500])
                    flag = flag + 1

            try:
                return response.choices[0].message.content
            except Exception as e:
                print("Failed to call LLM: " + str(e))
                return ""

        elif self.model.startswith("THUDM"):
            # THUDM/cogagent-chat-hf
            messages = payload["messages"]
            max_tokens = payload["max_tokens"]
            top_p = payload["top_p"]
            temperature = payload["temperature"]

            cog_messages = []

            for i, message in enumerate(messages):
                cog_message = {
                    "role": message["role"],
                    "content": []
                }

                for part in message["content"]:
                    if part['type'] == "image_url":
                        cog_message['content'].append(
                            {"type": "image_url", "image_url": {"url": part['image_url']['url']}})

                    if part['type'] == "text":
                        cog_message['content'].append({"type": "text", "text": part['text']})

                cog_messages.append(cog_message)

            # the cogagent not support system message in our endpoint, so we concatenate it at the first user message
            if cog_messages[0]['role'] == "system":
                cog_system_message_item = cog_messages[0]['content'][0]
                cog_messages[1]['content'].insert(0, cog_system_message_item)
                cog_messages.pop(0)

            payload = {
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": cog_messages,
                "temperature": temperature,
                "top_p": top_p
            }

            base_url = "http://127.0.0.1:8000"

            response = requests.post(f"{base_url}/v1/chat/completions", json=payload, stream=False)
            if response.status_code == 200:
                decoded_line = response.json()
                content = decoded_line.get("choices", [{}])[0].get("message", "").get("content", "")
                return content
            else:
                print("Failed to call LLM: ", response.status_code)
                return ""

        elif self.model in ["gemini-pro", "gemini-pro-vision"]:
            messages = payload["messages"]
            max_tokens = payload["max_tokens"]
            top_p = payload["top_p"]
            temperature = payload["temperature"]

            if self.model == "gemini-pro":
                assert self.observation_type in pure_text_settings, f"The model {self.model} can only support text-based input, please consider change based model or settings"

            gemini_messages = []
            for i, message in enumerate(messages):
                role_mapping = {
                    "assistant": "model",
                    "user": "user",
                    "system": "system"
                }
                gemini_message = {
                    "role": role_mapping[message["role"]],
                    "parts": []
                }
                assert len(message["content"]) in [1, 2], "One text, or one text with one image"

                # The gemini only support the last image as single image input
                if i == len(messages) - 1:
                    for part in message["content"]:
                        gemini_message['parts'].append(part['text']) if part['type'] == "text" \
                            else gemini_message['parts'].append(encoded_img_to_pil_img(part['image_url']['url']))
                else:
                    for part in message["content"]:
                        gemini_message['parts'].append(part['text']) if part['type'] == "text" else None

                gemini_messages.append(gemini_message)

            # the gemini not support system message in our endpoint, so we concatenate it at the first user message
            if gemini_messages[0]['role'] == "system":
                gemini_messages[1]['parts'][0] = gemini_messages[0]['parts'][0] + "\n" + gemini_messages[1]['parts'][0]
                gemini_messages.pop(0)

            # since the gemini-pro-vision donnot support multi-turn message
            if self.model == "gemini-pro-vision":
                message_history_str = ""
                for message in gemini_messages:
                    message_history_str += "<|" + message['role'] + "|>\n" + message['parts'][0] + "\n"
                gemini_messages = [{"role": "user", "parts": [message_history_str, gemini_messages[-1]['parts'][1]]}]
                # gemini_messages[-1]['parts'][1].save("output.png", "PNG")

            # print(gemini_messages)
            api_key = get_genai_api_key()
            assert api_key is not None, "Please set the GENAI_API_KEY environment variable"
            genai.configure(api_key=api_key)
            logger.info("Generating content with Gemini model: %s", self.model)
            request_options = {"timeout": 120}
            gemini_model = genai.GenerativeModel(self.model)

            response = gemini_model.generate_content(
                gemini_messages,
                generation_config={
                    "candidate_count": 1,
                    # "max_output_tokens": max_tokens,
                    "top_p": top_p,
                    "temperature": temperature
                },
                safety_settings={
                    "harassment": "block_none",
                    "hate": "block_none",
                    "sex": "block_none",
                    "danger": "block_none"
                },
                request_options=request_options
            )
            return response.text

        elif self.model.startswith("gemini"):
            messages = payload["messages"]
            max_tokens = payload["max_tokens"]
            top_p = payload["top_p"]
            temperature = payload["temperature"]

            gemini_messages = []
            for i, message in enumerate(messages):
                role_mapping = {
                    "assistant": "model",
                    "user": "user",
                    "system": "system"
                }
                assert len(message["content"]) in [1, 2], "One text, or one text with one image"
                gemini_message = {
                    "role": role_mapping[message["role"]],
                    "parts": []
                }

                # The gemini only support the last image as single image input
                for part in message["content"]:

                    if part['type'] == "image_url":
                        # Put the image at the beginning of the message
                        gemini_message['parts'].insert(0, encoded_img_to_pil_img(part['image_url']['url']))
                    elif part['type'] == "text":
                        gemini_message['parts'].append(part['text'])
                    else:
                        raise ValueError("Invalid content type: " + part['type'])

                gemini_messages.append(gemini_message)

            # the system message of gemini-1.5-pro-latest need to be inputted through model initialization parameter
            system_instruction = None
            if gemini_messages[0]['role'] == "system":
                system_instruction = gemini_messages[0]['parts'][0]
                gemini_messages.pop(0)

            api_key = get_genai_api_key()
            assert api_key is not None, "Please set the GENAI_API_KEY environment variable"
            genai.configure(api_key=api_key)
            logger.info("Generating content with Gemini model: %s", self.model)
            request_options = {"timeout": 120}
            gemini_model = genai.GenerativeModel(
                self.model,
                system_instruction=system_instruction
            )

            with open("response.json", "w") as f:
                messages_to_save = []
                for message in gemini_messages:
                    messages_to_save.append({
                        "role": message["role"],
                        "content": [part if isinstance(part, str) else "image" for part in message["parts"]]
                    })
                json.dump(messages_to_save, f, indent=4)

            response = gemini_model.generate_content(
                gemini_messages,
                generation_config={
                    "candidate_count": 1,
                    # "max_output_tokens": max_tokens,
                    "top_p": top_p,
                    "temperature": temperature
                },
                safety_settings={
                    "harassment": "block_none",
                    "hate": "block_none",
                    "sex": "block_none",
                    "danger": "block_none"
                },
                request_options=request_options
            )

            return response.text

        elif self.model.startswith("groq-") or self.model in GROQ_MODELS:
            # Groq API with Llama 4 vision models
            # Supports: llama-4-maverick, llama-4-scout (both multimodal)
            messages = payload["messages"]
            max_tokens = payload["max_tokens"]
            top_p = payload["top_p"]
            temperature = payload["temperature"]

            # Map user-friendly names to actual Groq model IDs
            if self.model.startswith("groq-"):
                model_name = self.model[5:]  # Remove "groq-" prefix
                groq_model = GROQ_MODEL_MAPPING.get(model_name, model_name)
            else:
                groq_model = self.model

            # Build messages with vision support (OpenAI-compatible format)
            groq_messages = []
            for message in messages:
                groq_message = {
                    "role": message["role"],
                    "content": []
                }
                for part in message["content"]:
                    if part['type'] == "text":
                        groq_message['content'].append({
                            "type": "text",
                            "text": part['text']
                        })
                    elif part['type'] == "image_url":
                        # Groq uses OpenAI-compatible image format
                        groq_message['content'].append({
                            "type": "image_url",
                            "image_url": {
                                "url": part['image_url']['url']
                            }
                        })
                groq_messages.append(groq_message)

            # Initialize Groq client
            if Groq is None:
                raise ImportError("groq package is not installed. Run: pip install groq")

            api_key = get_groq_api_key()
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable is required for Groq models")

            client = Groq(api_key=api_key)

            # Retry logic with context reduction on failure
            flag = 0
            response = None
            while flag <= 5:  # Reduced retries for vision models
                try:
                    logger.info("Generating content with Groq model: %s", groq_model)
                    response = client.chat.completions.create(
                        messages=groq_messages,
                        model=groq_model,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        temperature=temperature
                    )
                    break
                except Exception as e:
                    logger.warning("Groq API call failed (attempt %d): %s", flag + 1, str(e))
                    flag += 1
                    time.sleep(2)  # Brief delay before retry

            if response is None:
                logger.error("Failed to get response from Groq after %d retries", flag)
                return ""

            try:
                return response.choices[0].message.content
            except Exception as e:
                logger.error("Failed to parse Groq response: %s", str(e))
                return ""

        elif self.model.startswith("qwen3-vl") or self.model in QWEN3_VL_MODELS:
            # Qwen3-VL on Vertex AI (OpenAI-compatible API)
            # Deploy with: python scripts/vertex-ai/deploy_qwen3_vl.py
            messages = payload["messages"]
            max_tokens = payload["max_tokens"]
            top_p = payload["top_p"]
            temperature = payload["temperature"]

            # Get endpoint URL
            endpoint_url = get_qwen3_vl_endpoint_url()
            if not endpoint_url:
                raise ValueError(
                    "QWEN3_VL_ENDPOINT_URL environment variable is required. "
                    "Deploy the model first: python scripts/vertex-ai/deploy_qwen3_vl.py --project YOUR_PROJECT"
                )

            # Map user-friendly names to actual model names
            vertex_model = QWEN3_VL_MODEL_MAPPING.get(self.model, self.model)

            # Build OpenAI-compatible messages (Vertex AI uses this format)
            qwen3_messages = []
            for message in messages:
                qwen3_message = {
                    "role": message["role"],
                    "content": []
                }
                for part in message["content"]:
                    if part['type'] == "text":
                        qwen3_message['content'].append({
                            "type": "text",
                            "text": part['text']
                        })
                    elif part['type'] == "image_url":
                        qwen3_message['content'].append({
                            "type": "image_url",
                            "image_url": {"url": part['image_url']['url']}
                        })
                qwen3_messages.append(qwen3_message)

            # Use Vertex AI prediction API with chatCompletions format
            # Get GCP access token for authentication
            try:
                import google.auth
                import google.auth.transport.requests

                credentials, project = google.auth.default()
                auth_request = google.auth.transport.requests.Request()
                credentials.refresh(auth_request)
                access_token = credentials.token
            except Exception as e:
                raise ValueError(
                    f"Failed to get GCP credentials: {e}. "
                    "Run 'gcloud auth application-default login' to authenticate."
                )

            # Build request payload using Vertex AI prediction format
            # See: https://cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.endpoints/predict
            predict_payload = {
                "instances": [
                    {
                        "@requestFormat": "chatCompletions",
                        "messages": qwen3_messages,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                    }
                ]
            }

            # For dedicated Vertex AI Model Garden endpoints, the URL format is:
            # https://{endpoint_id}.{location}-{project_number}.prediction.vertexai.goog/v1/projects/{project_id}/locations/{location}/endpoints/{endpoint_id}:predict
            #
            # The endpoint URL env var should contain the full predict URL, or we construct it
            # Expected format in env: https://mg-endpoint-xxx.us-central1-12345.prediction.vertexai.goog
            import re
            base_url = endpoint_url.rstrip('/')
            if base_url.endswith('/v1'):
                base_url = base_url[:-3]

            # Parse the endpoint DNS to extract components
            # Format: {endpoint_id}.{location}-{project_number}.prediction.vertexai.goog
            # Example: mg-endpoint-xxx.us-central1-750082808015.prediction.vertexai.goog
            # Note: location can contain hyphens (e.g., us-central1), so we match until the last hyphen before project_number
            dns_match = re.match(r'https://([^.]+)\.(.+)-(\d+)\.prediction\.vertexai\.goog', base_url)
            if dns_match:
                endpoint_id = dns_match.group(1)
                location = dns_match.group(2)
                project_number = dns_match.group(3)
                # We need the project ID, not number - try to get from env or use number
                project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", f"projects/{project_number}")
                predict_url = f"{base_url}/v1/projects/{project_id}/locations/{location}/endpoints/{endpoint_id}:predict"
            else:
                # Fallback: assume the URL is already complete or use simple format
                predict_url = f"{base_url}:predict"

            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            }

            try:
                logger.info("Generating content with Qwen3-VL model: %s at %s", vertex_model, predict_url)
                response = requests.post(
                    predict_url,
                    headers=headers,
                    json=predict_payload,
                    timeout=3600,
                )

                if response.status_code != 200:
                    logger.error("Qwen3-VL API error %d: %s", response.status_code, response.text)
                    return ""

                result = response.json()
                # Parse the response - format may vary
                logger.debug("Qwen3-VL response: %s", result)

                # Try to extract content from various response formats
                content = None
                if "predictions" in result:
                    predictions = result["predictions"]
                    # predictions can be a dict (single response) or list
                    if isinstance(predictions, dict):
                        # OpenAI-compatible format with choices
                        if "choices" in predictions:
                            content = predictions["choices"][0]["message"]["content"]
                        # Direct content
                        elif "content" in predictions:
                            content = predictions["content"]
                    elif isinstance(predictions, list):
                        prediction = predictions[0]
                        if isinstance(prediction, str):
                            content = prediction
                        elif isinstance(prediction, dict):
                            if "choices" in prediction:
                                content = prediction["choices"][0]["message"]["content"]
                            elif "content" in prediction:
                                content = prediction["content"]
                            elif "message" in prediction:
                                content = prediction["message"].get("content", str(prediction))
                            else:
                                content = str(prediction)
                        else:
                            content = str(prediction)
                    else:
                        content = str(predictions)
                else:
                    logger.error("Unexpected response format: %s", result)
                    return ""

                # Scale Qwen3-VL coordinates from 0-999 normalized space to actual screen
                # Model is told screen is 1000x1000, always outputs in 0-999 range
                # We scale: x_actual = x * screen_width / 999, y_actual = y * screen_height / 999
                if content and self._qwen_original_width and self._qwen_original_height:
                    original_content = content
                    scaled = False
                    x_scale = self._qwen_original_width / 999.0
                    y_scale = self._qwen_original_height / 999.0

                    if 'pyautogui.' in content:
                        # Scale pyautogui coordinates from 0-999 to actual screen
                        content = scale_qwen_coordinates(content, self._qwen_original_width, self._qwen_original_height)
                        scaled = (content != original_content)

                    elif '"op"' in content:
                        # Handle JSON format - scale coordinates from 0-999 to screen
                        try:
                            # Extract JSON from markdown code block if present
                            json_match = re.search(r'```(?:json)?\s*\n?({.*?})\s*```', content, re.DOTALL)
                            if json_match:
                                json_str = json_match.group(1)
                            elif content.strip().startswith('{'):
                                json_str = content.strip()
                            else:
                                json_str = None

                            if json_str:
                                action_dict = json.loads(json_str)
                                args = action_dict.get("args", {})
                                x_val, y_val = None, None

                                # Handle array format: {"x": [14, 124]}
                                if "x" in args and isinstance(args["x"], list):
                                    coords = args["x"]
                                    if len(coords) >= 2:
                                        x_val, y_val = coords[0], coords[1]
                                # Handle separate x, y format
                                elif "x" in args and "y" in args:
                                    x_val, y_val = args["x"], args["y"]

                                if x_val is not None and y_val is not None:
                                    # Always scale from 0-999 to actual screen dimensions
                                    new_x = int(round(x_val * x_scale))
                                    new_y = int(round(y_val * y_scale))
                                    logger.info(f"Qwen JSON coords scaled (0-999 -> screen): ({x_val}, {y_val}) -> ({new_x}, {new_y})")

                                    action_dict["args"] = {"x": new_x, "y": new_y}
                                    content = json.dumps(action_dict)
                                    scaled = True
                        except (json.JSONDecodeError, KeyError, TypeError) as e:
                            logger.warning(f"Failed to parse JSON for scaling: {e}")

                    if scaled:
                        logger.info(
                            "Qwen3-VL coordinates scaled (0-999 -> %dx%d): %s -> %s",
                            self._qwen_original_width, self._qwen_original_height,
                            original_content.strip()[:80], content.strip()[:80]
                        )
                    else:
                        logger.info("Qwen3-VL response (no coords to scale): %s", content.strip()[:100])
                return content or ""

            except Exception as e:
                logger.error("Qwen3-VL API call failed: %s", str(e))
                return ""

        elif self.model.startswith("qwen"):
            messages = payload["messages"]
            max_tokens = payload["max_tokens"]
            top_p = payload["top_p"]
            temperature = payload["temperature"]

            qwen_messages = []

            for i, message in enumerate(messages):
                qwen_message = {
                    "role": message["role"],
                    "content": []
                }
                assert len(message["content"]) in [1, 2], "One text, or one text with one image"
                for part in message["content"]:
                    qwen_message['content'].append(
                        {"image": "file://" + save_to_tmp_img_file(part['image_url']['url'])}) if part[
                                                                                                      'type'] == "image_url" else None
                    qwen_message['content'].append({"text": part['text']}) if part['type'] == "text" else None

                qwen_messages.append(qwen_message)

            flag = 0
            while True:
                try:
                    if flag > 20:
                        break
                    logger.info("Generating content with model: %s", self.model)

                    if self.model in ["qwen-vl-plus", "qwen-vl-max"]:
                        response = dashscope.MultiModalConversation.call(
                            model=self.model,
                            messages=qwen_messages,
                            result_format="message",
                            max_length=max_tokens,
                            top_p=top_p,
                            temperature=temperature
                        )

                    elif self.model in ["qwen-turbo", "qwen-plus", "qwen-max", "qwen-max-0428", "qwen-max-0403",
                                        "qwen-max-0107", "qwen-max-longcontext"]:
                        response = dashscope.Generation.call(
                            model=self.model,
                            messages=qwen_messages,
                            result_format="message",
                            max_length=max_tokens,
                            top_p=top_p,
                            temperature=temperature
                        )

                    else:
                        raise ValueError("Invalid model: " + self.model)

                    if response.status_code == HTTPStatus.OK:
                        break
                    else:
                        logger.error('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                            response.request_id, response.status_code,
                            response.code, response.message
                        ))
                        raise Exception("Failed to call LLM: " + response.message)
                except:
                    if flag == 0:
                        qwen_messages = [qwen_messages[0]] + qwen_messages[-1:]
                    else:
                        for i in range(len(qwen_messages[-1]["content"])):
                            if "text" in qwen_messages[-1]["content"][i]:
                                qwen_messages[-1]["content"][i]["text"] = ' '.join(
                                    qwen_messages[-1]["content"][i]["text"].split()[:-500])
                    flag = flag + 1

            try:
                if self.model in ["qwen-vl-plus", "qwen-vl-max"]:
                    content = response['output']['choices'][0]['message']['content'][0]['text']
                    # Scale Qwen VL coordinates from 0-999 normalized space to actual screen pixels
                    original_content = content
                    content = scale_qwen_coordinates(content)
                    if content != original_content:
                        logger.info("Qwen VL coordinates scaled: %s -> %s",
                                    original_content.strip()[:100], content.strip()[:100])
                    return content
                else:
                    return response['output']['choices'][0]['message']['content']

            except Exception as e:
                print("Failed to call LLM: " + str(e))
                return ""

        elif self.model.startswith("langchain"):
            # EXPERIMENTAL: LangChain with DeepAgents integration
            # Model format: langchain-<model> e.g. langchain-gpt-4o
            # WARNING: This is experimental and not recommended for production.
            # Web search adds latency with limited benefit for GUI-based tasks.
            logger.warning(
                "EXPERIMENTAL: Using LangChain agent (%s). This is experimental and "
                "not recommended for production. Consider using the model directly "
                "(e.g., 'gpt-4o' instead of 'langchain-gpt-4o').",
                self.model
            )
            if not langchain_available:
                raise ImportError("LangChain not available. Install with: pip install langchain langchain-openai langchain-core")
            # DeepAgents is optional - only needed for web search capability
            if not deepagents_available:
                logger.info("DeepAgents not available - web search disabled. Install with: pip install deepagents")

            messages = payload["messages"]
            max_tokens = payload["max_tokens"]
            temperature = payload["temperature"]

            # Parse the model string to extract provider and model name
            # Format: langchain-<model> where model can be like "gpt-4o" or "claude-3-sonnet"
            model_parts = self.model.split("-", 1)
            if len(model_parts) > 1:
                underlying_model = model_parts[1]  # e.g., "gpt-4o" from "langchain-gpt-4o"
            else:
                underlying_model = "gpt-4o"  # default

            # Determine API key and provider based on underlying model
            if underlying_model.startswith("gpt") or underlying_model.startswith("o1") or underlying_model.startswith("o3"):
                api_key = os.environ.get("OPENAI_API_KEY")
                model_provider = "openai"
            elif underlying_model.startswith("claude"):
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                model_provider = "anthropic"
            else:
                api_key = os.environ.get("OPENAI_API_KEY")  # default to OpenAI
                model_provider = "openai"

            # Initialize LangChain chat model with explicit provider
            llm = init_chat_model(
                model=underlying_model,
                model_provider=model_provider,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Optionally wrap with DeepAgents for web search capability
            if deepagents_available and tavily_available and os.environ.get("TAVILY_API_KEY"):
                logger.info("Tavily web search enabled - wrapping LLM with DeepAgents")
                tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

                def tavily_search(query: str, max_results: int = 5, include_raw_content: bool = False):
                    """Run a web search"""
                    logger.info(f"[Tavily] Executing web search: query='{query}', max_results={max_results}")
                    result = tavily_client.search(
                        query,
                        max_results=max_results,
                        include_raw_content=include_raw_content,
                        topic="general",
                    )
                    logger.info(f"[Tavily] Search completed, got {len(result.get('results', []))} results")
                    return result

                llm = create_deep_agent(model=llm, system_prompt="", tools=[tavily_search])

            # Convert messages to LangChain format
            langchain_messages = []
            for message in messages:
                role = message["role"]
                content_parts = message["content"]

                # Build content for LangChain message
                if len(content_parts) == 1 and content_parts[0]["type"] == "text":
                    # Simple text message
                    content = content_parts[0]["text"]
                else:
                    # Multimodal message with potential images
                    content = []
                    for part in content_parts:
                        if part["type"] == "text":
                            content.append({"type": "text", "text": part["text"]})
                        elif part["type"] == "image_url":
                            content.append({
                                "type": "image_url",
                                "image_url": {"url": part["image_url"]["url"]}
                            })

                if role == "system":
                    langchain_messages.append(SystemMessage(content=content))
                elif role == "user":
                    langchain_messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    from langchain_core.messages import AIMessage
                    langchain_messages.append(AIMessage(content=content))

            logger.info("Generating content with LangChain model: %s (underlying: %s)", self.model, underlying_model)

            try:
                # Call the LLM - handle both raw model and DeepAgent wrapped model
                if hasattr(llm, 'invoke') and callable(llm.invoke):
                    if deepagents_available and tavily_available and os.environ.get("TAVILY_API_KEY"):
                        # DeepAgent expects dict with messages key
                        response = llm.invoke({"messages": langchain_messages})
                    else:
                        # Raw LangChain model
                        response = llm.invoke(langchain_messages)

                    if hasattr(response, 'content'):
                        return response.content
                    else:
                        return str(response)
                else:
                    raise ValueError("LLM does not have invoke method")

            except Exception as e:
                logger.error("Failed to call LangChain LLM: %s", str(e))
                return ""

        else:
            raise ValueError("Invalid model: " + self.model)

    def parse_actions(self, response: str, masks=None):

        if self.observation_type in ["screenshot", "a11y_tree", "screenshot_a11y_tree"]:
            # parse from the response
            if self.action_space == "computer_13":
                actions = parse_actions_from_string(response)
            elif self.action_space == "pyautogui":
                actions = parse_code_from_string(response)
            else:
                raise ValueError("Invalid action space: " + self.action_space)

            self.actions.append(actions)

            return actions
        elif self.observation_type in ["som"]:
            # parse from the response
            if self.action_space == "computer_13":
                raise ValueError("Invalid action space: " + self.action_space)
            elif self.action_space == "pyautogui":
                actions = parse_code_from_som_string(response, masks)
            else:
                raise ValueError("Invalid action space: " + self.action_space)

            self.actions.append(actions)

            return actions

    def reset(self, _logger=None):
        global logger
        logger = _logger if _logger is not None else logging.getLogger("desktopenv.agent")

        self.thoughts = []
        self.actions = []
        self.observations = []
