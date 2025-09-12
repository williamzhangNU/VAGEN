# vagen/mllm_agent/model_interface/openai/model.py
import base64
import logging
import re
import os
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from zhipuai import ZhipuAI
from PIL import Image
import io
from vagen.inference.model_interface.base_model import BaseModelInterface
from .model_config import ZhipuModelConfig

logger = logging.getLogger(__name__)

class ZhipuModelInterface(BaseModelInterface):
    """Model interface for Zhipu API with Qwen format compatibility."""
    
    def __init__(self, config: ZhipuModelConfig):
        super().__init__(config)
        self.config = config
        
        # Initialize Zhipu client
        self.client = ZhipuAI(api_key=os.getenv("ZAI_API_KEY"))
        
        # Thread pool for batch processing
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        logger.info(f"Initialized Zhipu interface with model {config.model_name}")
    
    def generate(self, prompts: List[Any], **kwargs) -> List[Dict[str, Any]]:
        """Generate responses using Zhipu API."""
        # Process prompts into Zhipu message format
        formatted_requests = []
        
        for prompt in prompts:
            messages = self._convert_qwen_to_openai_format(prompt)
            messages = self._convert_to_zhipu_format(messages)
            formatted_requests.append(messages)
        
        # Make parallel API calls
        futures = []
        for messages in formatted_requests:
            future = self.executor.submit(
                self._single_api_call,
                messages,
                **kwargs
            )
            futures.append(future)
        
        # Collect results
        results = []
        for future in futures:
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"API call failed: {e}")
                results.append({
                    "text": f"Error: {str(e)}",
                    "error": str(e)
                })
        
        return results
    
    def _convert_qwen_to_openai_format(self, prompt: List[Dict]) -> List[Dict]:
        """
        Convert Qwen format messages to OpenAI format.
        
        Qwen format: Text with <image> placeholders + separate multi_modal_data
        OpenAI format: Structured content array with text and image objects
        """
        openai_messages = []
        
        for message in prompt:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            # Create OpenAI message structure
            openai_msg = {
                "role": role,
                "content": []
            }
            
            # Handle multimodal content
            if "multi_modal_data" in message and "<image>" in content:
                # Extract images from multi_modal_data
                images = []
                for key, values in message["multi_modal_data"].items():
                    if key == "<image>" or "image" in key.lower():
                        images.extend(values)
                
                # Split content by <image> placeholders
                parts = content.split("<image>")
                
                # Build content array alternating text and images
                for i, part in enumerate(parts):
                    # Add text part if not empty
                    if part.strip():
                        openai_msg["content"].append({
                            "type": "text",
                            "text": part
                        })
                    
                    # Add image if available (except for last part)
                    if i < len(parts) - 1 and i < len(images):
                        image_data = self._process_image_for_openai(images[i])
                        openai_msg["content"].append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        })
            else:
                # Text-only message
                openai_msg["content"].append({
                    "type": "text",
                    "text": content
                })
            
            openai_messages.append(openai_msg)
        return openai_messages
    def _convert_to_zhipu_format(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert messages to Zhipu format where EACH content part becomes its own
        {'role': 'user', 'content': [single_block]} message, in sequence.

        Rules:
        - 'system' -> single Zhipu message; content must be a list with one text block.
        - 'assistant' -> keep as a single message; pack text into one text block.
        - 'user':
          * if content is str -> one 'user' message with a single text block.
          * if content is list -> explode into N 'user' messages, each with exactly one block.
        - Skip empty strings and empty lists.
        - Preserve original order.
        """
        ALLOWED_BLOCK_TYPES = {"text", "image_url"}
        out: List[Dict[str, Any]] = []

        def _mk_text_block(text: str) -> Dict[str, Any]:
            t = (text or "").strip()
            if not t:
                return {}
            return {"type": "text", "text": t}

        for m in messages:
            role = m.get("role")
            content = m.get("content", "")

            if role == "system":
                # Normalize to single text block
                if isinstance(content, str):
                    block = _mk_text_block(content)
                    if block:
                        out.append({"role": "system", "content": [block]})
                elif isinstance(content, list):
                    # find first non-empty text block
                    blocks = [b for b in content if b.get("type") == "text" and (b.get("text") or "").strip()]
                    if blocks:
                        out.append({"role": "system", "content": [blocks[0]]})
                continue

            if role == "user":
                if isinstance(content, str):
                    block = _mk_text_block(content)
                    if block:
                        out.append({"role": "user", "content": [block]})
                elif isinstance(content, list):
                    # explode: one user message per part (text/image/file/...)
                    for b in content:
                        # normalize dicts like {"image_url": {...}} to {"type":"image_url","image_url":{...}}
                        if "type" not in b and len(b) == 1:
                            k = next(iter(b))
                            if k in ALLOWED_BLOCK_TYPES:
                                b = {"type": k, k: b[k]}
                        btype = b.get("type")
                        if btype == "text":
                            t = (b.get("text") or "").strip()
                            if not t:
                                continue
                            out.append({"role": "user", "content": [{"type": "text", "text": t}]})
                        elif btype in ALLOWED_BLOCK_TYPES:
                            # keep non-text block as-is (image_url/video_url/file_url)
                            out.append({"role": "user", "content": [b]})
                continue

        return out

    def _process_image_for_openai(self, image: Any) -> str:
        """Convert image to base64 for OpenAI API."""
        if isinstance(image, Image.Image):
            # Ensure RGB mode
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Resize if too large to save tokens
            max_size = 1024
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=85)
            return base64.b64encode(buffered.getvalue()).decode()
            
        elif isinstance(image, dict) and "__pil_image__" in image:
            from vagen.server.serial import deserialize_pil_image
            pil_image = deserialize_pil_image(image)
            return self._process_image_for_openai(pil_image)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def _single_api_call(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Make a single API call to Zhipu."""
        try:
            msg_kwargs = {
                "model": self.config.model_name,
                "messages": messages,
                "temperature": kwargs.get("temperature", self.config.temperature),
            }
            msg_kwargs["max_tokens"] = kwargs.get("max_tokens", self.config.max_tokens)
            response = self.client.chat.completions.create(**msg_kwargs)
            
            # Extract text response
            response_text = response.choices[0].message.content
            
            return {
                "text": response_text,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "finish_reason": response.choices[0].finish_reason
            }
            
        except Exception as e:
            logger.error(f"Zhipu API error: {e}")
            raise
    
    def format_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """
        Format prompt for compatibility.
        
        Since OpenAI uses structured messages, this returns a string representation
        of the messages for logging/debugging purposes.
        """
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Handle Qwen special tokens if present
            if role == "system":
                formatted.append(f"System: {content}")
            elif role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
        
        return "\n".join(formatted)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed information about the model."""
        info = super().get_model_info()
        
        info.update({
            "name": self.config.model_name,
            "type": "multimodal" if "vision" in self.config.model_name.lower() else "text",
            "supports_images": "vision" in self.config.model_name.lower() or "4o" in self.config.model_name,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "config_id": self.config.config_id()
        })
        
        return info