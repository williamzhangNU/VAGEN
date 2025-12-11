import os
import json
import time
from typing import List, Dict, Any
from abc import ABC, abstractmethod
import requests

from vagen.inference.model_interface.openai.model import OpenAIModelInterface
from vagen.inference.model_interface.openai.model_config import OpenAIModelConfig
from vagen.inference.model_interface.claude.model import ClaudeModelInterface
from vagen.inference.model_interface.claude.model_config import ClaudeModelConfig

class BaseBatchProcessor(ABC):
    """Base class for batch processing."""
    
    def __init__(self, model_config: dict):
        self.model_config = model_config
        self.model_name = model_config.get("model_name", "")

    @abstractmethod
    def submit(self, messages_list: List[List[Dict[str, Any]]], metas: List[Dict[str, Any]], jsonl_path: str) -> str:
        """Submit batch job and return batch_id."""
        pass

    @abstractmethod
    def retrieve(self, batch_id: str) -> List[Dict[str, Any]]:
        """Retrieve batch results."""
        pass
    
    def _save_jsonl(self, data: List[Dict], path: str):
        """Helper to save list of dicts to JSONL."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            for line in data:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")

class OpenAIBatchProcessor(BaseBatchProcessor):
    """Batch processor for OpenAI and compatible models (e.g. Gemini)."""
    
    def __init__(self, model_config: dict):
        super().__init__(model_config)
        self.cfg = OpenAIModelConfig(**model_config)
        self.interface = OpenAIModelInterface(self.cfg)
        self.client = self.interface.client

    def submit(self, messages_list, metas, jsonl_path) -> str:
        lines = []
        for i, msgs in enumerate(messages_list):
            mid = (metas[i] or {}).get("message_id", f"req_{i}")
            
            body = self.interface._prepare_api_payload(msgs)

            lines.append({
                "custom_id": str(mid),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            })
        self._save_jsonl(lines, jsonl_path)

        batch_input = self.client.files.create(file=open(jsonl_path, "rb"), purpose="batch")
        batch = self.client.batches.create(
            input_file_id=batch_input.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        return batch.id

    def retrieve(self, batch_id: str) -> List[Dict[str, Any]]:
        while True:
            b = self.client.batches.retrieve(batch_id)
            if b.status in ("failed", "canceled"):
                raise RuntimeError(f"Batch {batch_id} status={b.status} reason={b.errors}")
            if b.status in ("completed", "expired"):
                break
            time.sleep(30)

        if not b.output_file_id:
            return []

        content = self.client.files.content(b.output_file_id)
        text = getattr(content, "text", None) or getattr(content, "content", None)
        if hasattr(text, "decode"):
            text = text.decode("utf-8")
        if not isinstance(text, str):
            text = content.read().decode("utf-8")

        results = []
        for line in text.splitlines():
            if not line.strip(): continue
            obj = json.loads(line)
            custom_id = obj.get("custom_id")
            body = ((obj.get("response") or {}).get("body") or {})
            choices = body.get("choices") or []
            llm_text = choices[0].get("message", {}).get("content", "") if choices else ""
            usage = body.get("usage") or {}
            results.append({"message_id": custom_id, "text": llm_text, "usage": usage})
        return results

class ClaudeBatchProcessor(BaseBatchProcessor):
    """Batch processor for Claude models."""
    
    def __init__(self, model_config: dict):
        super().__init__(model_config)
        self.cfg = ClaudeModelConfig(**model_config)
        self.interface = ClaudeModelInterface(self.cfg)
        self.client = self.interface.client

    def submit(self, messages_list, metas, jsonl_path) -> str:
        requests_data = []
        for i, msgs in enumerate(messages_list):
            mid = (metas[i] or {}).get("message_id", f"req_{i}")
            
            params = self.interface._prepare_api_payload(msgs)
            
            requests_data.append({
                "custom_id": str(mid),
                "params": params
            })
        
        self._save_jsonl(requests_data, jsonl_path)
        batch = self.client.messages.batches.create(requests=requests_data)
        return batch.id

    def retrieve(self, batch_id: str) -> List[Dict[str, Any]]:
        while True:
            b = self.client.messages.batches.retrieve(batch_id)
            if b.processing_status == "ended":
                break
            time.sleep(10)
        
        if not b.results_url:
            return []
            
        headers = {"x-api-key": self.client.api_key, "anthropic-version": "2023-06-01"}
        resp = requests.get(b.results_url, headers=headers)
        resp.raise_for_status()
        
        results = []
        for line in resp.text.splitlines():
            if not line.strip(): continue
            obj = json.loads(line)
            custom_id = obj.get("custom_id")
            res = obj.get("result", {})
            if res.get("type") == "succeeded":
                msg = res.get("message", {})
                content = msg.get("content", [])
                text = "".join(block.get("text", "") for block in content if block.get("type") == "text")
                usage = msg.get("usage", {})
                results.append({"message_id": custom_id, "text": text, "usage": usage})
        return results

def get_batch_processor(model_config: dict) -> BaseBatchProcessor:
    """Factory to get appropriate batch processor."""
    if "claude" in model_config.get("model_name", "").lower():
        return ClaudeBatchProcessor(model_config)
    return OpenAIBatchProcessor(model_config)
