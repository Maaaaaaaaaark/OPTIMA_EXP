#!/usr/bin/env python3
"""OpenAI-compatible Transformers server for one independent OPTIMA agent.

This is the T4 fallback for Gemma 2, whose attention-logit soft capping is
unsupported by the attention backends available in vLLM 0.6.3 on Turing.
Each process owns exactly one tokenizer and one model instance.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)


def _json_bytes(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def render_messages(tokenizer, messages: List[Dict[str, str]],
                    continue_final_message: bool = False):
    """Render messages, including the vLLM-compatible assistant prefill."""
    kwargs = {"tokenize": False}
    if continue_final_message:
        kwargs["continue_final_message"] = True
    else:
        kwargs["add_generation_prompt"] = True
    text = tokenizer.apply_chat_template(messages, **kwargs)
    return tokenizer(text, return_tensors="pt", add_special_tokens=False)


def trim_at_forbidden(text: str, forbidden_texts: List[str]):
    """Remove simulated partner speech from one agent's generated turn."""
    positions = [text.find(marker) for marker in forbidden_texts]
    positions = [position for position in positions if position >= 0]
    if not positions:
        return text, False
    return text[:min(positions)].rstrip(), True


class StopOnForbiddenText(StoppingCriteria):
    """Stop as soon as an Agent begins emitting its partner's name prefix."""

    def __init__(self, tokenizer, prompt_tokens: int,
                 forbidden_texts: List[str]) -> None:
        self.tokenizer = tokenizer
        self.prompt_tokens = prompt_tokens
        self.forbidden_texts = forbidden_texts

    def __call__(self, input_ids, scores, **kwargs):
        generated = self.tokenizer.decode(
            input_ids[0, self.prompt_tokens:], skip_special_tokens=True
        )
        return any(marker in generated for marker in self.forbidden_texts)


class ModelRuntime:
    def __init__(self, model_path: str, served_model_name: str, device: str,
                 dtype: str, forbidden_texts: List[str]) -> None:
        self.model_path = model_path
        self.served_model_name = served_model_name
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise RuntimeError("This server currently requires a CUDA device")
        dtype_map = {
            "half": torch.float16,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        if dtype not in dtype_map:
            raise ValueError(f"unsupported dtype {dtype!r}; use half or bfloat16")
        self.dtype = dtype_map[dtype]
        self.forbidden_texts = forbidden_texts
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Eager Transformers attention retains Gemma 2's native tanh
        # soft-capping. Never silently disable or remove that model behavior.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        ).to(self.device)
        self.model.eval()
        self.lock = threading.Lock()

    @torch.inference_mode()
    def complete(self, request: Dict[str, Any]) -> Dict[str, Any]:
        requested_model = request.get("model")
        if requested_model != self.served_model_name:
            raise ValueError(
                f"model {requested_model!r} is not served here; expected "
                f"{self.served_model_name!r}"
            )
        messages = request.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")
        max_new_tokens = int(request.get("max_tokens", 256))
        if max_new_tokens < 1:
            raise ValueError("max_tokens must be positive")
        temperature = float(request.get("temperature", 1.0))
        continue_final = bool(request.get("continue_final_message", False))
        encoded = render_messages(self.tokenizer, messages, continue_final)
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        prompt_tokens = int(encoded["input_ids"].shape[-1])

        generation: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.forbidden_texts:
            generation["stopping_criteria"] = StoppingCriteriaList([
                StopOnForbiddenText(
                    self.tokenizer, prompt_tokens, self.forbidden_texts
                )
            ])
        if temperature > 0:
            generation.update(do_sample=True, temperature=temperature)
            if "top_p" in request:
                generation["top_p"] = float(request["top_p"])
        else:
            generation["do_sample"] = False
        # Requests within one Agent are serialized. Alice and Bob still run
        # concurrently because they are separate OS processes/model instances.
        with self.lock:
            seed = request.get("seed")
            if seed is not None:
                torch.manual_seed(int(seed))
                torch.cuda.manual_seed_all(int(seed))
            output = self.model.generate(**encoded, **generation)
        new_ids = output[0, prompt_tokens:]
        completion_tokens = int(new_ids.numel())
        content = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        content, boundary_stop = trim_at_forbidden(
            content, self.forbidden_texts
        )
        if boundary_stop:
            completion_tokens = len(self.tokenizer.encode(
                content, add_special_tokens=False
            ))
        finish_reason = (
            "length"
            if completion_tokens >= max_new_tokens and not boundary_stop
            else "stop"
        )
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.served_model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }


def make_handler(runtime: ModelRuntime):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: Dict[str, Any]) -> None:
            body = _json_bytes(payload)
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/v1/models":
                self._send(200, {"object": "list", "data": [{
                    "id": runtime.served_model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "transformers",
                    "root": runtime.model_path,
                }]})
            else:
                self._send(404, {"error": {"message": "not found"}})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/chat/completions":
                self._send(404, {"error": {"message": "not found"}})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length))
                self._send(200, runtime.complete(request))
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self._send(400, {"error": {"message": str(exc)}})
            except Exception as exc:
                self._send(500, {"error": {"message": repr(exc)}})

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[{runtime.served_model_name}] {self.address_string()} " + fmt % args,
                  flush=True)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="half")
    parser.add_argument("--forbidden-text", action="append", default=[])
    args = parser.parse_args()
    runtime = ModelRuntime(
        args.model,
        args.served_model_name,
        args.device,
        args.dtype,
        args.forbidden_text,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(runtime))
    print(
        f"[ready] {args.served_model_name}: {args.model} on {args.device} -> "
        f"http://{args.host}:{args.port}/v1/chat/completions",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
