#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from titans.stage1_models import Stage1ModelConfig
from titans.stage1_runtime import Stage1DeploymentRuntime


LOGGER = logging.getLogger("stage1_service")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve stage1 memory/chat APIs.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--backbone-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--attn-implementation")
    parser.add_argument("--memory-slots", type=int, default=16)
    parser.add_argument("--memory-hidden-mult", type=float, default=2.0)
    parser.add_argument("--memory-dropout", type=float, default=0.0)
    parser.add_argument("--history-backbone-mode", default="full")
    parser.add_argument("--memory-update-source", default="last_hidden")
    parser.add_argument("--num-retrieved-memory-tokens", type=int, default=16)
    parser.add_argument("--loss-mask-scope", default="answer_only")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def build_runtime(args: argparse.Namespace) -> Stage1DeploymentRuntime:
    config = Stage1ModelConfig(
        backbone_name=args.backbone_name,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        history_backbone_mode=args.history_backbone_mode,
        memory_update_source=args.memory_update_source,
        num_retrieved_memory_tokens=args.num_retrieved_memory_tokens,
        loss_mask_scope=args.loss_mask_scope,
        memory_slots=args.memory_slots,
        memory_hidden_mult=args.memory_hidden_mult,
        memory_dropout=args.memory_dropout,
        trust_remote_code=args.trust_remote_code,
    )
    return Stage1DeploymentRuntime.from_model_config(
        config=config,
        checkpoint_path=args.checkpoint_path,
    )


class Stage1RequestHandler(BaseHTTPRequestHandler):
    runtime: Stage1DeploymentRuntime

    def _log_profile(self, kind: str, session_id: str, profile: dict[str, Any]) -> None:
        if not profile:
            return
        parts = [f"{key}={value:.4f}s" if isinstance(value, float) else f"{key}={value}" for key, value in profile.items()]
        LOGGER.info("%s session=%s %s", kind, session_id, " ".join(parts))

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/v1/memory/write":
            self._handle_write_memory()
            return
        if self.path == "/v1/chat/respond":
            self._handle_chat_respond()
            return
        self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_GET(self) -> None:  # noqa: N802
        session_id = self._parse_session_path()
        if session_id is None:
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        metadata = self.runtime.get_session_metadata(session_id)
        if metadata is None:
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "session_not_found"})
            return
        self._write_json(HTTPStatus.OK, metadata)

    def do_DELETE(self) -> None:  # noqa: N802
        session_id = self._parse_session_path()
        if session_id is None:
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        existed = self.runtime.delete_session(session_id)
        self._write_json(
            HTTPStatus.OK,
            {"session_id": session_id, "deleted": existed},
        )

    def log_message(self, format: str, *args: Any) -> None:
        LOGGER.info("%s - %s", self.address_string(), format % args)

    def _handle_write_memory(self) -> None:
        payload = self._read_json_body()
        session_id = self._require_string(payload, "session_id")
        content = payload.get("content")
        contents = payload.get("contents")

        if content is not None and not isinstance(content, str):
            raise ValueError("content must be a string")
        if contents is not None:
            if not isinstance(contents, list) or not all(isinstance(item, str) for item in contents):
                raise ValueError("contents must be a list of strings")

        result = self.runtime.write_memory(
            session_id=session_id,
            content=content,
            contents=contents,
            idempotency_key=payload.get("idempotency_key"),
        )
        self._log_profile("write_memory", session_id, result.profile)
        self._write_json(HTTPStatus.OK, asdict(result))

    def _handle_chat_respond(self) -> None:
        payload = self._read_json_body()
        session_id = self._require_string(payload, "session_id")
        query = self._require_string(payload, "query")
        generation_config = self._extract_generation_config(payload)
        include_debug = bool(payload.get("include_debug", False))
        mode = payload.get("mode", "memory")
        if mode not in {"memory", "direct_backbone"}:
            raise ValueError("mode must be one of: memory, direct_backbone")

        if mode == "memory":
            result = self.runtime.chat_with_memory(
                session_id=session_id,
                query=query,
                generation_config=generation_config,
                include_debug=include_debug,
            )
        else:
            result = self.runtime.chat_direct_backbone(
                session_id=session_id,
                query=query,
                generation_config=generation_config,
            )
        response = {
            "session_id": result.session_id,
            "answer": result.answer,
            "memory_version": result.memory_version,
        }
        if result.profile:
            response["profile"] = result.profile
        if include_debug and result.retrieval_weights is not None:
            response["retrieval_weights"] = result.retrieval_weights.tolist()
        self._log_profile("chat_respond", session_id, result.profile)
        self._write_json(HTTPStatus.OK, response)

    def _extract_generation_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        allowed_keys = {"max_new_tokens", "temperature", "do_sample", "top_p", "top_k"}
        config = payload.get("generation_config")
        if config is not None:
            if not isinstance(config, dict):
                raise ValueError("generation_config must be an object")
            unknown = set(config) - allowed_keys
            if unknown:
                raise ValueError(f"unsupported generation_config keys: {sorted(unknown)}")
            return dict(config)

        inline_config = {key: payload[key] for key in allowed_keys if key in payload}
        return inline_config

    def _parse_session_path(self) -> str | None:
        parsed = urlparse(self.path)
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) == 3 and parts[0] == "v1" and parts[1] == "sessions":
            return parts[2]
        return None

    def _read_json_body(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            raise ValueError("request body is required")
        raw_body = self.rfile.read(content_length)
        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise ValueError("request body must be valid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def _require_string(self, payload: dict[str, Any], key: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{key} must be a non-empty string")
        return value

    def _write_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def handle_one_request(self) -> None:
        try:
            super().handle_one_request()
        except ValueError as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover
            LOGGER.exception("Unhandled request error")
            self._write_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    runtime = build_runtime(args)
    LOGGER.info("Runtime device=%s dtype=%s", runtime.model.backbone.device, next(runtime.model.parameters()).dtype)
    handler = type("ConfiguredStage1RequestHandler", (Stage1RequestHandler,), {"runtime": runtime})
    server = ThreadingHTTPServer((args.host, args.port), handler)
    LOGGER.info("Serving stage1 on http://%s:%s", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Shutting down stage1 service")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
