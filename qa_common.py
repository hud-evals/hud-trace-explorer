"""Shared setup for QA workflow scenarios."""

import json as _json
import re as _re
from typing import Any, Literal, get_args, get_origin

from pydantic import TypeAdapter
from pydantic_core import PydanticUndefined

from env import logger

# ---------------------------------------------------------------------------
# Robust answer parsing — handles SDK data-loss, markdown fences, prose, etc.
# ---------------------------------------------------------------------------


def _extract_json_object(text: str) -> str | None:
    """Extract a JSON object from text that may contain markdown fences or prose."""

    def _brace_depth_scan(s: str) -> str | None:
        depth = 0
        start = None
        last_obj = None
        in_string = False
        escape = False
        for i, ch in enumerate(s):
            if escape:
                escape = False
                continue
            if ch == "\\":
                if in_string:
                    escape = True
                continue
            if ch == '"':
                if depth > 0:
                    in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    last_obj = s[start : i + 1]
        return last_obj

    fenced = _re.findall(r"```(?:json)?\s*(.*?)\s*```", text, _re.DOTALL)
    for block in reversed(fenced):
        obj = _brace_depth_scan(block)
        if obj is not None:
            try:
                _json.loads(obj)
                return obj
            except (ValueError, _json.JSONDecodeError):
                pass

    return _brace_depth_scan(text)


def _regex_extract_result(text: str, model_cls: type) -> Any | None:
    """Last-resort: extract key fields from plain prose via regex."""
    if not isinstance(text, str) or not text.strip():
        return None

    fields = model_cls.model_fields
    extracted: dict[str, Any] = {}

    for name, field_info in fields.items():
        annotation = field_info.annotation
        origin = get_origin(annotation)

        if annotation is bool or (hasattr(annotation, "__args__") and bool in getattr(annotation, "__args__", ())):
            pattern = _re.compile(
                rf"""(?:"|')?{_re.escape(name)}(?:"|')?\s*[:=]\s*(?:"|')?(true|false)(?:"|')?""",
                _re.IGNORECASE,
            )
            m = pattern.search(text)
            if m:
                extracted[name] = m.group(1).lower() == "true"

        elif origin is Literal:
            allowed = get_args(annotation)
            alternatives = "|".join(_re.escape(str(v)) for v in allowed)
            pattern = _re.compile(
                rf"""(?:"|')?{_re.escape(name)}(?:"|')?\s*[:=]\s*(?:"|')?({alternatives})(?:"|')?""",
                _re.IGNORECASE,
            )
            m = pattern.search(text)
            if m:
                matched = m.group(1).lower()
                for v in allowed:
                    if str(v).lower() == matched:
                        extracted[name] = v
                        break

        elif annotation is float:
            pattern = _re.compile(
                rf"""(?:"|')?{_re.escape(name)}(?:"|')?\s*[:=]\s*(?:"|')?([\d.]+)""",
                _re.IGNORECASE,
            )
            m = pattern.search(text)
            if m:
                try:
                    extracted[name] = float(m.group(1))
                except ValueError:
                    pass

        elif annotation is str:
            pattern = _re.compile(
                rf"""(?:"|')?{_re.escape(name)}(?:"|')?\s*[:=]\s*(?:"|')([^"'\n]+?)(?:"|'|,|\n|$)""",
                _re.IGNORECASE,
            )
            m = pattern.search(text)
            if m:
                extracted[name] = m.group(1).strip()

    if not extracted:
        return None

    for name, field_info in fields.items():
        if name in extracted:
            continue
        has_default = field_info.default is not PydanticUndefined
        has_default_factory = field_info.default_factory is not None
        if has_default or has_default_factory:
            continue
        if field_info.annotation is str:
            extracted[name] = "(extracted via fallback regex)"

    try:
        return model_cls.model_validate(extracted)
    except Exception:
        return None


def parse_qa_result(answer: Any, model_cls: type) -> Any | None:
    """Parse the agent's answer into a Pydantic model.

    Handles every plausible shape the SDK might deliver:
    - model instance, dict, JSON string, AgentAnswer wrapper,
      prose with embedded JSON, markdown-fenced JSON.
    Falls back to regex extraction from plain text as last resort.
    """
    adapter = TypeAdapter(model_cls)

    def _try_parse(raw: Any) -> Any | None:
        if isinstance(raw, model_cls):
            return raw
        if isinstance(raw, dict):
            try:
                return model_cls.model_validate(raw)
            except Exception:
                pass
            text = raw.get("content") or raw.get("text") or ""
            if isinstance(text, str) and text.strip():
                try:
                    return adapter.validate_json(text)
                except Exception:
                    pass
        if isinstance(raw, str) and raw.strip():
            try:
                return adapter.validate_json(raw)
            except Exception:
                pass
            try:
                obj = _json.loads(raw)
                if isinstance(obj, dict):
                    return model_cls.model_validate(obj)
            except Exception:
                pass
            extracted = _extract_json_object(raw)
            if extracted is not None:
                try:
                    return adapter.validate_json(extracted)
                except Exception:
                    pass
            regex_result = _regex_extract_result(raw, model_cls)
            if regex_result is not None:
                logger.warning("Used last-resort regex extraction for %s", model_cls.__name__)
                return regex_result
        return None

    for attr in ("content", "raw"):
        val = getattr(answer, attr, None)
        if val is not None:
            r = _try_parse(val)
            if r is not None:
                return r

    return _try_parse(answer)


def normalize_optional_bool(v: Any) -> bool | None:
    """MCP / JSON sometimes leaves booleans as strings."""
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1", "yes"):
            return True
        if s in ("false", "0", "no", ""):
            return False
    return v if isinstance(v, bool) else None  # type: ignore[return-value]
