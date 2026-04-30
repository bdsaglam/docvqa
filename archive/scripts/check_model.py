"""Check if a model is accessible and supports image inputs via litellm.

Usage:
    uv run python scripts/check_model.py vertex_ai/gemini-3-flash-preview
    uv run python scripts/check_model.py openai/Qwen/Qwen3.5-27B --api-base http://localhost:8927/v1
"""

from __future__ import annotations

import argparse
import sys

import litellm

def _make_test_image_b64() -> str:
    """Create a small 64x64 red square PNG as base64."""
    from io import BytesIO
    from PIL import Image
    img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    buf = BytesIO()
    img.save(buf, format="PNG")
    import base64
    return base64.b64encode(buf.getvalue()).decode()


def check_text(model: str, **kwargs) -> bool:
    try:
        resp = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "Say 'ok'"}],
            max_tokens=5,
            **kwargs,
        )
        text = resp.choices[0].message.content or ""
        print(f"  text: ok ({text.strip()!r})")
        return True
    except Exception as e:
        print(f"  text: FAILED ({e})")
        return False


def _no_thinking_kwargs(kwargs: dict) -> dict:
    """Return kwargs with thinking disabled via extra_body (for vLLM/Qwen)."""
    kw = dict(kwargs)
    extra = dict(kw.pop("extra_body", None) or {})
    extra["chat_template_kwargs"] = {"enable_thinking": False}
    kw["extra_body"] = extra
    return kw


def check_structured_text(model: str, thinking: bool = True, **kwargs) -> bool:
    label = "structured (text)" if thinking else "structured (text, no-think)"
    call_kwargs = kwargs if thinking else _no_thinking_kwargs(kwargs)
    try:
        resp = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "Give me a color and a number."}],
            max_tokens=16384,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "color_number",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "color": {"type": "string"},
                            "number": {"type": "integer"},
                        },
                        "required": ["color", "number"],
                        "additionalProperties": False,
                    },
                },
            },
            **call_kwargs,
        )
        text = resp.choices[0].message.content or ""
        import json
        parsed = json.loads(text)
        print(f"  {label}: ok ({parsed})")
        return True
    except Exception as e:
        print(f"  {label}: FAILED ({e})")
        return False


def check_structured_vision(model: str, thinking: bool = True, **kwargs) -> bool:
    label = "structured (vision)" if thinking else "structured (vision, no-think)"
    call_kwargs = kwargs if thinking else _no_thinking_kwargs(kwargs)
    try:
        resp = litellm.completion(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What color is this image? Return as JSON."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{_make_test_image_b64()}"},
                        },
                    ],
                }
            ],
            max_tokens=16384,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "image_color",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "color": {"type": "string"},
                        },
                        "required": ["color"],
                        "additionalProperties": False,
                    },
                },
            },
            **call_kwargs,
        )
        text = resp.choices[0].message.content or ""
        import json
        parsed = json.loads(text)
        print(f"  {label}: ok ({parsed})")
        return True
    except Exception as e:
        print(f"  {label}: FAILED ({e})")
        return False


def check_vision(model: str, **kwargs) -> bool:
    try:
        resp = litellm.completion(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What color is this image? One word."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{_make_test_image_b64()}"},
                        },
                    ],
                }
            ],
            max_tokens=10,
            **kwargs,
        )
        text = resp.choices[0].message.content or ""
        print(f"  vision: ok ({text.strip()!r})")
        return True
    except Exception as e:
        print(f"  vision: FAILED ({e})")
        return False


def main():
    parser = argparse.ArgumentParser(description="Check model accessibility and vision support")
    parser.add_argument("model", help="litellm model string (e.g. vertex_ai/gemini-3-flash-preview)")
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    kwargs = {}
    if args.api_base:
        kwargs["api_base"] = args.api_base
    if args.api_key:
        kwargs["api_key"] = args.api_key

    print(f"Checking {args.model}" + (f" at {args.api_base}" if args.api_base else ""))

    text_ok = check_text(args.model, **kwargs)
    vision_ok = check_vision(args.model, **kwargs) if text_ok else False
    struct_text_ok = check_structured_text(args.model, thinking=True, **kwargs) if text_ok else False
    struct_vision_ok = check_structured_vision(args.model, thinking=True, **kwargs) if vision_ok else False
    struct_text_nothink = check_structured_text(args.model, thinking=False, **kwargs) if text_ok else False
    struct_vision_nothink = check_structured_vision(args.model, thinking=False, **kwargs) if vision_ok else False

    yn = lambda v: "yes" if v else "no"
    print()
    print(f"Result:")
    print(f"  text:                        {yn(text_ok)}")
    print(f"  vision:                      {yn(vision_ok)}")
    print(f"  structured (text):           {yn(struct_text_ok)}")
    print(f"  structured (vision):         {yn(struct_vision_ok)}")
    print(f"  structured (text, no-think): {yn(struct_text_nothink)}")
    print(f"  structured (vision, no-think): {yn(struct_vision_nothink)}")
    sys.exit(0 if text_ok else 1)


if __name__ == "__main__":
    main()
