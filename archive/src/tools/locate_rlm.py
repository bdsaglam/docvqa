"""RLM-based locate tool — adaptive coarse-to-fine object localization.

Unlike the fixed-loop locate, this uses an RLM agent that can reason about
what it sees and adapt its search strategy: zoom in, try different regions,
adjust crop sizes, and decide when it's confident enough to return.

The RLM agent gets the original image as `image` and can crop/resize it
freely. It calls `look(image, query)` for VLM perception and reasons in
Python to refine coordinates.
"""

from __future__ import annotations

import logging
import os
import tempfile

import dspy
import logfire
import pydantic

from docvqa.rlm import RLM

logger = logging.getLogger(__name__)


class LocateResult(pydantic.BaseModel):
    found: bool
    description: str
    bbox: list[int] | None = None  # [left, top, right, bottom] in original image pixels
    center: list[int] | None = None  # [x, y]
    orientation: float | None = None  # degrees, None if upright


LOCATE_INSTRUCTIONS = (
    "You are a visual localization agent. Your task is to precisely locate an object in an image.\n\n"
    "## DATA\n"
    "- `image`: a PIL Image to search in. Use `image.size` to get (width, height).\n"
    "- `query`: what to find.\n\n"
    "## TOOLS\n"
    "- look(img, query) -> str: Send a PIL Image to the VLM with a query. Returns text response.\n"
    "- batch_look(requests) -> list[str]: Parallel VLM calls. Input: list of (image, query) tuples.\n\n"
    "## APPROACH\n"
    "1. Start with the full image: `look(image, query)` to get a rough idea of where the target is.\n"
    "2. Crop the region of interest: `image.crop((left, top, right, bottom))`.\n"
    "3. Ask the VLM about the cropped region to refine your estimate.\n"
    "4. Repeat until the target fills the frame well — not too much empty space, not cropped.\n"
    "5. Track coordinates in the ORIGINAL image frame at all times.\n\n"
    "## GUIDELINES\n"
    "- Ask the VLM simple questions: 'Is {query} visible?', 'Where in this image is it? top/bottom/left/right?'\n"
    "- Convert VLM's relative descriptions to pixel coordinates using the crop's offset and size.\n"
    "- If the VLM can't find the target, try neighboring regions or zoom out.\n"
    "- When confident, SUBMIT the result.\n\n"
    "## OUTPUT\n"
    "SUBMIT with found, description, bbox, center, orientation.\n"
    "bbox is [left, top, right, bottom] in pixels of the ORIGINAL image.\n"
    "center is [x, y] in pixels. orientation is degrees or None.\n"
)


def _build_locate_signature() -> dspy.Signature:
    fields: dict = {
        "query": (str, dspy.InputField(desc="What to find in the image")),
        "image_size": (str, dspy.InputField(desc="Image dimensions as 'WIDTHxHEIGHT'")),
        "result": (
            LocateResult,
            dspy.OutputField(
                desc="Localization result with bbox in original image pixels"
            ),
        ),
    }
    return dspy.Signature(fields, LOCATE_INSTRUCTIONS)


def _build_sandbox_code(image_path: str) -> str:
    return f'''
import os
import tempfile
from PIL import Image

image = Image.open({image_path!r})

def look(img, query):
    """Send a PIL Image to the VLM. Returns text response."""
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmp, format="PNG")
    tmp.close()
    return _look_impl(tmp.name, query)

def batch_look(requests):
    """Send multiple images to the VLM in parallel.
    Input: list of (image, query) tuples. Returns: list of str answers."""
    import json as _json
    paths = []
    for img, query in requests:
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img.save(tmp, format="PNG")
        tmp.close()
        paths.append({{"path": tmp.name, "query": query}})
    return _batch_look_impl(_json.dumps(paths))
'''


def locate(
    image,  # PIL Image
    query: str,
    vlm_predict: dspy.Predict,
    vlm_lm: dspy.LM,
    llm: dspy.LM | None = None,
    max_iterations: int = 5,
) -> LocateResult:
    """Locate an object using an RLM agent that adaptively zooms and reasons.

    Args:
        image: PIL Image to search in
        query: what to find
        vlm_predict: dspy.Predict for VLM calls
        vlm_lm: dspy.LM for VLM context
        llm: dspy.LM for reasoning (uses default if None)
        max_iterations: max RLM iterations
    """
    w, h = image.size

    # Save image to temp file for sandbox loading
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image.save(tmp, format="PNG")
    tmp.close()
    image_path = tmp.name

    def _look_impl(img_path: str, q: str) -> str:
        from PIL import Image as PILImage

        with logfire.span("look", query=q) as span:
            img = PILImage.open(img_path)
            with dspy.context(lm=vlm_lm):
                result = vlm_predict(image=dspy.Image(img), query=q)
                answer = result.answer or ""
                span.set_attribute("answer", answer[:500])
                return answer

    def _batch_look_impl(requests_json: str) -> list[str]:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import json as _json
        requests = _json.loads(requests_json)
        if not requests:
            return []
        results: list[str] = [""] * len(requests)

        def _do(idx: int, path: str, query: str) -> tuple[int, str]:
            return idx, _look_impl(path, query)

        with logfire.span("batch_look", num_requests=len(requests)):
            with ThreadPoolExecutor(max_workers=min(len(requests), 4)) as pool:
                futures = {
                    pool.submit(_do, i, r["path"], r["query"]): i
                    for i, r in enumerate(requests)
                }
                for future in as_completed(futures):
                    idx, answer = future.result()
                    results[idx] = answer
        return results

    with logfire.span("locate", query=query, image_size=f"{w}x{h}") as span:
        rlm = RLM(
            signature=_build_locate_signature(),
            max_iterations=max_iterations,
            max_llm_calls=max_iterations * 3,
            tools=[_look_impl, _batch_look_impl],
            verbose=True,
            sandbox_code=_build_sandbox_code(image_path),
        )

        if llm:
            with dspy.context(lm=llm):
                prediction = rlm(query=query, image_size=f"{w}x{h}")
        else:
            prediction = rlm(query=query, image_size=f"{w}x{h}")

        span.set_attribute("result", prediction.result.model_dump(mode="json"))

    # Clean up temp file
    try:
        os.unlink(image_path)
    except OSError:
        pass

    # DSPy returns the pydantic object directly
    result = prediction.result
    if isinstance(result, LocateResult):
        return result

    # Fallback: parse from string/dict
    if isinstance(result, str):
        try:
            return LocateResult.model_validate_json(result)
        except (pydantic.ValidationError, ValueError):
            pass
    if isinstance(result, dict):
        try:
            return LocateResult.model_validate(result)
        except pydantic.ValidationError:
            pass

    return LocateResult(
        found=False, description=f"Failed to parse locate result for '{query}'"
    )
