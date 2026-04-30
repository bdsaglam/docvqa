import marimo

__generated_with = "0.13.0"
app = marimo.App(width="full")


@app.cell
def imports():
    import marimo as mo
    import sys
    sys.path.insert(0, "/home/baris/repos/docvqa/src")
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = 500_000_000
    from docvqa.solvers.flat_batch_solver import _ocr_via_docling
    return mo, Image, _ocr_via_docling


@app.cell
def load_pages(mo, Image):
    """Load sample pages from OCR-heavy documents."""
    import os
    from datasets import load_dataset

    ds = load_dataset("VLR-CVC/DocVQA-2026", split="val")
    doc_map = {s["doc_id"]: s for s in ds}

    # Pick table/figure-heavy docs
    test_docs = ["business_report_3", "business_report_1", "infographics_1", "science_paper_2"]
    available = [d for d in test_docs if d in doc_map]

    doc_selector = mo.ui.dropdown(available, value=available[0], label="Document")
    doc_selector
    return doc_map, doc_selector


@app.cell
def select_page(mo, doc_map, doc_selector):
    sample = doc_map[doc_selector.value]
    pages = sample["document"]
    num_pages = len(pages)

    page_selector = mo.ui.slider(0, num_pages - 1, value=0, label=f"Page (0-{num_pages-1})")
    page_selector
    return pages, page_selector


@app.cell
def show_page(mo, pages, page_selector, Image):
    page_img = pages[page_selector.value]
    if not isinstance(page_img, Image.Image):
        page_img = Image.open(page_img)
    w, h = page_img.size
    mo.md(f"**Page {page_selector.value}** — {w}x{h} pixels")
    return page_img, w, h


@app.cell
def crop_controls(mo, w, h):
    left = mo.ui.number(value=0, start=0, stop=w, label="Left")
    top = mo.ui.number(value=0, start=0, stop=h, label="Top")
    right = mo.ui.number(value=w, start=0, stop=w, label="Right")
    bottom = mo.ui.number(value=h, start=0, stop=h, label="Bottom")
    mo.hstack([left, top, right, bottom], justify="start", gap=1)
    return left, top, right, bottom


@app.cell
def run_ocr_button(mo):
    run_btn = mo.ui.run_button(label="Run OCR on crop")
    run_btn
    return (run_btn,)


@app.cell
def do_ocr(mo, page_img, left, top, right, bottom, run_btn, _ocr_via_docling):
    mo.stop(not run_btn.value, mo.md("*Click 'Run OCR on crop' to extract text*"))

    _l, _t, _r, _b = left.value, top.value, right.value, bottom.value
    crop = page_img.crop((_l, _t, _r, _b))

    # Save crop to temp file for OCR
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    crop.save(tmp, format="PNG")
    tmp.close()

    ocr_result = _ocr_via_docling(tmp.name)

    mo.hstack(
        [
            mo.vstack([
                mo.md(f"**Crop** ({_r-_l}x{_b-_t})"),
                mo.image(crop, width=500),
            ]),
            mo.vstack([
                mo.md("**OCR Result (Markdown)**"),
                mo.md(f"```\n{ocr_result}\n```"),
                mo.md("---"),
                mo.md("**Rendered:**"),
                mo.md(ocr_result),
            ]),
        ],
        widths=[1, 2],
        gap=2,
    )
    return (ocr_result,)


@app.cell
def presets(mo, w, h):
    mo.md(f"""
    ### Quick crop presets
    - **Full page**: 0, 0, {w}, {h}
    - **Top half**: 0, 0, {w}, {h//2}
    - **Bottom half**: 0, {h//2}, {w}, {h}
    - **Top-left quadrant**: 0, 0, {w//2}, {h//2}
    - **Top-right quadrant**: {w//2}, 0, {w}, {h//2}
    """)
    return


if __name__ == "__main__":
    app.run()
