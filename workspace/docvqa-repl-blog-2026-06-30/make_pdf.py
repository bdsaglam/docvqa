"""Render draft.md -> blog.pdf (weasyprint). Run via:
   uv run --with markdown --with weasyprint python make_pdf.py
Relative figure paths (figures/*.png) resolve against this directory.
"""
import os
import markdown
from weasyprint import HTML

HERE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(HERE, "draft.md")) as f:
    md = f.read()

# Pull the H1 title out so we can render it as a styled header.
lines = md.splitlines()
title = "Perceive-Reason-Code"
if lines and lines[0].startswith("# "):
    title = lines[0][2:].strip()
    md = "\n".join(lines[1:])

html_body = markdown.markdown(
    md,
    extensions=["tables", "fenced_code", "footnotes", "attr_list", "sane_lists"],
)

CSS = """
@page { size: A4; margin: 2cm 2.2cm; @bottom-center {
    content: counter(page); color:#999; font-size:9pt; } }
html { font-size: 11pt; }
body { font-family: 'Georgia','DejaVu Serif',serif; line-height: 1.5;
    color:#1d1d1f; max-width: 100%; }
h1.title { font-family:'Helvetica','DejaVu Sans',sans-serif; font-size:22pt;
    line-height:1.2; margin:0 0 0.2em 0; color:#111; }
.byline { font-family:'Helvetica','DejaVu Sans',sans-serif; color:#666;
    font-size:10pt; margin-bottom:1.4em; border-bottom:1px solid #e3e3e3;
    padding-bottom:1em; }
h2 { font-family:'Helvetica','DejaVu Sans',sans-serif; font-size:15pt;
    margin-top:1.6em; color:#111; border-bottom:1px solid #ececec;
    padding-bottom:0.15em; }
h3 { font-family:'Helvetica','DejaVu Sans',sans-serif; font-size:12.5pt;
    margin-top:1.2em; color:#222; }
p { margin: 0.55em 0; }
a { color:#2f6f9f; text-decoration:none; word-break:break-all; }
img { max-width: 100%; display:block; margin: 1.0em auto 0.3em auto; }
blockquote { background:#f6f8fa; border-left:3px solid #2f6f9f;
    margin:1.2em 0; padding:0.6em 1em; font-size:10.5pt; }
blockquote p { margin:0.35em 0; }
table { border-collapse: collapse; margin: 1em auto; font-size:10pt;
    font-family:'Helvetica','DejaVu Sans',sans-serif; }
th,td { border:1px solid #d0d0d0; padding:5px 10px; text-align:left; }
th { background:#f0f3f6; }
pre { background:#f6f8fa; border:1px solid #e6e6e6; border-radius:4px;
    padding:0.7em 0.9em; font-size:9pt; overflow-x:auto; line-height:1.35; }
code { font-family:'DejaVu Sans Mono',monospace; font-size:9pt; }
p code,li code { background:#f0f1f3; padding:0.5px 3px; border-radius:3px; }
.footnote { font-size:9pt; color:#444; border-top:1px solid #e3e3e3;
    margin-top:2em; padding-top:0.5em; }
.footnote ol { padding-left:1.1em; }
hr { border:none; border-top:1px solid #e3e3e3; }
"""

doc = f"""<!doctype html><html><head><meta charset="utf-8">
<style>{CSS}</style></head><body>
<h1 class="title">{title}</h1>
<div class="byline">Barış Deniz Sağlam</div>
{html_body}
</body></html>"""

out = os.path.join(HERE, "blog.pdf")
HTML(string=doc, base_url=HERE).write_pdf(out)
print("wrote", out)
