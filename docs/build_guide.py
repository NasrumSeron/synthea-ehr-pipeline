"""Build docs/index.html from GUIDE.md (GUIDE.md is the source of truth).

    pip install markdown          # build-time only; not needed to run the pipeline
    python docs/build_guide.py

The page is self-contained: no external scripts, fonts or trackers.
"""

import html
import re
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "GUIDE.md"
OUT = ROOT / "docs" / "index.html"
TPL = Path(__file__).resolve().parent / "template.html"

MAP_SVG = """<figure class="fig"><svg viewBox="0 0 760 292" role="img" aria-label="One command runs phases B to G; the toolkit adds H to J">
<defs><marker id="m-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" class="s-head"/></marker></defs>
<text x="16" y="24" class="s-title">One command runs B to G; the toolkit adds H to J</text>
<rect x="120" y="44" width="624" height="92" rx="10" class="s-zone"/>
<text x="134" y="63" class="s-zone-label">python -m pipeline</text>
<rect x="120" y="176" width="624" height="92" rx="10" class="s-zone"/>
<text x="134" y="195" class="s-zone-label">toolkit/</text>
<g class="s-lines">
<path d="M104 96H136" marker-end="url(#m-arrow)"/>
<path d="M226 96H236" marker-end="url(#m-arrow)"/><path d="M326 96H336" marker-end="url(#m-arrow)"/>
<path d="M426 96H436" marker-end="url(#m-arrow)"/><path d="M526 96H536" marker-end="url(#m-arrow)"/>
<path d="M626 96H636" marker-end="url(#m-arrow)"/>
<path d="M682 120V202" marker-end="url(#m-arrow)"/>
<path d="M324 228H337" marker-end="url(#m-arrow)"/><path d="M525 228H538" marker-end="url(#m-arrow)"/>
</g>
<g><rect x="16" y="72" width="88" height="48" rx="8" class="s-box"/><text x="60" y="91" class="s-k">A</text><text x="60" y="108" class="s-n">Generate</text></g>
<g><rect x="138" y="72" width="88" height="48" rx="8" class="s-box"/><text x="182" y="91" class="s-k">B</text><text x="182" y="108" class="s-n">Load</text></g>
<g><rect x="238" y="72" width="88" height="48" rx="8" class="s-box"/><text x="282" y="91" class="s-k">C</text><text x="282" y="108" class="s-n">Clean</text></g>
<g><rect x="338" y="72" width="88" height="48" rx="8" class="s-box"/><text x="382" y="91" class="s-k">D</text><text x="382" y="108" class="s-n">Features</text></g>
<g><rect x="438" y="72" width="88" height="48" rx="8" class="s-box"/><text x="482" y="91" class="s-k">E</text><text x="482" y="108" class="s-n">Label</text></g>
<g><rect x="538" y="72" width="88" height="48" rx="8" class="s-box"/><text x="582" y="91" class="s-k">F</text><text x="582" y="108" class="s-n">Quality</text></g>
<g><rect x="638" y="72" width="88" height="48" rx="8" class="s-accent"/><text x="682" y="91" class="s-k">G</text><text x="682" y="108" class="s-n">Model</text></g>
<g><rect x="138" y="204" width="186" height="48" rx="8" class="s-box"/><text x="231" y="223" class="s-n">H · Look</text><text x="231" y="240" class="s-sub">6 charts + HTML page</text></g>
<g><rect x="339" y="204" width="186" height="48" rx="8" class="s-box"/><text x="432" y="223" class="s-n">I · Robustness</text><text x="432" y="240" class="s-sub">many seeds, read the spread</text></g>
<g><rect x="540" y="204" width="186" height="48" rx="8" class="s-box"/><text x="633" y="223" class="s-n">J · Freeze &amp; deploy</text><text x="633" y="240" class="s-sub">score new data, recalibrate</text></g>
<text x="60" y="140" class="s-sub">Synthea, once</text>
</svg><figcaption>Phases A to J. Data generation runs once; experiments repeat B to J.</figcaption></figure>"""

LOOP_SVG = """<figure class="fig"><svg viewBox="0 0 760 300" role="img" aria-label="Only a change that beats the seed spread moves on to the one-time test">
<defs><marker id="l-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" class="s-head"/></marker></defs>
<text x="16" y="24" class="s-title">Only a change that beats the seed spread moves on to the one-time test</text>
<g class="s-lines">
<path d="M168 130H202" marker-end="url(#l-arrow)"/><path d="M358 130H392" marker-end="url(#l-arrow)"/>
<path d="M548 130H586" marker-end="url(#l-arrow)"/>
<path d="M650 90V64H280V106" marker-end="url(#l-arrow)"/>
<path d="M650 170V236" marker-end="url(#l-arrow)"/>
<path d="M565 260H525" marker-end="url(#l-arrow)"/><path d="M375 260H335" marker-end="url(#l-arrow)"/><path d="M185 260H165" marker-end="url(#l-arrow)"/>
<path d="M450 236V196H330V154" class="s-dash" marker-end="url(#l-arrow)"/>
</g>
<g><rect x="12" y="106" width="156" height="48" rx="8" class="s-box"/><text x="90" y="127" class="s-n">Reproduce</text><text x="90" y="143" class="s-sub">baseline AUROC 0.9144</text></g>
<g><rect x="202" y="106" width="156" height="48" rx="8" class="s-box"/><text x="280" y="127" class="s-n">Change one thing</text><text x="280" y="143" class="s-sub">on a branch, run pytest</text></g>
<g><rect x="392" y="106" width="156" height="48" rx="8" class="s-box"/><text x="470" y="127" class="s-n">Compare on dev</text><text x="470" y="143" class="s-sub">10 split seeds, mean ± SD</text></g>
<g><polygon points="586,130 650,90 714,130 650,170" class="s-box"/><text x="650" y="127" class="s-n2">Beats the</text><text x="650" y="143" class="s-n2">spread?</text></g>
<g><rect x="565" y="236" width="150" height="48" rx="8" class="s-accent"/><text x="640" y="257" class="s-n">Lock recipe</text><text x="640" y="273" class="s-sub s-ink">use test set once</text></g>
<g><rect x="375" y="236" width="150" height="48" rx="8" class="s-box"/><text x="450" y="257" class="s-n">Freeze model</text><text x="450" y="273" class="s-sub">score new data</text></g>
<g><rect x="185" y="236" width="150" height="48" rx="8" class="s-box"/><text x="260" y="257" class="s-n">Recalibrate</text><text x="260" y="273" class="s-sub">locally, if needed</text></g>
<g><rect x="15" y="236" width="150" height="48" rx="8" class="s-box"/><text x="90" y="257" class="s-n">Report</text><text x="90" y="273" class="s-sub">with known limits</text></g>
<text x="465" y="56" class="s-sub">no: revert, log, try the next idea</text>
<text x="660" y="206" class="s-sub s-left">yes</text>
<text x="342" y="184" class="s-sub s-left">new data shows a problem</text>
</svg><figcaption>The experiment loop. Most time is spent in the top row.</figcaption></figure>"""


def build():
    md_text = SRC.read_text(encoding="utf-8")
    # Title + intro come from the template; drop the H1 and the pointer to this very page.
    md_text = re.sub(r"\A# .*\n", "", md_text)
    md_text = re.sub(r"\nA formatted version of this guide.*\n", "\n", md_text)
    # Text-art map -> drawn map.
    md_text = re.sub(r"```\n A Generate.*?```", "<!--MAP-->", md_text, flags=re.S)
    conv = markdown.Markdown(extensions=["tables", "fenced_code", "toc", "sane_lists"],
                             extension_configs={"toc": {"permalink": False}})
    body = conv.convert(md_text)
    # The hero already carries the intro sentence; drop the first paragraph. Hide header rows with no words.
    body = re.sub(r"\A\s*<p>How to run.*?</p>\s*", "", body, flags=re.S)
    body = re.sub(r"<thead>\s*<tr>\s*(?:<th></th>\s*)+</tr>\s*</thead>", "", body)
    body = body.replace("<p><!--MAP--></p>", MAP_SVG).replace("<!--MAP-->", MAP_SVG)
    # Experiment-loop drawing before the numbered steps of section 2.
    body = re.sub(r"(<ol>\s*<li><strong>Branch:)", lambda m: LOOP_SVG + "\n" + m.group(1), body, count=1)
    # Task lists, callouts, scrollable tables, senior notes.
    body = re.sub(r"<li>\[ \] ", '<li class="task"><span class="box" aria-hidden="true"></span>', body)
    body = body.replace("<blockquote>", '<blockquote class="callout">')
    body = body.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")
    body = body.replace("<p><strong>Senior notes", '<p class="senior"><strong>Senior notes')
    toc = "\n".join(f'<li><a href="#{t["id"]}">{html.escape(t["name"])}</a></li>' for t in conv.toc_tokens)
    page = TPL.read_text(encoding="utf-8").replace("{{TOC}}", toc).replace("{{BODY}}", body)
    OUT.write_text(page, encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(page):,} bytes)")


if __name__ == "__main__":
    build()
