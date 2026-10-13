"""Generate browser/index.html: static feature browser (no build step).
Embeds top-100 per layer + layer summaries. Vanilla JS, filter by layer,
search examples. ~200KB single file.
"""
import json
from pathlib import Path

HTML = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>SAE-Bilingual feature browser (llm-jp-3-1.8B)</title>
<style>body{font-family:system-ui,sans-serif;max-width:900px;margin:2em auto;padding:0 1em}
.feat{border:1px solid #ccc;border-radius:8px;padding:0.7em;margin:0.6em 0}
.ex{font-size:0.85em;color:#444;margin:0.15em 0}.r{font-weight:700}
button{margin-right:0.4em}#q{width:60%}</style></head><body>
<h1>Shared JA-EN features (TopK SAE, llm-jp-3-1.8B)</h1>
<p id="sum"></p>
<div><button data-l="4">Layer 4</button><button data-l="12">Layer 12</button>
<button data-l="20">Layer 20</button>
<input id="q" placeholder="filter examples…"></div>
<div id="list"></div>
<script>const DATA=__DATA__;
let layer="12", q="";
document.getElementById("sum").textContent =
 "L4 share(r>.5) 0.25 · L12 0.29 · L20 0.13 (20k Tatoeba pairs, sentence-max Pearson). r = paired correlation; examples = top-activating sentences.";
function render(){const d=DATA[layer].filter(f=>!q||f.examples.some(e=>e.text.includes(q)));
 document.getElementById("list").innerHTML=d.map(f=>
 `<div class="feat"><span class="r">f${f.feature} r=${f.r}</span> jā=${f.ja_mean} en̄=${f.en_mean}`+
 f.examples.map(e=>`<div class="ex">[${e.lang}] ${e.text} (${e.act})</div>`).join("")+`</div>`).join("");}
document.querySelectorAll("button").forEach(b=>b.onclick=()=>{layer=b.dataset.l;render();});
document.getElementById("q").oninput=e=>{q=e.target.value;render();};
render();</script></body></html>"""


def main() -> None:
    data = {}
    for layer in [4, 12, 20]:
        d = json.load(open(f"results/sharing_l{layer}.json", encoding="utf-8"))
        data[str(layer)] = d["top100"]
    Path("browser").mkdir(exist_ok=True)
    Path("browser/index.html").write_text(
        HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False)))
    print("browser written, bytes:", Path("browser/index.html").stat().st_size)


if __name__ == "__main__":
    main()
