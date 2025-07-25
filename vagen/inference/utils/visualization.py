import os
import re
from html import escape
from itertools import zip_longest
from pathlib import Path
from PIL import Image
import io, base64
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL|re.IGNORECASE)
ANS_RE   = re.compile(r"<answer>(.*?)</answer>", re.DOTALL|re.IGNORECASE)

def parse_conversation(output_str: str):
    """
    Splits an output_str into turns. Each turn is a dict:
      {'user': str,
       'think': str,
       'answer': str,
       'n_images': int}
    """
    turns = []
    # split on lines that start with "User:"
    chunks = re.split(r"(?m)^User:", output_str)
    # first chunk is everything before the first User: (system). skip it.
    for chunk in chunks[1:]:
        # chunk now starts with the user content until next "User:" or end
        text = chunk.lstrip()  # strip leading whitespace
        # separate out assistant block
        m = re.search(r"(?m)^Assistant:", text)
        if m:
            user_block = text[:m.start()].strip()
            assist_block = text[m.start():]
        else:
            user_block = text.strip()
            assist_block = ""
        # count how many <image> placeholders
        img_count = user_block.count("<image>")
        # remove them from display text
        user_text = user_block.replace("<image>", "").strip()

        # extract think & answer from assist_block
        think_m = THINK_RE.search(assist_block)
        ans_m   = ANS_RE.search(assist_block)
        think   = think_m.group(1).strip() if think_m else ""
        answer  = ans_m.group(1).strip() if ans_m  else ""

        turns.append({
            "user":     user_text,
            "think":    think,
            "answer":   answer,
            "n_images": img_count
        })
    return turns


def visualize_html(entries, output_html):
    """
    Generate an interactive HTML dashboard for a list of result dicts.

    Each entry should have:
      - 'env_id': str
      - 'config_id': str
      - 'output_str': str with User/Assistant blocks
      - 'image_data': list of PIL Image objects
      - 'metrics': dict of final metrics

    output_html: path to output HTML file.
    save_img_dir: optional directory to save extracted images.
    """
    out_dir = os.path.dirname(output_html)
    os.makedirs(out_dir, exist_ok=True)
    base = Path(output_html).stem

    total = len(entries)

    # Write HTML
    with open(output_html, "w") as f:
        # --- header + nav ---
        f.write(f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Dashboard</title>
<style>
  body{{font-family:Arial,sans-serif;margin:0;padding:0;background:#fafafa}}
  #nav{{position:fixed;top:0;width:100%;background:#333;color:#fff;
       text-align:center;padding:10px;z-index:1000}}
  #nav button,#nav input{{margin:0 6px;padding:6px;border:none;border-radius:4px}}
  .sample{{display:none;padding:80px 24px;max-width:800px;margin:auto}}
  .sample.active{{display:block}}
  .panel{{border-radius:4px;padding:12px;margin:12px 0;}}
  .user{{background:#e8f4ff;border-left:6px solid #4299e1}}
  .think{{background:#fff7e6;border-left:6px solid #ed8936;font-style:italic}}
  .answer{{background:#e6ffed;border-left:6px solid #38a169}}
  .metrics{{background:#fff;padding:12px;border:1px solid #ddd}}
  img.turn-img{{max-width:200px;margin:8px 4px;border:1px solid #ccc}}
</style>
<script>
let pg=0;
function show(n){{pg=Math.max(0,Math.min({total}-1,n));
  document.querySelectorAll('.sample').forEach((s,i)=>s.classList.toggle('active',i===pg));
  document.getElementById('ctr').innerText=(pg+1)+' / {total}';
}}
function nextP(){{show(pg+1)}}function prevP(){{show(pg-1)}}
function gotoP(){{let v=parseInt(document.getElementById('goto').value)||1;show(v-1)}}
window.addEventListener('keydown',e=>{{if(e.key=='ArrowRight')nextP();if(e.key=='ArrowLeft')prevP()}});
window.addEventListener('load',()=>{{show(0)}});
</script>
</head><body>
<div id="nav">
  <button onclick="prevP()">Prev</button>
  <button onclick="nextP()">Next</button>
  <input id="goto" type="number" min="1" max="{total}" value="1">
  <button onclick="gotoP()">Go</button>
  <span id="ctr"></span>
</div>
<h1 style="padding-top:60px;text-align:center;">Inference Dashboard</h1>
""")
# --- each sample page ---
        for si, rec in enumerate(entries):
            turns = parse_conversation(rec["output_str"])
            imgs  = list(rec.get("image_data", []))
            # save images to disk and collect filenames
            saved = []
            for idx, im in enumerate(imgs):
                fn = f"{base}_{rec['env_id']}_img{idx}.png"
                pth = os.path.join(out_dir, fn)
                im.save(pth)
                saved.append(fn)
            """
            data_uris = []
            for im in rec.get("image_data", []):
                buf = io.BytesIO()
                im.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                data_uris.append(f"data:image/png;base64,{b64}")"""

            f.write(f"<section class='sample'>\n")
            f.write(f"<h2>Sample {si+1}: {escape(rec['env_id'])}</h2>\n")
            f.write(f"<div class='metrics'><strong>Config:</strong> {escape(rec['config_id'])}</div>\n")

            img_ptr = 0
            for turn in turns:
                # user panel
                f.write(f"<div class='panel user'><strong>User</strong><br>{escape(turn['user'])}</div>\n")
                # inline images
                for _ in range(turn['n_images']):
                    if img_ptr < len(saved):
                        f.write(f"<img src='{saved[img_ptr]}' class='turn-img'>\n")
                        img_ptr += 1
                """
                # inline images from data_uris
                for _ in range(turn['n_images']):
                    if img_ptr < len(data_uris):
                        f.write(f"<img src='{data_uris[img_ptr]}' class='turn-img'>\n")
                        img_ptr += 1
                """
                # think panel
                if turn['think']:
                    f.write(f"<div class='panel think'><strong>Agent Think</strong><br>{escape(turn['think'])}</div>\n")
                # answer panel
                if turn['answer']:
                    f.write(f"<div class='panel answer'><strong>Agent Answer</strong><br>{escape(turn['answer'])}</div>\n")

            # final metrics
            f.write("<div class='metrics'><strong>Final Metrics:</strong><br>\n")
            for k,v in rec.get("metrics", {}).items():
                f.write(f"{escape(k)}: {escape(str(v))}<br>\n")
            f.write("</div>\n</section>\n")

        f.write("</body></html>")

    print(f"Dashboard written to {output_html}")
    return output_html