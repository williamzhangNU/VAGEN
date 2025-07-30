# visualize_json.py
import json, os, re
from pathlib import Path
from html import escape
from itertools import zip_longest
from typing import List, Dict
from vagen.env.spatial.Base.tos_base.core.room import Room
# ------- regex helpers -------
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
ANS_RE   = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)

def _extract(tag_re, text):
	m = tag_re.search(text)
	return m.group(1).strip() if m else ""

def split_into_turns(messages: List[Dict]):
	turns = []
	cur = {"user":"", "think":"", "answer":"", "reward":"", "state":"", "raw":""}
	for m in messages:
		role = m.get("role","")
		content = m.get("content","")
		cur["raw"] += content + "\n"
		if role == "user":
			cur["user"] = content
		elif role == "assistant":
			cur["think"]  = _extract(THINK_RE, content)
			cur["answer"] = _extract(ANS_RE, content)
			turns.append(cur)
			cur = {"user":"", "think":"", "answer":"", "reward":"", "state":"", "raw":""}
	if any(cur.values()):
		turns.append(cur)
	return turns

def dict_to_html(d: Dict):
	return "".join(f"<div><strong>{escape(str(k))}:</strong> {escape(str(v))}</div>" for k,v in d.items())

def squash_exp_logs(exp_log: List[Dict]) -> List[Dict]:
	merged = []
	it = iter(exp_log)
	for first in it:
		try:
			second = next(it)
			merged.append({**first, **second})
		except StopIteration:
			merged.append(first)
	return merged

def plot_initial_room(entry, out_dir, base, idx):
	room = Room.from_dict(entry["env_info"]["initial_room"])
	img_name = f"{base}_turn{idx+1}.png"
	img_path = os.path.join(out_dir, img_name)
	room.plot(render_mode='img', save_path=img_path)
	return img_name

"""def plot_final_room(entry, out_dir, base, idx):
	room = Room.from_dict(entry["env_info"]["final_room"])
	img_name = f"{base}_turn{idx+1}_final.png"
	img_path = os.path.join(out_dir, img_name)
	room.plot(render_mode='img', save_path=img_path)
	return img_name"""

def visualize_json(json_path: str, output_html: str, plot_rooms: bool = True):
	"""
	json_path: path to the aggregated json you showed
	output_html: final html file path
	plot_rooms: draw initial room png per sample (needs Room class)
	"""
	with open(json_path, "r") as f:
		data = json.load(f)

	meta = data.get("meta_info", {})
	config_groups = data["config_groups"]  
	total_groups = len(config_groups)
	env_data = data.get("env_data", [])
	overall = data.get("overall_performance", {})

	out_dir = os.path.dirname(output_html)
	base = Path(output_html).stem

	samples_per_group = {
	  gname: len(group["env_data"])
	  for gname, group in config_groups.items()
	}
	total_samples = sum(samples_per_group.values())
	total_pages   = 1 + total_samples   # page 0 = TOC

	# Build a flat list of (group_name, sample_idx, entry)
	flat = []
	for gname, group in config_groups.items():
		for sidx, entry in enumerate(group["env_data"]):
			flat.append((gname, sidx, entry))

	with open(output_html, "w") as f:
		f.write(f"""<!DOCTYPE html>
			<html>
			<head>
			<meta charset="utf-8">
			<title>SpatialGym Dashboard</title>
			<style>
			body{{font-family:Arial,Helvetica,sans-serif;margin:0;padding:0;background:#fafafa;}}
			#nav{{position:fixed;top:0;width:100%;background:#333;color:#fff;padding:10px 0;text-align:center;z-index:999;}}
			#nav button{{margin:0 8px;padding:6px 12px;border:none;background:#555;color:#fff;cursor:pointer;border-radius:4px;}}
			#nav button:hover{{background:#777;}}
			#nav input{{width:70px;padding:4px;border-radius:4px;border:1px solid #999;margin-left:12px;}}
			#counter{{margin-left:12px;font-size:0.9em;opacity:0.85;}}

			.sample-page{{display:none;padding:80px 24px 24px 24px;max-width:1000px;margin:auto;}}
			.sample-page.active{{display:block;}}

			.turn{{background:#fff;border:1px solid #ddd;border-radius:6px;margin:14px 0;padding:12px;}}
			.turn h3{{margin:0 0 6px 0;font-size:16px;color:#333;}}
			.block{{padding:8px 10px;border-radius:4px;margin:6px 0;font-size:14px;line-height:1.5;}}
			.block.user{{background:#e8f4ff;border-left:4px solid #4299e1;}}
			.block.think{{background:#fff7e6;border-left:4px solid #ed8936;font-style:italic;}}
			.block.answer{{background:#e6ffed;border-left:4px solid #38a169;}}

			.metrics{{margin-top:8px;font-size:13px;color:#444;}}
			.metrics div{{margin:2px 0;}}
		img.room{{max-width:220px;height:auto;border:1px solid #ccc;margin-top:6px;}}
		h1{{margin-top:40px;color:#222;text-align:center;}}
		h2{{color:#444;margin-top:20px;}}
		</style>""")

		f.write(f"""
		<script>
		let currentPage = 0;
function showPage(n,total){{
  currentPage = Math.max(0, Math.min(total-1, n));
  const pages = document.querySelectorAll('.sample-page');
  pages.forEach((p,i)=>{{ p.classList.toggle('active', i===currentPage); }});
  document.getElementById('counter').innerText = (currentPage+1)+' / '+total;
  document.getElementById('goto').value = currentPage+1;
  location.hash = '#p'+(currentPage+1);
}}
function nextPage(total){{ showPage(currentPage+1,total); }}
function prevPage(total){{ showPage(currentPage-1,total); }}
function gotoPage(total){{
  const v = parseInt(document.getElementById('goto').value,10);
  if(!isNaN(v)) showPage(v-1,total);
}}
document.addEventListener('keydown', (e)=>{{
  if(e.key==='ArrowRight' || e.key==='PageDown') nextPage({total_pages});
  if(e.key==='ArrowLeft'  || e.key==='PageUp')   prevPage({total_pages});
}});
window.addEventListener('load', ()=>{{
  const m = location.hash.match(/#p(\\d+)/);
  if(m) showPage(parseInt(m[1],10)-1, {total_pages});
  else showPage(0, {total_pages});
}});
</script>


</head>
<body>
<div id='nav'>
  <button onclick="prevPage({total_pages})">Prev</button>
  <button onclick="nextPage({total_pages})">Next</button>
  <input id="goto" type="number" min="1" max="{total_pages}" placeholder="page" onkeydown="if(event.key==='Enter') gotoPage({total_pages});">
  <button onclick="gotoPage({total_pages})">Go</button>
  <span id='counter'></span>
</div>
""")
		f.write(f"<h1>Model: {escape(meta['model_name'])}</h1>\n")
		# --- TOC page ---
		f.write("<section class='sample-page' id='page0'>\n")
		f.write("<h2>Table of Contents</h2>\n<ul>\n")
		running_page = 1
		for gname, group in config_groups.items():
			f.write(f"<li><strong>{escape(gname)}</strong>\n  <ul>\n")
			for idx in range(len(group["env_data"])):
				label = f"Sample {idx+1}"
				f.write(
				  f"    <li>"
				  f"<a href='#' onclick=\"showPage({running_page}, {total_pages});return false;\">"
				  f"{label}</a></li>\n"
				)
				running_page += 1
			f.write("  </ul>\n</li>\n")
		f.write("</ul>\n</section>\n")

		# --- One section per sample ---
		for page_idx, (gname, sidx, entry) in enumerate(flat, start=1):
			f.write(f"<section class='sample-page' id='page{page_idx}'>\n")
			f.write(f"<h2>{escape(gname)} — Sample {sidx+1}</h2>\n")

			if plot_rooms:
				img_name = plot_initial_room(entry, out_dir, base, page_idx)
				if img_name:
					f.write(f"<img src='{img_name}' class='room'>\n")

			# env config
			cfg = entry["env_info"]["config"]
			f.write("<div class='metrics'><strong>Env Config</strong>")
			f.write(dict_to_html(cfg))
			f.write("</div>\n")

			# conversation turns + metrics + evaluation answers
			turns   = split_into_turns(entry["message"])
			exp_log = squash_exp_logs(entry["exploration_metrics_log"])
			eval_answers = entry.get("evaluation_metrics_log", [])
			# attach eval answers to last turns
			for i, ans in enumerate(eval_answers):
				turn_idx = len(turns) - len(eval_answers) + i
				if 0 <= turn_idx < len(turns):
					turns[turn_idx]["eval_answer"] = ans

			for t_idx, (turn, em) in enumerate(zip_longest(turns, exp_log, fillvalue={})):
				f.write("<div class='turn'>\n")
				f.write(f"<h3>Turn {t_idx+1}</h3>\n")
				if turn.get("user"):
					u = escape(turn["user"]).replace("\n","<br>")
					f.write(f"<div class='block user'><strong>User</strong><br>{u}</div>\n")
				if turn.get("think"):
					th = escape(turn["think"]).replace("\n","<br>")
					f.write(f"<div class='block think'><strong>Agent Think</strong><br>{th}</div>\n")
				if turn.get("answer"):
					an = escape(turn["answer"]).replace("\n","<br>")
					f.write(f"<div class='block answer'><strong>Agent Answer</strong><br>{an}</div>\n")
				if "eval_answer" in turn:
					eva = ", ".join(turn["eval_answer"])
					f.write(f"<div class='block evaluation'><strong>Evaluation Answer</strong><br>{escape(eva)}</div>\n")

				f.write("<div class='metrics'><strong>Turn Metrics</strong>")
				f.write(dict_to_html(em) or "<div>(none)</div>")
				f.write("</div>\n")
				f.write("</div>\n")
			"""if plot_rooms:
				img_name = plot_final_room(entry, out_dir, base, page_idx)
				if img_name:
					f.write(f"<img src='{img_name}' class='room'>\n")"""
			# final metrics
			fe = entry["exploration_efficiency"]
			fv = entry["evaluation_performance"]
			f.write("<div class='metrics'><strong>Sample Final Metrics</strong>")
			f.write(dict_to_html({**fe, **fv}))
			f.write("</div>\n")

			f.write("</section>\n")

		f.write("</body></html>")

	return output_html


		