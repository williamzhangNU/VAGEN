#!/usr/bin/env python3
"""
Compute per-sample evaluation statistics from env_data.json files.

Usage:
  python scripts/compute_env_samples_stats.py /path/to/env_data.json [more.json ...]

The script accepts 1-3 JSON file paths. For each file it finds keys named
`sample_run0` .. `sample_run24` (or any keys starting with `sample_run`) and
extracts numeric evaluation scores for each sample. For each sample it
computes:
  - count: number of numeric scores found
  - mean
  - variance (population p-variance)
  - max difference (max - min)

It also computes statistics across the 25 sample means: their variance and
max difference.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from statistics import mean, pvariance
from typing import Any, Dict, List, Tuple


def is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def find_numeric_leaves(obj: Any) -> List[float]:
    """Recursively collect numeric leaves from a JSON-like object."""
    out: List[float] = []
    if is_number(obj):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(find_numeric_leaves(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(find_numeric_leaves(v))
    return out


def extract_scores_for_sample(value: Any) -> List[float]:
    """Try to intelligently extract a list of numeric evaluation scores for a sample."""
    # If it's already a list of numbers, return that.
    if isinstance(value, list) and all(is_number(x) for x in value):
        return [float(x) for x in value]

    # If it's a dict and maps run names -> numbers, collect numeric values.
    if isinstance(value, dict):
        # Common keys that may hold scores
        for key in ("evaluation", "eval", "score", "scores", "reward", "acc", "accuracy"):
            if key in value:
                v = value[key]
                if isinstance(v, list) and all(is_number(x) for x in v):
                    return [float(x) for x in v]
                if is_number(v):
                    return [float(v)]
                # otherwise fall through to recursive search

        # If dict looks like run->number mapping, collect numeric leaves at this level
        numeric_vals = [float(v) for v in value.values() if is_number(v)]
        if numeric_vals:
            return numeric_vals

    # Fallback: collect all numeric leaves recursively
    return find_numeric_leaves(value)


def compute_stats(values: List[float]) -> Tuple[int, float, float, float]:
    """Return (count, mean, variance (pvariance), max_diff)."""
    cnt = len(values)
    if cnt == 0:
        return 0, float('nan'), float('nan'), float('nan')
    m = mean(values)
    var = pvariance(values) if cnt >= 1 else float('nan')
    max_diff = float(max(values) - min(values)) if cnt >= 1 else float('nan')
    return cnt, m, var, max_diff


def compute_cross_file_stats(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute statistics across files for the same sample_run and metric.

    Returns a mapping with keys 'sample_runs' -> sample_key -> {
      'mean_across_files': {count, mean, variance, max_diff},
      'per_modality_across_files': {modality: {count, mean_across_files, variance_across_files, max_diff_across_files}}
    }
    """
    sample_keys = set()
    for res in all_results:
        sample_keys.update(res.get('samples', {}).keys())

    cross: Dict[str, Any] = {'sample_runs': {}}

    for key in sorted(sample_keys):
        # collect per-file sample means and per-modality values
        means: List[float] = []
        mod_vals: Dict[str, List[float]] = {}

        for res in all_results:
            s = res.get('samples', {}).get(key)
            if not s:
                continue
            # compute this file's sample mean from its per-modality values
            per_mod = s.get('per_modality', {})
            vals = [float(v) for v in per_mod.values() if is_number(v) and not math.isnan(v)]
            for mod, v in per_mod.items():
                if is_number(v) and not math.isnan(v):
                    mod_vals.setdefault(mod, []).append(float(v))

        sample_entry: Dict[str, Any] = {}

        per_mod_stats: Dict[str, Dict[str, float]] = {}
        for mod, vals in mod_vals.items():
            c, m, var, maxdiff = compute_stats(vals)
            per_mod_stats[mod] = {
                'count': c,
                'mean_across_files': m,
                'variance_across_files': var,
                'max_diff_across_files': maxdiff,
            }

        sample_entry['per_modality_across_files'] = per_mod_stats
        cross['sample_runs'][key] = sample_entry

    return cross


def format_cross_file_results(cross: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append('sample_run')
    header = f"{ 'sample':20s} {'mean_cnt':>8s} {'mean':>12s} {'mean_var':>12s} {'mean_maxdiff':>12s}"
    lines.append(header)
    lines.append('-' * len(header))
    for key, info in sorted(cross.get('sample_runs', {}).items(), key=lambda kv: kv[0]):
        lines.append(f"{key:20s}")

        # per-modality lines (indented)
        per_mod = info.get('per_modality_across_files', {})
        if per_mod:
            lines.append('  Per-modality:')
            for mod, s in sorted(per_mod.items(), key=lambda kv: kv[0]):
                lines.append(
                    f"    {mod:30s} {s['count']:5d} {s['mean_across_files']:12.6g} {s['variance_across_files']:12.6g} {s['max_diff_across_files']:12.6g}"
                )

    return '\n'.join(lines)


def process_file(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text())
    samples_obj = data['samples']

    # samples_obj is expected to be a mapping: sample_runN -> sample_data
    pattern = re.compile(r"^sample_run(\d+)")
    ordered_keys: List[Tuple[int, str]] = []
    for k in samples_obj.keys():
        m = pattern.match(k)
        if not m:
            # skip any keys that are not sample_runN
            continue
        idx = int(m.group(1))
        # only include sample indices 0..24 (inclusive)
        if 0 <= idx <= 24:
            ordered_keys.append((idx, k))
        else:
            # skip samples outside the 0-24 range
            continue

    ordered_keys.sort()

    results: Dict[str, Any] = {
        'file': str(path),
        'samples': {},
    }

    # We'll extract avg_accuracy per modality (task) for each sample.
    # Expected modalities example: 'vision_active_think', 'vision_passive_think_scout',
    # 'text_active_think', 'text_passive_think_strategist'.

    for _, key in ordered_keys:
        sample_data = samples_obj.get(key)
        if sample_data is None:
            continue

        per_mod_vals: Dict[str, float] = {}

        # sample_data contains modalities as keys. Read the fixed path
        # `metrics -> evaluation -> overall -> avg_accuracy` for each modality
        # without any type checks or fallbacks.
        for mod_key, mod_obj in sample_data.items():
            val = float(mod_obj['metrics']['evaluation']['overall']['avg_accuracy'])
            per_mod_vals[mod_key] = val

        # Store per-modality values for this sample. Do not compute per-sample
        # mean/variance here — cross-file aggregation will compute those.
        results['samples'][key] = {
            'per_modality': per_mod_vals,
        }

    # Summary: only report how many samples were found in this file. Don't
    # compute across-sample statistics here; that's the responsibility of the
    # cross-file aggregator.
    results['summary'] = {
        'num_samples_considered': len(results['samples']),
    }

    return results


def format_results(results: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"File: {results['file']}")
    lines.append("")
    lines.append("Per-sample (per-modality) values:")
    for key, info in sorted(results['samples'].items(), key=lambda kv: kv[0]):
        lines.append(f"{key}:")
        per_mod = info.get('per_modality', {})
        if per_mod:
            for mod, v in sorted(per_mod.items(), key=lambda kv: kv[0]):
                lines.append(f"  {mod:30s} {v:12.6g}")
        else:
            lines.append("  (no modality values)")

    lines.append("")
    s = results['summary']
    lines.append("Summary:")
    lines.append(f"  Samples considered: {s['num_samples_considered']}")

    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='*', help='1-3 paths to env_data.json files',
                    default=['results_arxiv_4room_1/gpt-5.2/env_data.json',
                             'results_arxiv_4room_2/gpt-5.2/env_data.json',
                             ])
    ap.add_argument('--save-json', '-o', help='optional path to save JSON summary')
    args = ap.parse_args(argv)

    files = [Path(p) for p in args.files]
    if not (1 <= len(files) <= 3):
        ap.error('Provide between 1 and 3 JSON file paths')

    all_results = []
    for p in files:
        if not p.exists():
            print(f"File not found: {p}")
            continue
        try:
            res = process_file(p)
        except Exception as e:
            print(f"Error processing {p}: {e}")
            continue
        print(format_results(res))
        print('\n' + '='*60 + '\n')
        all_results.append(res)

    if args.save_json and all_results:
        # Save cross-file aggregated metrics (variances, max_diffs, counts).
        # The user requested saving the aggregated metrics rather than each
        # file's raw data.
        cross_to_save = compute_cross_file_stats(all_results)
        out_obj = {
            'files': [r.get('file') for r in all_results],
            'cross_file_stats': cross_to_save,
        }
        Path(args.save_json).write_text(json.dumps(out_obj, indent=2))
        print(f"Saved cross-file JSON summary to {args.save_json}")

    # If multiple files were provided, compute cross-file statistics for the
    # same `sample_run` and the same metric (per-modality values and sample mean).
    if len(all_results) > 1:
        cross = compute_cross_file_stats(all_results)
        print('\nCross-file (per-sample_run) stats across provided files:')
        print(format_cross_file_results(cross))

if __name__ == '__main__':
    main()
