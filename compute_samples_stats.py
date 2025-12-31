#!/usr/bin/env python3
"""
Compute per-sample evaluation statistics from env_data.json files.

Usage:
  python scripts/compute_env_samples_stats.py /path/to/env_data.json [more.json ...] --samples 0-24
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Set, Tuple, Union

def is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)

def parse_indices(s: str) -> Set[int]:
    """Parse string like '0-24' or '0,1,5' into a set of integers."""
    res = set()
    if not s:
        return res
    for part in s.split(','):
        part = part.strip()
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                res.update(range(start, end + 1))
            except ValueError:
                continue
        else:
            try:
                res.add(int(part))
            except ValueError:
                continue
    return res

def compute_stats(values: List[float]) -> Dict[str, float]:
    """Return {count, mean, stdev}."""
    cnt = len(values)
    if cnt == 0:
        return {'count': 0, 'mean': float('nan'), 'stdev': float('nan')}
    m = mean(values)
    s = stdev(values) if cnt > 1 else 0.0
    return {'count': cnt, 'mean': m, 'stdev': s}

def compute_pooled_stats(all_values: List[List[float]]) -> Dict[str, float]:
    """Compute overall mean and pooled standard deviation."""
    flat_values = [x for sub in all_values for x in sub]
    if not flat_values:
         return {'mean': float('nan'), 'pooled_stdev': float('nan')}
    
    overall_mean = mean(flat_values)
    
    # Pooled variance: sum((n-1)*var) / sum(n-1)
    numerator = 0.0
    denominator = 0
    
    for group in all_values:
        n = len(group)
        if n > 1:
            s = stdev(group)
            numerator += (n - 1) * (s ** 2)
            denominator += (n - 1)
            
    pooled_sd = math.sqrt(numerator / denominator) if denominator > 0 else 0.0
    
    return {'mean': overall_mean, 'pooled_stdev': pooled_sd}

def process_file(path: Path, sample_indices: Set[int]) -> Dict[str, Any]:
    data = json.loads(path.read_text())
    samples_obj = data.get('samples', {})
    
    pattern = re.compile(r"^sample_run(\d+)")
    results: Dict[str, Any] = {'file': str(path), 'samples': {}}
    
    for k, v in samples_obj.items():
        m = pattern.match(k)
        if not m: continue
        idx = int(m.group(1))
        if idx not in sample_indices: continue
        
        # Structure: modality -> { overall_acc, tasks: {name -> acc} }
        per_modality = {}
        if isinstance(v, dict):
             for mod_name, mod_data in v.items():
                 if not isinstance(mod_data, dict): continue
                 metrics = mod_data.get('metrics', {}).get('evaluation', {})
                 
                 # Overall
                 overall_acc = metrics.get('overall', {}).get('avg_accuracy')
                 
                 # Tasks
                 tasks_data = metrics.get('per_task', {})
                 tasks_acc = {}
                 if isinstance(tasks_data, dict):
                     for t_name, t_val in tasks_data.items():
                         if isinstance(t_val, dict) and 'avg_accuracy' in t_val:
                             tasks_acc[t_name] = float(t_val['avg_accuracy'])
                 
                 if overall_acc is not None:
                     per_modality[mod_name] = {
                         'overall': float(overall_acc),
                         'tasks': tasks_acc
                     }

        results['samples'][k] = {'per_modality': per_modality}
        
    results['summary'] = {'num_samples_considered': len(results['samples'])}
    return results

def compute_cross_file_stats(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    # map: sample -> modality -> 'overall' -> [vals]
    #                          -> 'tasks' -> task_name -> [vals]
    data_map = {} 
    
    for res in all_results:
        for s_key, s_val in res['samples'].items():
            if s_key not in data_map: data_map[s_key] = {}
            for mod, mod_val in s_val['per_modality'].items():
                if mod not in data_map[s_key]: 
                    data_map[s_key][mod] = {'overall': [], 'tasks': {}}
                
                data_map[s_key][mod]['overall'].append(mod_val['overall'])
                
                for t_name, t_acc in mod_val['tasks'].items():
                    if t_name not in data_map[s_key][mod]['tasks']:
                        data_map[s_key][mod]['tasks'][t_name] = []
                    data_map[s_key][mod]['tasks'][t_name].append(t_acc)

    cross_samples = {}
    global_agg = {}
    
    for s_key, modalities in data_map.items():
        cross_samples[s_key] = {}
        for mod, content in modalities.items():
            overall_stats = compute_stats(content['overall'])
            
            tasks_stats = {}
            for t_name, vals in content['tasks'].items():
                tasks_stats[t_name] = compute_stats(vals)
                
            cross_samples[s_key][mod] = {
                'overall': overall_stats,
                'tasks': tasks_stats
            }
            
            if mod not in global_agg: 
                global_agg[mod] = {'overall': [], 'tasks': {}}
            
            global_agg[mod]['overall'].append(content['overall'])
            for t_name, vals in content['tasks'].items():
                if t_name not in global_agg[mod]['tasks']:
                    global_agg[mod]['tasks'][t_name] = []
                global_agg[mod]['tasks'][t_name].append(vals)
                
    global_stats = {}
    for mod, content in global_agg.items():
        overall_pooled = compute_pooled_stats(content['overall'])
        tasks_pooled = {}
        for t_name, groups in content['tasks'].items():
            tasks_pooled[t_name] = compute_pooled_stats(groups)
            
        global_stats[mod] = {
            'overall': overall_pooled,
            'tasks': tasks_pooled
        }
        
    return {'sample_runs': cross_samples, 'overall_stats': global_stats}

def format_stats(stats_dict: Dict[str, Any], title: str) -> str:
    lines = [title, '-' * len(title)]
    for key, info in sorted(stats_dict.items()):
        # Check if info is Modality->Stats (Overall Stats case) or Sample->Modality->Stats (Sample Runs case)
        if 'overall' in info and 'tasks' in info:
            # Overall Stats case: key is Modality, info is Stats
            mod = key
            s = info
            ov = s['overall']
            if 'stdev' in ov:
                lines.append(f"  {mod:30s} Overall: Mean={ov['mean']:.4f}, SD={ov['stdev']:.4f}, N={ov['count']}")
            elif 'pooled_stdev' in ov:
                lines.append(f"  {mod:30s} Overall: Mean={ov['mean']:.4f}, PooledSD={ov['pooled_stdev']:.4f}")
            
            for t_name, t_s in sorted(s['tasks'].items()):
                if 'stdev' in t_s:
                     lines.append(f"    {t_name:30s} Mean={t_s['mean']:.4f}, SD={t_s['stdev']:.4f}, N={t_s['count']}")
                elif 'pooled_stdev' in t_s:
                     lines.append(f"    {t_name:30s} Mean={t_s['mean']:.4f}, PooledSD={t_s['pooled_stdev']:.4f}")
        else:
            # Sample Runs case: key is SampleName, info is Dict[Modality, Stats]
            lines.append(f"{key}:")
            if isinstance(info, dict):
                for mod, s in sorted(info.items()):
                    if not isinstance(s, dict) or 'overall' not in s: continue
                    
                    ov = s['overall']
                    if 'stdev' in ov:
                        lines.append(f"  {mod:30s} Overall: Mean={ov['mean']:.4f}, SD={ov['stdev']:.4f}, N={ov['count']}")
                    elif 'pooled_stdev' in ov:
                        lines.append(f"  {mod:30s} Overall: Mean={ov['mean']:.4f}, PooledSD={ov['pooled_stdev']:.4f}")
                    
                    for t_name, t_s in sorted(s['tasks'].items()):
                        if 'stdev' in t_s:
                             lines.append(f"    {t_name:30s} Mean={t_s['mean']:.4f}, SD={t_s['stdev']:.4f}, N={t_s['count']}")
                        elif 'pooled_stdev' in t_s:
                             lines.append(f"    {t_name:30s} Mean={t_s['mean']:.4f}, PooledSD={t_s['pooled_stdev']:.4f}")
    return '\n'.join(lines)

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='*', help='paths to env_data.json files')
    ap.add_argument('--samples', default='0-24', help='Range of samples to process (e.g. 0-24, 0,1,5)')
    ap.add_argument('--save-json', '-o', help='optional path to save JSON summary')
    args = ap.parse_args(argv)

    files = [Path(p) for p in args.files]
    sample_indices = parse_indices(args.samples)
    
    assert files

    all_results = []
    for p in files:
        if not p.exists():
            continue
        try:
            res = process_file(p, sample_indices)
            # Print simple summary for file
            print(f"File: {res['file']}")
            print(f"Samples considered: {res['summary']['num_samples_considered']}")
            print("-" * 40)
            all_results.append(res)
        except Exception as e:
            print(f"Error processing {p}: {e}")

    if not all_results:
        print("No valid results found.")
        return

    cross = compute_cross_file_stats(all_results)
    
    if len(all_results) > 1:
        print('\n' + format_stats(cross['sample_runs'], "Per-Sample Statistics (across files)"))

    print('\n' + format_stats(cross['overall_stats'], "Overall Statistics (across all samples)"))

    if args.save_json:
        out_obj = {
            'files': [r.get('file') for r in all_results],
            'cross_file_stats': cross,
        }
        Path(args.save_json).write_text(json.dumps(out_obj, indent=2))
        print(f"Saved JSON summary to {args.save_json}")

if __name__ == '__main__':
    main()
