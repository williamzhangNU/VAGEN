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
import sys
sys.path.insert(0, str(Path(__file__).parent))
from vagen.env.spatial.Base.tos_base.utils.cogmap.correlation import compute_correlation_metrics

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

def extract_samples_from_file(path: Path, sample_indices: Set[int], pattern: re.Pattern) -> Dict[str, Any]:
    """
    Extract sample data from a single env_data.json file.
    
    Returns:
        Dict mapping sample_key -> sample data
    """
    data = json.loads(path.read_text())
    samples_obj = data.get('samples', {})
    samples_data = {}
    
    for k, v in samples_obj.items():
        m = pattern.match(k)
        if not m:
            continue
        idx = int(m.group(1))
        if idx not in sample_indices:
            continue
        
        if isinstance(v, dict):
            samples_data[k] = {'idx': idx, 'data': v}
    
    return samples_data

def process_file(path: Path, sample_indices: Set[int]) -> Dict[str, Any]:
    pattern = re.compile(r"^sample_run(\d+)")
    samples_data = extract_samples_from_file(path, sample_indices, pattern)
    
    results: Dict[str, Any] = {'file': str(path), 'samples': {}}
    
    for k, sample_info in samples_data.items():
        v = sample_info['data']
        
        # Structure: modality -> { overall_acc, tasks: {name -> acc} }
        per_modality = {}
        for mod_name, mod_data in v.items():
            if not isinstance(mod_data, dict):
                continue
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

def compute_cogmap_correlation_across_paths(file_paths: List[Path], sample_indices: Set[int]) -> Dict[str, Any]:
    """
    Compute average cogmap last_global_vs_gt_full overall metric across multiple paths
    (for each sample) and calculate correlation with each evaluation task.
    Each modality containing 'active' will have its own correlation results.
    
    Args:
        file_paths: List of paths to env_data.json files
        sample_indices: Set of sample indices to process
        
    Returns:
        Dictionary containing correlation metrics per modality
    """
    # Collect data grouped by modality and sample index
    # modality_name -> sample_idx -> list of metrics from different paths
    modality_sample_data_map = {}
    pattern = re.compile(r"^sample_run(\d+)")
    
    for path in file_paths:
        if not path.exists():
            print(f"Warning: File {path} does not exist, skipping.")
            continue
            
        try:
            samples_data = extract_samples_from_file(path, sample_indices, pattern)
            
            for k, sample_info in samples_data.items():
                idx = sample_info['idx']
                v = sample_info['data']
                
                # Extract metrics from modalities containing 'active' in name
                for mod_name, mod_data in v.items():
                    if not isinstance(mod_data, dict):
                        continue
                    
                    if 'active' in mod_name:
                        metrics = mod_data.get('metrics', {})
                        if metrics:
                            if mod_name not in modality_sample_data_map:
                                modality_sample_data_map[mod_name] = {}
                            if idx not in modality_sample_data_map[mod_name]:
                                modality_sample_data_map[mod_name][idx] = []
                            modality_sample_data_map[mod_name][idx].append({
                                'file': str(path),
                                'metrics': metrics
                            })
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue
    
    if not modality_sample_data_map:
        return {
            'status': 'error',
            'message': 'No valid modality data containing "active" found',
            'n_samples': 0
        }
    
    # Compute correlation results for each modality separately
    all_modality_results = {}
    
    print(f"\nProcessing {len(modality_sample_data_map)} modalities...")
    for mod_name, sample_data_map in modality_sample_data_map.items():
        print(f"\n=== Modality: {mod_name} ===")
        print(f"  Processing {len(sample_data_map)} samples...")
        
        # Compute average metrics for each sample across paths
        averaged_env_data = []
        
        for idx in sorted(sample_data_map.keys()):
            path_metrics_list = sample_data_map[idx]
            print(f"    Sample {idx}: Found {len(path_metrics_list)} entries")
            
            # Collect and average cogmap scores
            cogmap_scores = [
                float(pm['metrics']['cogmap']['exploration']['correctness']['last_global_vs_gt_full']['overall'])
                for pm in path_metrics_list
                if isinstance(pm['metrics'].get('cogmap', {}).get('exploration', {}).get('correctness', {})
                           .get('last_global_vs_gt_full', {}).get('overall'), (int, float))
            ]
            
            if not cogmap_scores:
                continue
            
            # Collect and average evaluation metrics
            overall_accs = []
            per_task_accs = {}
            
            for pm in path_metrics_list:
                metrics = pm['metrics']
                eval_m = metrics.get('evaluation', {})
                
                # Overall accuracy
                overall_acc = eval_m.get('overall', {}).get('avg_accuracy')
                if isinstance(overall_acc, (int, float)):
                    overall_accs.append(float(overall_acc))
                
                # Per-task accuracy
                for task_name, task_data in eval_m.get('per_task', {}).items():
                    if isinstance(task_data, dict):
                        task_acc = task_data.get('avg_accuracy')
                        if isinstance(task_acc, (int, float)):
                            per_task_accs.setdefault(task_name, []).append(float(task_acc))
            
            # Construct averaged env_data entry
            eval_metrics = {}
            if overall_accs:
                eval_metrics['overall'] = {'avg_accuracy': mean(overall_accs)}
            if per_task_accs:
                eval_metrics['per_task'] = {task: {'avg_accuracy': mean(accs)} 
                                           for task, accs in per_task_accs.items()}
            
            averaged_env_data.append({
                'metrics': {
                    'cogmap': {
                        'exploration': {
                            'correctness': {
                                'last_global_vs_gt_full': {
                                    'overall': mean(cogmap_scores)
                                }
                            }
                        }
                    },
                    'evaluation': eval_metrics
                },
                'sample_idx': idx,
                'num_paths': len(cogmap_scores),
                'cogmap_scores_across_paths': cogmap_scores
            })
        
        if not averaged_env_data:
            all_modality_results[mod_name] = {
                'status': 'error',
                'message': 'No valid averaged data could be computed',
                'n_samples': 0
            }
            continue
        
        # Calculate correlations and add summary
        correlation_results = compute_correlation_metrics(averaged_env_data, exp_type='active')
        correlation_results.update({
            'files_processed': len(file_paths),
            'samples_averaged': len(averaged_env_data),
            'sample_indices': sorted(sample_data_map.keys())
        })
        
        all_modality_results[mod_name] = correlation_results
    
    return {
        'modalities': all_modality_results,
        'num_modalities': len(all_modality_results)
    }

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='*', help='paths to env_data.json files')
    ap.add_argument('--samples', default='0-24', help='Range of samples to process (e.g. 0-24, 0,1,5)')
    ap.add_argument('--save-json', '-o', help='optional path to save JSON summary')
    ap.add_argument('--compute-cogmap-correlation', action='store_true', 
                    help='Compute cogmap correlation across paths')
    args = ap.parse_args(argv)

    files = [Path(p) for p in args.files]
    sample_indices = parse_indices(args.samples)
    
    assert files

    # If compute-cogmap-correlation flag is set, use the new function
    if args.compute_cogmap_correlation:
        print("Computing cogmap correlation across paths...")
        results = compute_cogmap_correlation_across_paths(files, sample_indices)
        
        print("\n" + "=" * 80)
        print(f"COGMAP CORRELATION RESULTS ({results.get('num_modalities', 0)} modalities)")
        print("=" * 80)
        
        for mod_name, corr_res in results.get('modalities', {}).items():
            print(f"\n{'=' * 80}\nMODALITY: {mod_name}\n{'=' * 80}")
            print(f"Files: {corr_res.get('files_processed', 0)} | "
                  f"Samples: {corr_res.get('samples_averaged', 0)} | "
                  f"Valid: {corr_res.get('n_samples', 0)}")
            
            if 'cogmap_acc_correlations' in corr_res:
                print("\n--- Cogmap vs Accuracy Correlations ---")
                for task, data in corr_res['cogmap_acc_correlations'].items():
                    if isinstance(data, dict) and data.get('pearson_r') is not None:
                        sig = " *" if data.get('significant', False) else ""
                        print(f"  {task:40s}: r={data['pearson_r']:7.4f}, "
                              f"p={data['p_value']:7.4f}, n={data['n_samples']}{sig}")
                    else:
                        print(f"  {task:40s}: No correlation (insufficient data)")
            
            if 'cogmap_infogain_correlation' in corr_res:
                print("\n--- Cogmap vs Information Gain Correlation ---")
                data = corr_res['cogmap_infogain_correlation']
                if isinstance(data, dict) and data.get('pearson_r') is not None:
                    sig = " *" if data.get('significant', False) else ""
                    print(f"  Information Gain: r={data['pearson_r']:7.4f}, "
                          f"p={data['p_value']:7.4f}, n={data['n_samples']}{sig}")
                else:
                    print(f"  Information Gain: No correlation (insufficient data)")
        
        if args.save_json:
            Path(args.save_json).write_text(json.dumps(results, indent=2))
            print(f"\nSaved correlation results to {args.save_json}")
        
        return

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
