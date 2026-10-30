"""Independent stdlib-only replay of immutable Stage E records; no relabelling."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = json.loads((a.input/'social_results.json').read_text())
    episodes = {e['id']: e for e in result['episodes']}
    assert len(episodes) == 18 and result['protocol']['collision_population'] == 'physical_all'
    frames = {}
    with (a.input/'social_observations.csv').open(newline='') as stream:
        for r in csv.DictReader(stream):
            key = (r['episode_id'], round(float(r['time_s']), 8))
            frame = frames.setdefault(key, {'objects': {}, 'ego': (float(r['robot_x']), float(r['robot_y']))})
            assert int(r['id']) not in frame['objects']
            frame['objects'][int(r['id'])] = (float(r['x']), float(r['y']))
    checked = 0
    initial_skipped = 0
    with (a.input/'social_predictions.jsonl').open() as stream:
        for line in stream:
            r = json.loads(line)
            key = (r['episode_id'], round(r['time_s'], 8))
            if r['control'] == 0:
                initial_skipped += 1  # Initial observation frame is not in the CSV.
                continue
            frame = frames[key]
            predicted = {int(o['id']): o['current'] for o in r['predictions']}
            assert set(predicted) == set(frame['objects'])
            assert all(math.dist(predicted[k], frame['objects'][k]) < 1e-8 for k in predicted)
            checked += 1
    counts = {}
    for name, e in episodes.items():
        ff = sorted((t, f) for (identity, t), f in frames.items() if identity == name)
        gaps = [(t, min(math.dist(f['ego'], v)-.625 for v in f['objects'].values()),
                 3.5-abs(f['ego'][1])-.325) for t, f in ff]
        assert abs(min(g[1] for g in gaps)-e['min_sampled_ped_gap_m']) < 1e-8
        assert abs(min(g[2] for g in gaps)-e['min_sampled_wall_gap_m']) < 1e-8
        collision = any(min(g[1:]) <= 0 for g in gaps)
        assert collision == (e['outcome'] == 'collision')
        assert len(ff[-1][1]['objects']) == e['initial_targets']
        assert e['outcome'] in ['success', 'collision', 'timeout']
        if e['outcome'] == 'success':
            assert ff[-1][1]['ego'][0] > 40 and e['simulation_time_s'] <= 60.000001
        counts.setdefault(e['variant']['method'], {'success': 0, 'collision': 0, 'timeout': 0})[e['outcome']] += 1
    for n in [4, 8, 12]:
        for seed in [5, 17]:
            pair = [e for e in episodes.values() if e['initial_targets']==n and e['seed']==seed]
            assert len(pair)==3 and len({e['initial_input_sha256'] for e in pair})==1
    out = {'stage_e_commit': 'd252bcfdc13f47dd7de8e109f82663ecb44c0844',
           'run_id': 41002729726, 'episodes': len(episodes), 'frames': len(frames),
           'prediction_observation_frames_independently_matched': checked,
           'initial_frames_not_in_csv': initial_skipped, 'outcomes': counts,
           'physical_identity_assertion_every_world_step_reported_by_runner': result['physical_population_verified_every_world_step'],
           'scope': 'Independent replay at recorded20Hz frames; prediction identity replay at recorded5Hz excluding initial. Not continuous collision verification or future-model calibration.',
           'sha256': {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in a.input.iterdir() if f.is_file()}}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='sha256'}, indent=2))


if __name__ == '__main__':
    main()
