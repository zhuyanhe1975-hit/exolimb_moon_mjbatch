"""Read existing training logs; never start training or access the GPU."""
import argparse
import json
from pathlib import Path
from statistics import mean, median
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('runs',nargs='+',type=Path)
parser.add_argument('--warmup',type=int,default=20)
args=parser.parse_args()
if args.warmup<0: parser.error('--warmup must be nonnegative')
for path in args.runs:
    config=json.loads((path/'training_config.json').read_text())
    runner=config['runner'];mdp=config['mdp']
    events=EventAccumulator(str(path));events.Reload()
    result={'run':str(path),'mdp_sha256':config['mdp_sha256'],
        'worlds':runner['nworld'],'steps_per_env':runner['num_steps_per_env'],
        'compile_mode':runner.get('torch_compile_mode'),
        'ppo':runner['algorithm'],'physics_dt':mdp['physics_dt'],
        'control_dt':mdp['control_dt'],'warmup_excluded_before_step':args.warmup}
    for tag in ('Perf/collection_time','Perf/learning_time','Perf/total_fps','Perf/wall_cycle_time','Perf/wall_cycle_fps'):
        if tag not in events.Tags()['scalars']:continue
        values=[x.value for x in events.Scalars(tag) if x.step>=args.warmup]
        if not values: raise ValueError(f'{path}: no {tag} samples after warmup')
        result[tag]={'samples':len(values),'mean':mean(values),'median':median(values)}
    print(json.dumps(result,ensure_ascii=False,indent=2))
