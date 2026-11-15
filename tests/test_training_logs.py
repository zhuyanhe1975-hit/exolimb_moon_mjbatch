from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from training_logs import clarify_reward_logs


def test_reward_log_labels_keep_raw_and_weighted_values():
    log={'reward_episode/centroidal_angular_momentum':19966.1152,
         'reward_episode/weighted/centroidal_angular_momentum':-3.327686,
         'reward_step/joint_torques':4457.,
         'reward_step/weighted/joint_torques':-.00015,
         'Metrics/momentum_kg_m2_s':8.22}
    out=clarify_reward_logs(log)
    assert out['reward_episode/centroidal_angular_momentum']==-3.327686
    assert out['raw_episode/centroidal_angular_momentum']==19966.1152
    assert out['reward_step/joint_torques']==-.00015
    assert out['raw_step/joint_torques']==4457.
    assert out['Metrics/momentum_kg_m2_s']==8.22
