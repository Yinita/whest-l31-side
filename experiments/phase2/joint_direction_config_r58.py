"""Fixed R58 development network identities; no corpus ID confusion."""
from pathlib import Path

OUT=Path('results/phase2/joint_direction_mc_r58')
PREPARATION=Path('results/phase2/joint_direction_preparation_r58_r02.json')
SOURCE=Path('estimators/sprint_public_v37b_h18_r01/estimator.py')
SOURCE_SHA='c3dd3282106ba72f5882b2c2e1a0143c3627c373acaae0642c1e0033788ffda2'
DATA_REV='aa99830fdc09fad15407b10e8e3459d3e18bba0a'
LAYERS=(8,14,15)
COLUMNS=tuple(range(0,1024,16))
SAMPLES=65536
SHARDS=64
NETWORKS={
 0:dict(name='logan-fitzgerald',setup_seed=3909487726,
        weights='benchmark/data/rpc_inputs/mlp_0.npz',
        weight_file_sha256='7fb2e3f519b4a90f2c3de9e64d45ad89b7f20ffd254d887e4ba4fa5c1fdf0754',
        inputs='benchmark/data/rpc_inputs/inputs.json',
        dataset='benchmark/data/mini_first8',local_row=0),
 16:dict(name='adam-morrison',setup_seed=250561177,
         weights='benchmark/data/rpc_inputs/frozen_shard02_r01/mlp_0.npz',
         weight_file_sha256='abc7991fdf8e285788b5dfa57753e57ed0db849600c047f0bb9ae76ba8af9a76',
         inputs='benchmark/data/rpc_inputs/frozen_shard02_r01/inputs.json',
         dataset='benchmark/data/mini_shard02',local_row=0)}


def job_seed(network,stream,shard):
    return 2610580000+network*10000+stream*1000+shard
