import os
import time

import jax
from jax import numpy as jp
import numpy as np
from flax.training import orbax_utils
from flax import struct
from matplotlib import pyplot as plt
import mediapy as media
from orbax import checkpoint as ocp

import mujoco
from mujoco import mjx

from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import html, mjcf, model

xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

URDF_PATH = '../data/urdf/franka_lnd.urdf'

with open(URDF_PATH) as file:
    xml = file.read()

mj_model = mujoco.MjModel.from_xml_string(xml)
mj_data = mujoco.MjData(mj_model)
renderer = mujoco.Renderer(mj_model)

mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.put_data(mj_model, mj_data)

print(mj_data.qpos, type(mj_data.qpos))
print(mjx_data.qpos, type(mjx_data.qpos), mjx_data.qpos.devices())

scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

NUM_WORLDS = 32
NUM_STEPS = 1000

batch = jax.vmap(lambda _ : mjx_data)(jp.arange(NUM_WORLDS))
jit_step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))

print("Finished jit, running sim")
start_time = time.perf_counter()
for _ in range(NUM_STEPS):
    batch = jit_step(mjx_model, batch)
end_time = time.perf_counter()
print("Finished sim")

elapsed_time = end_time - start_time

fps = NUM_STEPS * NUM_WORLDS / elapsed_time
avg_step_time = 1000.0 * elapsed_time / NUM_STEPS

print(f"FPS = {fps}")
print(f"Average step time = {avg_step_time}")

"""
# Running Mujoco MJX simulation multi-world
print("Running multi-world MJX")

NUM_WORLDS = 1024

batch = jax.vmap(lambda _ : mjx_data)(jp.arange(NUM_WORLDS))

jit_step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))
batch = jit_step(mjx_model, batch)

print(batch.qpos)
"""
