# Third-party notices

The bundled `legged_gym` base environment and `rsl_rl` PPO implementation are
derived from the copies used by LZHMine, which in turn are based on NVIDIA's
Legged Gym and ETH Zurich's RSL-RL. Their source files retain the original SPDX
headers and copyright notices. The applicable BSD 3-Clause license text is
included in `licenses/BSD-3-Clause.txt`.

The DreamWaQ history encoder, variational estimator, asymmetric actor-critic,
and PPO integration are adapted from the local `DreamWAQ-Go2W` reference
implementation. The Nezha version preserves its network/data-flow design while
making the observation dimensions task-specific and clearing history on reset.

NVIDIA Isaac Gym is not redistributed by this project. Install Isaac Gym
Preview 4 separately under NVIDIA's license before running simulation or
training.
