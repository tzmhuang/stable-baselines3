import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gym import spaces

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv

SelfOnPolicyAlgorithm = TypeVar("SelfOnPolicyAlgorithm", bound="OnPolicyAlgorithm")


class OnPolicyAlgorithm(BaseAlgorithm):
    """
    The base for On-Policy algorithms (ex: A2C/PPO).

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    :param supported_action_spaces: The action spaces supported by the algorithm.
    """

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        n_steps: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: Optional[Tuple[spaces.Space, ...]] = None,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            support_multi_env=True,
            seed=seed,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            supported_action_spaces=supported_action_spaces,
        )

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        buffer_cls = DictRolloutBuffer if isinstance(self.observation_space, spaces.Dict) else RolloutBuffer

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        # NOTE: for debug only
        # infos_list = []

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            if env.reward_shaper != None:
                rewards = env.reward_shaper(rewards)
            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs)
            self._last_obs = new_obs
            self._last_episode_starts = dones

            # NOTE: for debug only
            # infos_list.append(infos)

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True
        # return True, infos_list # NOTE: for debug only

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def learn(
        self: SelfOnPolicyAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfOnPolicyAlgorithm:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)
            # continue_training, info_list = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                    # vvv modified vvv
                    self.logger.record("rollout/ep_success_rate", safe_mean([ep_info["is_success"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_mean_dtg", safe_mean([ep_info["dist_to_goal"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                # Added rollout log for reward min max

                # # get reward keys   NOTE: Debug only
                # reward_keys = [k for k in info_list[0][0].keys() if k.split('_')[0] == 'reward']
                # # # arrange info into {<rewards>: [step, envs]}
                # for rk in reward_keys:
                #     rwd_arr = np.array([info_env[rk].detach().cpu().numpy() for info_step in info_list for info_env in info_step])
                #     self.logger.record("reward/max_{}".format(rk), rwd_arr.max())
                #     self.logger.record("reward/min_{}".format(rk), rwd_arr.min())

                # # get rewards max/min id for each
                # for rk in reward_keys:
                #     rwd_arr = np.array([info_env[rk].detach().cpu().numpy() for info_step in info_list for info_env in info_step])
                #     self.logger.record("reward/argmax_{}".format(rk), np.argmax(rwd_arr))
                #     self.logger.record("reward/argmin_{}".format(rk), np.argmin(rwd_arr))


                # for rk in reward_keys:
                #     step_n = 6
                #     # env_id = 3988
                #     rwd_arr = np.array([info_env[rk].detach().cpu().numpy() for info_env in info_list[step_n]])
                #     # rwd_arr = info_list[step_n][env_id][rk].item()
                #     self.logger.record("reward_6/max_{}".format(rk), rwd_arr.max())
                #     argmax_idx = np.argmax(rwd_arr)
                #     self.logger.record("reward_6/argmax_{}".format(rk), argmax_idx)
                #     # save restart for all rollout
                
                # rots = []
                # torso_pos = []
                # torso_rot = []
                # torso_vel = []
                # torso_ang_vel = []
                # dof_pos = []
                # dof_vel = []
                # for info_step in info_list:
                #     for info_env in info_step:
                #         rots.append(info_env['init_heading'].detach().cpu().numpy())
                #         torso_pos.append(info_env['torso_position'].detach().cpu().numpy())
                #         torso_rot.append(info_env['torso_rotation'].detach().cpu().numpy())
                #         torso_vel.append(info_env['torso_velocity'].detach().cpu().numpy())
                #         torso_ang_vel.append(info_env['torso_ang_velocity'].detach().cpu().numpy())
                #         dof_pos.append(info_env['dof_pos'].detach().cpu().numpy())
                #         dof_vel.append(info_env['dof_vel'].detach().cpu().numpy())
                

                # np.savez_compressed("/data/zanming/Omniverse/OmniIsaacGymEnvs/omniisaacgymenvs/logs/SimplePPO_3a_fix_visualize_foundid_rgb_4_earlyObsClamp/debug/debug_{:06d}.npz".format(iteration),\
                #     dones_buf=self.rollout_buffer.episode_starts, start_rot=rots, iteration=iteration, num_timestep=self.num_timesteps,
                #     actions=self.rollout_buffer.actions, torso_pos=torso_pos, torso_rot=torso_rot, torso_vel=torso_vel, torso_ang_vel=torso_ang_vel, 
                #     dof_pos=dof_pos, dof_vel=dof_vel, obs=self.rollout_buffer.observations, values=self.rollout_buffer.values, rewards=self.rollout_buffer.rewards, advs=self.rollout_buffer.advantages)

        

                # # log observations  NOTE: Debug only
                # num_observations = self.rollout_buffer.obs_shape # tuple
                # for i in range(num_observations[0]):
                #     self.logger.record("obs/{}_max".format(i), self.rollout_buffer.observations[:,:,i].max())
                #     self.logger.record("obs/{}_min".format(i), self.rollout_buffer.observations[:,:,i].min())


                self.logger.dump(step=self.num_timesteps)   # dump to log

                # NOTE: for debug only
                # save dones (buffer episode_start)
                # save rewards (rollout_buffer.rewards)
                # save observations (rollout_buffer.observations)
                if False: #iteration > 1205 and iteration < 1215:
                    np.savez_compressed("/data/zanming/Omniverse/OmniIsaacGymEnvs/omniisaacgymenvs/logs/SimplePPO_3a_test_10_1/debug/debug_{:06d}.npz".format(iteration), \
                        dones=self.rollout_buffer.episode_starts, rewards=self.rollout_buffer.rewards, obs=self.rollout_buffer.observations, values=self.rollout_buffer.values,\
                        advs=self.rollout_buffer.advantages, log_probs=self.rollout_buffer.log_probs, \
                        pi_weights_mean=[a.weight.mean().item() for a in self.policy.mlp_extractor.policy_net if isinstance(a, th.nn.Linear)], \
                        pi_weights_std=[a.weight.std().item() for a in self.policy.mlp_extractor.policy_net if isinstance(a, th.nn.Linear)], \
                        vf_weights_mean=[a.weight.mean().item() for a in self.policy.mlp_extractor.value_net if isinstance(a, th.nn.Linear)], \
                        vf_weights_std=[a.weight.std().item() for a in self.policy.mlp_extractor.value_net if isinstance(a, th.nn.Linear)], \
                        reward_alive = np.array([info_env['reward_alive'].detach().cpu().numpy() for info_step in info_list for info_env in info_step]),
                        reward_progress = np.array([info_env['reward_progress'].detach().cpu().numpy() for info_step in info_list for info_env in info_step]),
                        reward_up = np.array([info_env['reward_up'].detach().cpu().numpy() for info_step in info_list for info_env in info_step]),
                        reward_height = np.array([info_env['reward_height'].detach().cpu().numpy() for info_step in info_list for info_env in info_step]),
                        reward_actions = np.array([info_env['reward_actions'].detach().cpu().numpy() for info_step in info_list for info_env in info_step]),
                        reward_energy = np.array([info_env['reward_energy'].detach().cpu().numpy() for info_step in info_list for info_env in info_step]),
                        reward_dof = np.array([info_env['reward_dof'].detach().cpu().numpy() for info_step in info_list for info_env in info_step]),
                        reward_forward = np.array([info_env['reward_forward'].detach().cpu().numpy() for info_step in info_list for info_env in info_step]))

            self.train()

        callback.on_training_end()

        return self

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []
