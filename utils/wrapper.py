import gym
import torch
from vec_env.base_vec_env import VecEnvWrapper
from collections import deque


class DoneOnSuccessWrapper(gym.Wrapper):
    """
    Reset on success and offsets the reward.
    Useful for GoalEnv.
    """
    def __init__(self, env, reward_offset=1.0):
        super(DoneOnSuccessWrapper, self).__init__(env)
        self.reward_offset = reward_offset

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        done = done or info.get('is_success', False)
        reward += self.reward_offset
        return obs, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        reward = self.env.compute_reward(achieved_goal, desired_goal, info)
        return reward + self.reward_offset


class SwitchGoalWrapper(gym.Wrapper):
    def __init__(self, env):
        super(SwitchGoalWrapper, self).__init__(env)
        if not hasattr(self.env.unwrapped, "goals"):
            assert hasattr(self.env.unwrapped, "goal")
            self.env.unwrapped.goals = []

    def set_goals(self, goals):
        if isinstance(goals, tuple):
            goals = list(goals)
        assert isinstance(goals, list)
        self.env.unwrapped.goals = goals
        self.env.unwrapped.goal = self.env.unwrapped.goals.pop(0).copy()
        self.env.unwrapped.visualize_goal(self.env.unwrapped.goal)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if done:
            if len(self.env.goals) > 0:
                if info.get("TimeLimit.truncated", False):
                    info["is_success"] = False
                else:
                    self.env.unwrapped.goal = self.env.unwrapped.goals.pop(0).copy()
                    self.env.unwrapped.visualize_goal(self.env.unwrapped.goal)
                    done = False
                    info["is_success"] = False
        return obs, reward, done, info


class ResetWrapper(gym.Wrapper):
    # applied before SwitchGoalWrapper and FlattenDictWrapper
    def __init__(self, env, ratio=0.5):
        super(ResetWrapper, self).__init__(env)
        self.states_to_restart = deque(maxlen=50)  # dicts of state and goal
        self.ratio = ratio
        self.is_restart = False

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self.is_restart = False
        if len(self.states_to_restart) > 0 and self.env.unwrapped.np_random.uniform() < self.ratio:
            state_to_restart = self.states_to_restart.popleft()
            self.env.unwrapped.set_state(state_to_restart["state"])
            self.env.unwrapped.goal = state_to_restart["goal"]
            self.env.unwrapped.sync_attr()
            result = self.env.get_obs()
            self.is_restart = True
        return result

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info["is_restart"] = self.is_restart
        return obs, reward, done, info

    def add_state_to_reset(self, state_and_goal: dict):
        self.states_to_restart.append(state_and_goal)


class ScaleRewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_scale=1.0):
        super(ScaleRewardWrapper, self).__init__(env)
        self.reward_scale = reward_scale

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward /= self.reward_scale
        return obs, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        reward = self.env.compute_reward(achieved_goal, desired_goal, info)
        return reward / self.reward_scale


class FlexibleTimeLimitWrapper(gym.Wrapper):
    '''
    ONLY applicable to Stacking environment!
    We can set max_episode_steps = None for gym, (so gym.TimeLimitWrapper is not applied),
    then use this class to avoid potential conflict.
    '''
    def __init__(self, env):
        super(FlexibleTimeLimitWrapper, self).__init__(env)
        # self.time_limit = time_limit
        assert 'BulletStack' in env.spec.id
        assert env.spec.max_episode_steps is None
        self._elapsed_steps = None

    def step(self, action):
        assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"
        time_limit = self.env.unwrapped.n_to_stack * 50 if self.env.unwrapped.n_to_stack > 2 else 100
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= time_limit:
            info['TimeLimit.truncated'] = not done
            done = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        # reward = np.expand_dims(reward, axis=1).astype(np.float32)
        return obs, reward, done, info
