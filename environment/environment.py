import gym
import numpy as np
from gym import spaces


class MultiAgentEnv(gym.Env):
    def __init__(self, world, reset_func, reward_func=None, observation_func=None, info_func=None, done_func=None):
        self.world = world
        self.agents = world.agents
        # set required vectorized gym env property
        self.n = len(self.agents)
        # scenario functions
        self.reset_func = reset_func
        self.reward_func = reward_func
        self.observation_func = observation_func
        self.info_func = info_func
        self.done_func =done_func

        self.time_slot = 0

        # configure  spaces
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            # action space
            self.action_space.append(spaces.Discrete(agent.service.branchy_model.branches_num + 1))
            # observation space
            obs_dim = len(observation_func(agent, self.world))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim, ), dtype=np.float32))

    def step(self, action_n):
        obs_n = []
        done_n = []
        info_n = {'n': []}
        # set action for each agent
        for i, agent in enumerate(self.agents):
            agent.action = action_n[i]
        # update world state
        self.world.step()
        # record observation for each agent
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            info_n['n'].append(self._get_info(agent))
            done_n.append(self._get_done(agent))
        # get reward
        reward = self._get_reward()
        reward_n = [reward] * self.n  # all agents get total reward in cooperative case
        self.time_slot += 1
        return obs_n, reward_n, done_n, info_n

    def reset(self):
        # reset world
        self.reset_func(self.world)
        # record initial observations for each agent
        obs_n = []
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        return obs_n

    def _get_obs(self, agent):
        if self.observation_func is None:
            raise np.zeros(0)
        return self.observation_func(agent, self.world)

    def _get_reward(self):
        if self.reward_func is None:
            raise 0.0
        return self.reward_func(self.world, self.time_slot)

    def _get_info(self, agent):
        if self.info_func is None:
            return {}
        return self.info_func(agent)

    def _get_done(self, agent):
        if self.done_func is None:
            return False
        return self.done_func(agent, self.world)