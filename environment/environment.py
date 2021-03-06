import gym
import time
import numpy as np
import tensorflow as tf
from gym import spaces


class MultiAgentEnv(gym.Env):
    def __init__(self, world, reset_func, reward_func=None, observation_func=None, info_func=None, done_func=None):
        self.world = world
        self.agents = world.agents
        # set required vectorized gym env property
        self.n = len(self.agents)
        # Number of optional actions TODO: Assume that the number of optional actions for all services is equal currently
        self.action_number = self.agents[0].service.branchy_model.branches_num + 1
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
            self.action_space.append(spaces.Discrete(self.action_number))
            # observation space
            obs_dim = len(observation_func(agent, self.world, self.time_slot))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim, ), dtype=np.float32))

    def step(self, action_n):
        obs_n = []
        done_n = []
        info_n = []
        # set action for each agent
        for i, agent in enumerate(self.agents):
            action_list = action_n[i].tolist()
            agent.action = action_list.index(max(action_list))
        # update world state
        self.world.step()
        self.time_slot += 1
        # record observation for each agent
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            info_n.append(self._get_info(agent))
            done_n.append(self._get_done(agent))
        # get reward
        reward, qoe_list = self._get_reward()
        reward_n = [reward] * self.n  # all agents get total reward in cooperative case
        return obs_n, reward_n, done_n, info_n, qoe_list

    def reset(self):
        for agent in self.agents:
            if agent.action is not None:
                print(agent.name, " latency: ", agent.latency, " action: ", agent.action, " accuracy: ", agent.acc_sum/self.time_slot, " remain_task: ", agent.remain_task)
        self.time_slot = 0
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
        return self.observation_func(agent, self.world, self.time_slot)

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
