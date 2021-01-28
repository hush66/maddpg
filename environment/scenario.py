import random
import numpy as np
from environment.core import World, BranchyModel, Service, Agent, BaseStation
from environment.hyperParameters import GAMMA, RHO

# TODO: temp value
# branchy DNN model info
COMP_INTENSITY = [1, 1, 1, 1]
ACC_TABLE = [0.9, 0.8, 0.7, 0.6]
INPUT_SIZE = 100
# service info
MAX_WAIT_TIME = 1
ACC_LIMIT = 85
# agent number in the world
AGENT_NUMBER = 50
# computation ability's bound for agents
MAX_ABILITY = 10000
MIN_ABILITY = 10
# base station's computation ability
BS_ABILITY = 100000
# world info
TIME_SLOT_DURATION = 1


def make_world():
    # branchy model
    branchy_model = BranchyModel(COMP_INTENSITY, ACC_TABLE, INPUT_SIZE)
    # TODO: a fixed service now
    service = Service("service 1", branchy_model, MAX_WAIT_TIME, ACC_LIMIT)
    # add agents
    agents = []
    for i in range(AGENT_NUMBER):
        comp_ability = random.randint(MIN_ABILITY, MAX_ABILITY)
        agent = Agent(comp_ability, service)
        agent.name = 'IoT device %d' % i
        agents.append(agent)
    # create base station
    bs = BaseStation(BS_ABILITY, service)
    # create world
    world = World(agents, bs)
    # initialize world
    reset_world(world)
    return world


def reset_world(world: World):
    for agent in world.agents:
        agent.reset()
    world.bs.reset()
    world.reward = 0
    world.update_agents_states()


def reward(world: World, cur_time_slot: int):
    agents = world.agents
    qoe_list = []
    for agent in agents:
        avg_accuracy = agent.acc_sum / (cur_time_slot + 1)
        exe_efficiency = agent.service.branchy_model.input_data / agent.latency
        # QoE of current agent
        qoe = agent.rho * avg_accuracy + (1 - agent.rho) * exe_efficiency
        qoe_list.append(qoe)
    world.reward = world.reward * GAMMA + sum(qoe_list)


def observation(agent: Agent, world: World):
    service = agent.service
    # arrived bit size
    tasks_amount = agent.arrived_tasks * service.branchy_model.input_data
    channel_gain = world.channel_gain[agent.channel_gain_id]
    return np.concatenate([agent.remain_task, world.bs.remain_task, tasks_amount, channel_gain])
