import random
import math
import numpy as np
from environment.core import World, BranchyModel, Service, Agent, BaseStation
from environment.hyperParameters import RHO

# branchy DNN model info
#[97, 124, 139, 165]
COMP_INTENSITY = [150, 200, 250, 300]
ACC_TABLE = [0.59 , 0.68, 0.76, 0.78]
INPUT_SIZE = 1.1 * math.pow(10, 6)
# service info
MAX_WAIT_TIME = 1.5
ACC_LIMIT = 0.75
# agent number in the world
AGENT_NUMBER = 5
# computation ability's bound for agents  0.1GHz-0.5GHz
MAX_ABILITY = 0.1
MIN_ABILITY = 0.3
# base station's computation ability
BS_ABILITY = 2*math.pow(10, 9)
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
        comp_ability = (MIN_ABILITY + (MAX_ABILITY - MIN_ABILITY) * random.random()) * math.pow(10, 9)
        agent = Agent(comp_ability, service)
        agent.name = 'IoT device %d' % i
        agents.append(agent)
    # create base station
    bs = BaseStation(BS_ABILITY, service)
    # create world
    world = World(agents, bs, TIME_SLOT_DURATION)
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
        exe_efficiency = agent.arrived_tasks / agent.latency
        # normalize exe_efficiency in [0, 5]
        exe_efficiency = exe_efficiency / 5
        # QoE of current agent
        #print("accuracy: ", avg_accuracy, "latency: ", agent.latency/5, "agent", agent.is_offloaded(), "remain_task: ", agent.remain_task)
        qoe = agent.rho * avg_accuracy + (1 - agent.rho) * exe_efficiency
        qoe_list.append(qoe)
    return sum(qoe_list)


def observation(agent: Agent, world: World):
    service = agent.service
    # arrived bit size
    tasks_amount = agent.arrived_tasks * service.branchy_model.input_data
    channel_gain = world.channel_gain[agent.channel_gain_id]
    return np.array([agent.remain_task, world.bs.remain_task, tasks_amount, channel_gain])
