import random
import math
import numpy as np
from environment.core import World, BranchyModel, Service, Agent, BaseStation
from environment.hyperParameters import RHO, DROP_PENALTY, ACCURACY_PENALTY

# branchy DNN model info
#COMP_INTENSITY = [97, 124, 139, 165]
COMP_INTENSITY = [100, 150, 200, 250]
ACC_TABLE = [0.59 , 0.68, 0.76, 0.78]
INPUT_SIZE = 1.1 * math.pow(10, 6)
# service info
MAX_WAIT_TIME = 1.5
ACC_LIMIT = 0.75
# agent number in the world
AGENT_NUMBER = 20
# computation ability's bound for agents  0.1GHz-0.5GHz
MAX_ABILITY = 0.2
MIN_ABILITY = 0.1
# base station's computation ability
BS_ABILITY = 2*math.pow(10, 9)


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
        exe_efficiency = agent.arrived_tasks / agent.latency
        # normalize exe_efficiency in [0, 5]
        exe_efficiency = exe_efficiency / 5
        # QoE of current agent
        acc_penalty = 0
        if agent.service.acc_limit > avg_accuracy:
            acc_penalty = 1
        qoe = agent.rho * avg_accuracy + (1 - agent.rho) * exe_efficiency - agent.is_dropped * DROP_PENALTY - acc_penalty * ACCURACY_PENALTY
        #print("accuracy: ", avg_accuracy, "latency: ", agent.latency/5, "agent", agent.is_offloaded(), "remain_task: ", agent.remain_task, " qoe:", qoe, "drop penalty: ", agent.is_dropped * DROP_PENALTY, "acc penalty: ", acc_penalty * ACCURACY_PENALTY)
        qoe_list.append(qoe)
    rwd = sum(qoe_list) + world.bs.utilization_rate * len(agents)
    #print("qoe list: ", qoe_list, "util: ", world.bs.utilization_rate)
    return rwd


def observation(agent: Agent, world: World, time_slot: int):
    service = agent.service
    # arrived bit size
    tasks_amount = agent.arrived_tasks * agent.service.branchy_model.input_data
    channel_gain = world.channel_gain[agent.channel_gain_id]
    # observation values are normalized to the same order of magnitude
    if time_slot == 0:
        return np.array([agent.remain_task / math.pow(10, 6), world.bs.remain_task / math.pow(10, 6), tasks_amount / math.pow(10, 6), channel_gain * math.pow(10, 13), 0])
    return np.array([agent.remain_task / math.pow(10, 6), world.bs.remain_task / math.pow(10, 6), tasks_amount / math.pow(10, 6), channel_gain * math.pow(10, 13), agent.acc_sum / (time_slot+1) * 100])

def information(agent: Agent):
    return agent.latency