import random
from environment.core import World, BranchyModel, Service, Agent, BaseStation

class BaseScenario:
    # create elements of the world
    def make_world(self):
        raise NotImplementedError()

    # create initial conditions of the world
    def reset_world(self):
        raise NotImplementedError()


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


class TaskOffloading(BaseScenario):
    def make_world(self):
        # branchy model
        branchy_model = BranchyModel(COMP_INTENSITY, ACC_TABLE, INPUT_SIZE)
        # TODO: a fixed service now
        service = Service(branchy_model, MAX_WAIT_TIME)
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
        self.reset_world(world)
        return world

    def reset_world(self, world):
        for agent in world.agents:
            agent.gain = 0
            agent.remain_task = 0
            agent.acc_sum = 0
            # TODO: 补充

    def reward(self, agent, world):
        # TODO: implement reward
        raise NotImplementedError()
