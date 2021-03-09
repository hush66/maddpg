import numpy as np
import random
import tensorflow as tf
import maddpg.common.tf_util as U

from maddpg.common.distributions import make_pdtype
from maddpg import AgentTrainer
from maddpg.trainer.replay_buffer import ReplayBuffer

def baseline_calculation(a, b):
    assert len(a) == len(b)
    baseline = 0
    for i, j in zip(a, b):
        baseline += i * j
    return baseline

def p_train(make_obs_ph_n, act_space_n, p_index, p_func, q_func, optimizer, num_outputs,
            grad_norm_clipping=None, local_q_func=False, num_units=64, scope="coma_trainer", reuse=None):

    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        # act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]
        act_ph_n = [tf.placeholder(tf.int32, [None], name="action"+str(i)) for i in range(len(act_space_n))]

        # actor的输入为本地的obs
        p_input = obs_ph_n[p_index]

        p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="coma_p_func", num_units=num_units)
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))

        # wrap parameters in distribution
        act_pd = act_pdtype_n[p_index].pdfromflat(p)

        # 得到各个action的概率
        act_sample = act_pd.sample()
        # sample操作即gumble softmax  coma训练需要某个特定的动作,所以需要一个argmax操作
        act_picked = [act.tolist().index(max(act)) for act in act_sample]

        p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))

        # 为什么要加一个[]
        act_input_n = act_ph_n + []
        # 动作概率分布  替换当前agent的动作
        act_input_n[p_index] = act_picked
        q_input = tf.concat(obs_ph_n + act_input_n, 1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[p_index], act_input_n[p_index]], 1)
        q = q_func(q_input, num_outputs, scope="coma_q_func", reuse=True, num_units=num_units)

        # 反事实基线
        baseline = [baseline_calculation(act_distribute, q_list) for act_distribute, q_list in zip(act_sample, q)]
        # 根据真实采取的动作获得q
        actual_picked_q = [q_list[act] for act, q_list in zip(act_picked, q)]
        # 计算当前动作的q相对于反事实基线的差值
        a = [q - b for q, b in zip(actual_picked_q, baseline)]

        pg_loss = -tf.reduce_mean(a)

        loss = pg_loss + p_reg * 1e-3

        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n, outputs=loss, updates=[optimize_expr])
        act = U.function(inputs=[obs_ph_n[p_index]], outputs=act_sample)
        p_values = U.function([obs_ph_n[p_index]], p)

        # target network
        target_p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="coma_target_p_func", num_units=num_units)
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars)

        target_act_sample = act_pdtype_n[p_index].pdfromflat(target_p).sample()
        target_act = U.function(inputs=[obs_ph_n[p_index]], outputs=target_act_sample)

        return act, train, update_target_p, {'p_values': p_values, 'target_act': target_act}

def q_train(make_obs_ph_n, act_space_n, q_index, q_func, optimizer, grad_norm_clipping=None, scope="coma_trainer", reuse=None, num_units=64, num_outputs=1):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [tf.placeholder(tf.int32, [None], name="action"+str(i)) for i in range(len(act_space_n))]
        target_ph = [tf.placeholder(tf.float32, [None], name="target") for _ in range(len(act_space_n))]
        
        # 在一维进行拼接
        q_input = tf.concat(obs_ph_n + act_ph_n, 1)
        q = q_func(q_input, num_outputs, scope="coma_q_func", num_units=num_units)
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

        q_loss = tf.reduce_mean(tf.square(q - target_ph))

        # viscosity solution to Bellman differential equation in place of an initial condition
        q_reg = tf.reduce_mean(tf.square(q))
        loss = q_loss 

        optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n + [target_ph], outputs=loss, updates=[optimize_expr])
        q_values = U.function(obs_ph_n + act_ph_n, q)

        # target network
        target_q = q_func(q_input, num_outputs, scope="coma_target_q_func", num_units=num_units)
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars)

        target_q_values = U.function(obs_ph_n + act_ph_n, target_q)

        return train, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values}

class COMAAgentTrainer(AgentTrainer):
    def __init__(self, name, model, obs_shape_n, act_space_n, agent_index, action_number, args):
        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.args = args
        obs_ph_n = []
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation"+str(i)).get())

        # Create all the functions necessary to train the model
        self.q_train, self.q_update, self.q_debug = q_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            q_index=agent_index,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            num_units=args.num_units,
            num_outputs=action_number
        )
        self.act, self.p_train, self.p_update, self.p_debug = p_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            p_index=agent_index,
            p_func=model,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=False,
            num_units=args.num_units,
            num_outputs=action_number
        )
        # Create experience buffer
        self.replay_buffer = ReplayBuffer(1e6)
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None

    def get_inputs(self):
        pass

    def action(self, obs):
        return self.act(obs[None])[0]

    def experience(self, obs, act, rew, new_obs, done, terminal):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, agents, t):
        if len(self.replay_buffer) < self.max_replay_buffer_len: # replay buffer is not large enough
            return
        if not t % 100 == 0:  # only update every 100 steps
            return

        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)
        # collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []
        index = self.replay_sample_index
        # 每个agent获得replay batch
        for i in range(self.n):
            obs, act, rew, obs_next, done = agents[i].replay_buffer.sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_picked = [softmax_act.tolist().index(max(softmax_act)) for softmax_act in act]
            act_n.append(act_picked)
        # 当前trainer的replay batch
        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)

        # train q network
        # 向后看1步，相当于TD1？
        num_sample = 1
        target_q = 0.0
        for i in range(num_sample):
            # 获得下一step的所有agent动作，每个agent根据 本地 的下一step的obs做决策
            target_act_next_n = [agents[i].p_debug['target_act'](obs_next_n[i]) for i in range(self.n)]

            act_picked = []
            for i in range(self.n):
                act_picked += [softmax_act.tolist().index(max(softmax_act)) for softmax_act in target_act_next_n[i]]

            # 利用target网络得到target q值
            target_q_next = self.q_debug['target_q_values'](*(obs_next_n + act_picked))

            # Q network得到的是当前用户取不同动作的q，计算loss需要得到真实动作下的Q
            target_q_picked_next = [q[act] for act, q in zip(act_picked[self.agent_index], target_q_next)]

            target_q += rew + self.args.gamma * (1.0 - done) * target_q_picked_next
        target_q /= num_sample
        q_loss = self.q_train(*(obs_n + act_n + [target_q]))

        # train p network
        p_loss = self.p_train(*(obs_n + act_n))

        self.p_update()
        self.q_update()

        return [q_loss, p_loss, np.mean(target_q), np.mean(rew), np.mean(target_q_next), np.std(target_q)]
