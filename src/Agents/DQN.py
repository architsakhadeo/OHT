#!/usr/bin/python3
import numpy as np
import pickle as pkl
from importlib import import_module
import torch
from torch import nn
from Agents.BaseAgent import BaseAgent
import utils.tiles3 as tc

np.set_printoptions(precision=3)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# non-linear Q
class NonLinearVF(torch.nn.Module):
    def __init__(self, dims):
        super(NonLinearVF, self).__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(torch.nn.Linear(dims[i], dims[i+1]))
            nn.init.kaiming_normal_(layers[-1].weight)
            if i != len(dims) - 2:
                layers.append(getattr(nn, "ReLU")())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        return x

# class LinearVF(torch.nn.Module):
#     def __init__(self, dims):
#         super(LinearVF, self).__init__()
#         layers = []
#         for i in range(len(dims) - 1):
#             layers.append(torch.nn.Linear(dims[i], dims[i+1]))
#             nn.init.kaiming_normal_(layers[-1].weight)
#         self.net = nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.net(x)
#         return x


class BufferControl():
    def __init__(self, length):
        self.b_length = length
        self.b_empty = [i for i in range(self.b_length)]
        self.b_filled = []
        self.b_filled_length = None

    def insert(self):
        if len(self.b_empty) == 0:
            index = self.b_filled[0]
            self.b_filled = self.b_filled[1:]
            self.b_filled.append(index)
        else:
            index = self.b_empty[0]
            if index not in self.b_filled:
                self.b_filled.append(index)
            self.b_empty = self.b_empty[1:]

        self.b_filled_length = len(self.b_filled)

        return index

    def remove(self, index):
        self.b_empty.append(index)

    def force_remove(self, index):
        self.b_filled.remove(index)
        self.b_empty.append(index)
        self.b_filled_length = len(self.b_filled)

    def get_filled(self):
        return self.b_filled

    def get_filled_length(self):
        return self.b_filled_length

    def get_empty(self):
        return self.b_empty

class DQN(BaseAgent):

    # Default values
    def __init__(self):
        super().__init__()
        return

    def set_param(self, param):
        # self.learning = False # Before getting the first non-zero reward, the agent chooses random action
        self.learning = True # does not affect learning
        self._sample_seqs_from_buffer = self._sample_seqs_from_buffer_random # random buffer

        self.alpha = param.alpha
        self.epsilon = param.epsilon
        self.decreasing_epsilon = param.decreasing_epsilon
        self.gamma = param.gamma

        self.num_planning = param.num_planning

        self.num_action = param.num_action
        self.action_list = [i for i in range(self.num_action)]

        self.dim_observation = param.dim_state # observation dimension

        # non-linear function
        self._get_tde = self._get_tde_nonLinear
        self._vf_weight_update = self._nonLinear_q_update
        self._policy = self._policy_nonLinear
        self._single_planning = self._single_planning_nonLinear_ER

        self.minibatch = param.dqn_minibatch
        self.nlq_sync = param.dqn_sync
        self.nlq_count = 0

        self.param = param
        self.state_normalize = self.param.state_normalize

        if self.param.rep_type == "sepTC":
            self.div_actBit = self.param.num_tilings * self.dim_observation
            self.alpha = param.alpha / float(self.div_actBit)
            self.num_tiling = self.param.num_tilings
            self.num_tile = self.param.num_tiles
            self.tc_mem_size = self.param.tc_mem_size
            self.iht = tc.IHT(self.tc_mem_size)
            self.dim_state = self.tc_mem_size * self.dim_observation  # representation dimension
            param.dim_state = self.dim_state
            self._state_representation = self._separate_tc_rep
        elif self.param.rep_type == "obs":
            self.dim_state = self.dim_observation
            self._state_representation = self._obs_normalization

        node = [self.dim_state] + param.nonLinearQ_node + [self.num_action]
        self.nlq_learn = NonLinearVF(node).to(device)
        self.nlq_target = NonLinearVF(node).to(device)
        # print(self.nlq_learn)
        # print(self.nlq_target)
        self.nlq_learn_optimizer = torch.optim.Adam(self.nlq_learn.parameters(), lr=self.alpha, betas=(self.param.dqn_beta[0], self.param.dqn_beta[1]))
        self.nlq_loss = torch.nn.MSELoss()

        self.len_buffer = param.len_buffer
        self.buffer = np.zeros((self.len_buffer, self.dim_state * 2 + 4))
        self.b_control = BufferControl(self.len_buffer)
        self.pri_thrshd = param.pri_thrshd # Not used in random buffer

        return

    def save(self, file_name):
        torch.save(self.nlq_learn.state_dict(), file_name + ".pth")
        print("model saved in " + file_name + ".pth")

    def load(self, file_name):
        self.nlq_learn.load_state_dict(torch.load(file_name + ".pth"))
        self.nlq_target.load_state_dict(torch.load(file_name + ".pth"))
        print("Load trained weight from", file_name)

    def check_q_value(self):
        with torch.no_grad():
            state = torch.from_numpy(self.state).float().to(device)
            sp_values = self.nlq_target(state)
            return sp_values.data

    """
    Input: state
    Return: action
    """
    def start(self, state):
        state = self._state_representation(state)
        self.state = state
        self.action = self._policy(state)

        # change epsilon based on number of episodes
        if self.decreasing_epsilon == "ep":
            self.epsilon = max(self.epsilon - 0.05, self.param.epsilon)
            # print("Change epsilon per episode:", self.epsilon)

        return self.action

    """
    Input: int, state
    Return: action
    """
    def step(self, reward, state, end_of_ep=False):
        # change epsilon based on number of steps
        if self.decreasing_epsilon == "step":
            self.epsilon = max(self.epsilon - 1.0/10000, self.param.epsilon)
            # print("Change epsilon per step:", self.epsilon)

        if end_of_ep:
            gamma = 0
        else:
            gamma = self.gamma

        if not self.learning:
            if reward != 0:
                self.learning = True

        state = self._state_representation(state)

        # update variables
        self.last_state = self.state
        self.last_action = self.action
        self.state = state
        self.reward = reward

        # get tde
        tde = self._get_tde(self.last_state, self.last_action, self.state, reward, gamma)

        # update buffer
        self._buffer_insert_seq(self.last_state, self.last_action, self.state, reward, gamma,
                                np.abs(tde) + self.pri_thrshd)

        for _ in range(self.num_planning):
            self._single_planning()

        # choose new action
        self.action = self._policy(self.state)

        other_info = None
        return self.action, other_info

    """
    Input: int, state
    Return: None
    """
    # def end(self, reward):
    #     placeholder = self.state
    #     self.step(reward, placeholder, end_of_ep=True)
    #     return

    def _format_data(self, s, a, sp, r, gamma):
        s, a, sp, r, gamma, _ = self._array_to_seq(self._seq_to_array(s, a, sp, r, gamma, 0).reshape((1, -1)))
        return s, a, sp, r, gamma

    def _nonLinear_q_update(self, s, a, sp, r, gamma):
        if len(s.shape) == 1:
            s, a, sp, r, gamma = self._format_data(s, a, sp, r, gamma)

        s = torch.from_numpy(s).float().to(device)
        a = torch.from_numpy(a.reshape(-1, 1)).float().type(torch.LongTensor).to(device)
        s_values = self.nlq_learn(s)
        prediction = s_values.gather(1, a).view((-1))

        with torch.no_grad():
            gamma = torch.from_numpy(gamma).float().to(device)
            r = torch.from_numpy(r).float().to(device)
            sp = torch.from_numpy(sp).float().to(device)
            sp_values = self.nlq_target(sp)

            # vanilla DQN
            sp_q_value = sp_values.data.max(1)[0]

            # # double Q-learning DQN
            # sp_values_learn = self.dqn_learn(sp)
            # ap = sp_values_learn.Data.max(1)[1].view((-1, 1))
            # sp_q_value = sp_values.gather(1, ap).view((-1))

        y = r + gamma * sp_q_value
        loss = self.nlq_loss(prediction, y)

        self.nlq_learn_optimizer.zero_grad()
        loss.backward()
        self.nlq_learn_optimizer.step()

        del s, a, s_values, prediction, gamma, r, sp, sp_values, sp_q_value, y, loss

        # DQN block
        self.nlq_count += 1
        if self.nlq_count % self.nlq_sync == 0:
            self._synchronize_networks()
            # print("DQN sync")


    def _synchronize_networks(self):
        params_from = self.nlq_learn.named_parameters()
        params_to = self.nlq_target.named_parameters()

        dict_params_to = dict(params_to)

        for name, param in params_from:
            if name in dict_params_to:
                dict_params_to[name].data.copy_(param.data)
            del name, param
        del params_from, params_to


    """
    Insert sample into buffer
    Input: state-last, action-last, state, reward, gamma, TD-error
    Return: None
    """
    def _buffer_insert_seq(self, last_state, last_action, state, reward, gamma, tde):
        new_sequence = self._seq_to_array(last_state, last_action, state, reward, gamma, tde)
        index = self.b_control.insert()
        self.buffer[index] = new_sequence
        return

    """
    Update sample's priority in buffer
    Input: index, state-last, action-last, state, reward, gamma
    Return: None
    """
    # def _update_priority(self, index, last_state, last_action, state, reward, gamma):
    #     new_pri = np.abs(self._get_tde(last_state, last_action, state, reward, gamma))
    #     self.buffer[index, -1] = new_pri
    #     if self.adpt_thrshd:
    #         self.pri_thrshd = np.mean(self.buffer[self.b_control.get_filled()][:, -1])
    #     return

    """
    Choose action according to given policy
    Input: state
    Return: action
    """
    # def _policy_linear(self, state):
    #     if np.random.random() < self.epsilon or (not self.learning):
    #         return np.random.choice(self.action_list)
    #     else:
    #         return self._max_action(state)

    def _policy_nonLinear(self, state):
        if np.random.random() < self.epsilon or (not self.learning):
            return np.random.choice(self.action_list)
        else:
            with torch.no_grad():
                state = torch.from_numpy(np.array(state)).float().to(device)
                q_values = self.nlq_learn(state).detach().cpu().numpy()
            action = self._break_tie(q_values)
            del state, q_values
            return action

    """
    Choose the optimal action, linear case
    Input: state
    Return: optimal action
    """
    # def _max_action(self, state):
    #     all_choices = []
    #     for a in self.action_list:
    #         feature = self._TC_feature_construction(state, a)
    #         # all_choices.append(np.dot(self.weight.weight.data, feature))
    #         # all_choices.append(np.dot(self.weight, feature))
    #         all_choices.append(np.sum(self.weight[feature]))
    #     valid_index = self._break_tie(all_choices)
    #     return valid_index

    """
    Break tie fairly
    Input: qvalue
    Return: optimal action
    """
    def _break_tie(self, xarray):
        max_v = np.max(xarray)
        valid_choices = np.where(xarray == max_v)[0]
        try:
            return np.random.choice(valid_choices)
        except:
            print(valid_choices)
            print(self.weight)
            print(xarray)
            raise RuntimeError("Error: Break tie")

    """
    Calculate TD error with linear tile coding
    Input: feature-last, feature, reward, gamma, weight
    Return: TD-error
    """
    # def _linear_td_error(self, last_feature, feature, reward, gamma, weight):
    #     # tde = reward + gamma * np.dot(feature, weight.weight.data.reshape(-1)) \
    #     #       - np.dot(last_feature, weight.weight.data.reshape(-1))
    #     # tde = reward + gamma * np.dot(feature, weight) - np.dot(last_feature, weight)
    #     tde = reward + gamma * np.sum(weight[feature]) - np.sum(weight[last_feature])
    #     return tde

    """
    Calculate TD error given state (x,y) or representation
    Input: state-last, action-last, state-current, reward, gamma
    Return: TD-error
    """
    # def _get_tde_linear(self, last_state, last_action, state, reward, gamma):
    #     last_feature = self._TC_feature_construction(last_state, last_action)
    #     feature = self._TC_feature_construction(state, self._max_action(state))
    #     tde = self._linear_td_error(last_feature, feature, reward, gamma, self.weight)
    #     return tde

    def _get_tde_nonLinear(self, last_state, last_action, state, reward, gamma):
        last_state = last_state.reshape((-1, self.dim_state))
        with torch.no_grad():
            last_state = torch.from_numpy(last_state).float().to(device)
            last_action = torch.from_numpy(last_action.reshape(-1, 1)).float().type(torch.LongTensor).to(device)
            s_values = self.nlq_learn(last_state)
            prediction = s_values.gather(1, last_action).view((-1)).detach().cpu().numpy()
            if gamma != 0:
                state = state.reshape((-1, self.dim_state))
                state = torch.from_numpy(state).float().to(device)
                sp_values = self.nlq_target(state)
                sp_q_value = sp_values.data.max(1)[0].detach().cpu().numpy()
                y = reward + gamma * sp_q_value
                del sp_values, sp_q_value
            else:
                y = reward
            tde = prediction - y
            tde = tde[0] if len(tde) == 1 else tde
        del last_state, last_action, s_values, prediction, state
        return tde

    """
    Feature vector using separate tile coding
    Input: observation
    Output: feature vector
    """
    def _separate_tc_rep(self, state):
        rep = np.zeros(self.dim_state)
        for i in range(self.dim_observation):
            normalized_bit = (np.array([state[i]]) / self.state_normalize[i] + 1) / 2
            assert 1>= normalized_bit >= 0, normalized_bit
            ind = np.array(tc.tiles(self.iht, self.num_tiling, float(self.num_tile) * normalized_bit))
            rep[ind + self.tc_mem_size * i] = 1
        # rep = self.obs_to_rep.state_representation(np.array(state))
        return rep

    """
    Normalize input to range: [-1, 1]
    Input: observation
    Output: normalized observation
    """
    def _obs_normalization(self, state):
        normalized = state / self.state_normalize
        return normalized


    """
    Planning step
    Input: lr, number of planning, index in buffer, sasprg-array
    Return: dictionary
    """
    # def _single_planning_linear_ER(self):
    #     indexs, seqs = self._sample_seqs_from_buffer(1)
    #     s, a, sp, r, gamma, _ = self._array_to_seq(seqs)
    #     self._vf_weight_update(s, a, sp, r, gamma)
    #     return True

    def _single_planning_nonLinear_ER(self):
        indexs, seqs = self._sample_seqs_from_buffer(self.minibatch)  # (self.num_planning):
        s, a, sp, r, gamma, _ = self._array_to_seq(seqs)
        self._vf_weight_update(s, a, sp, r, gamma)

    """
    Choose sequence from buffer
    For now we use elif block
    Input: number of plannings
    Return: index in buffer, sasprg-array
    """
    def _sample_seqs_from_buffer_random(self, n):
        filled_ind = self.b_control.get_filled()
        filled_ind_length = self.b_control.get_filled_length()
        indexs = np.random.choice(filled_ind, size=min(filled_ind_length, n))
        return self._buffer_return_samples(indexs)

    # def _sample_seqs_from_buffer_prioity(self, n):
    #     filled_ind = self.b_control.get_filled()
    #     filled_ind_length = self.b_control.get_filled_length()
    #     indexs = self._sample_break_tie(self.buffer[filled_ind, -1], min(filled_ind_length, n))
    #     return self._buffer_return_samples(indexs)
    #     # return self._buffer_return_samples(filled_ind, indexs)

    def _buffer_return_samples(self, indexs):
        if len(indexs) == 0:
            return [], []
        else:
            seqs = np.copy(self.buffer[indexs])
            return indexs, seqs

    """
    Choose samples with highest priority
    """
    def _sample_break_tie(self, pris, num):
        indexs = []
        if num > 1:
            pris_copy = np.copy(pris)
            for i in range(num):
                indexs.append(self._break_tie(pris_copy))
                pris_copy[indexs[i]] = -1000000
        else:
            indexs.append(self._break_tie(pris))
        return np.array(indexs)

    """
    Save sample in an array
    Input: [x, y]-last, action-last, [x, y], reward, gamma, TD-error
    Return: sasprg-array
    """
    def _seq_to_array(self, last_state, last_action, state, reward, gamma, tde):
        return np.concatenate((last_state, np.array([last_action]), state,
                               np.array([reward]), np.array([gamma]), np.array([tde])),
                              axis=0)
    """
    Get sample from array
    Input: sasprg-array
    Return: state-last, action-last, state-current, reward, gamma, TD-error
    """
    def _array_to_seq(self, seq):
        if seq.ndim == 1:
            last_state = seq[:self.dim_state]
            last_action = seq[self.dim_state]
            state = seq[self.dim_state+1: self.dim_state*2+1]
            reward = seq[self.dim_state*2+1]
            gamma = seq[self.dim_state*2+2]
            tde = seq[self.dim_state*2+3]
        else:
            last_state = seq[:, :self.dim_state]
            last_action = seq[:, self.dim_state]
            state = seq[:, self.dim_state + 1: self.dim_state * 2 + 1]
            reward = seq[:, self.dim_state * 2 + 1]
            gamma = seq[:, self.dim_state * 2 + 2]
            tde = seq[:, self.dim_state * 2 + 3]
        return last_state, last_action, state, reward, gamma, tde


def init_agent():
    agent = DQN()
    return agent
