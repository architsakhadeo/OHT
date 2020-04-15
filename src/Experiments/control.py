import include_parent_folder
import numpy as np
from importlib import import_module
import torch
import time
import math
from utils.collect_config import ParameterConfig, Sweeper
from utils.collect_parser import CollectInput
from utils.log_and_plot_func import write_param_log
import os


def saved_file_name(config, run_idx, eval=False):
    learning = config.learning
    if learning == "offline" and eval:
        learning += "_eval"
    agent_params = config.agent_params
    input_ = agent_params.rep_type
    input_ = input_[0].upper() + input_[1:]
    if agent_params.rep_type in ["TC", "sepTC"]:
        input_ += "{}x{}".format(agent_params.num_tilings, agent_params.num_tiles)
    if config.agent == "DQN":
        other_info = "_B{}_sync{}_NN{}".format(agent_params.len_buffer,
                                             agent_params.dqn_sync,
                                             str(agent_params.nonLinearQ_node),
                                             )
    else:
        other_info = ""
    file_path = "{}/{}/{}_{}{}_alpha{}_input{}".format(
        agent_params.exp_result_path,
        learning,
        config.environment,
        config.agent,
        other_info,
        agent_params.alpha,
        input_
    )
    file_path += "/"
    file_name = "run" + str(run_idx)
    return file_path, file_name

class Experiment():
    def __init__(self, config, parsers):
        self.run_idx = parsers.run_idx
        run_seed = config.exp_params.random_seed * self.run_idx
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # define env
        self.domain = parsers.domain.lower()
        env_code = import_module("Environments.{}".format(config.environment))
        self.env_name = config.environment
        self.env = env_code.init_env()
        self.env.set_param(config.env_params)

        num_action = self.env.num_action()
        self.dim_state = self.env.state_dim()
        state_normalize = self.env.state_range()
        setattr(config.agent_params, "num_action", num_action)
        setattr(config.agent_params, "dim_state", self.dim_state)
        setattr(config.agent_params, "state_normalize", state_normalize)
        self.config = config

        # define agent
        self.agent_code = import_module("Agents.{}".format(config.agent))
        self.agent = self.agent_code.init_agent()
        self.agent.set_param(config.agent_params)
        self.gamma = config.agent_params.gamma

        self.reset_exp()

    def reset_exp(self):
        self.num_steps = self.config.exp_params.num_steps
        if hasattr(self.config.exp_params, "max_step_ep") and self.config.exp_params.max_step_ep != 0:
            self.max_step_ep = self.config.exp_params.max_step_ep
        else:
            self.max_step_ep = np.inf
        self.step_log = []
        self.reward_log = np.zeros(self.num_steps)
        self.trajectory_log = np.zeros((self.num_steps, self.dim_state*2+3))
        self.num_episode = self.config.exp_params.num_episodes
        self.control_ep = False if self.config.exp_params.num_episodes == 0 else self.config.exp_params.num_episodes
        self.learning = False
        self.count_ep = 0
        self.count_total_step = 0
        self.count_learning_step = 0
        self.old_count_learning_step = 0
        self.old_count_total_step = 0
        self.log_interval = 200
        self.old_time = time.time()

    def save_log(self, save_step=True, save_reward=True, save_traj=False, save_Q=False, eval=False):
        path, name = saved_file_name(self.config, self.run_idx, eval=eval)
        if not os.path.exists(path):
            os.makedirs(path)
        if save_step:
            np.save(path+name+"_stepPerEp", self.step_log)
        if save_reward:
            np.save(path+name+"_rewardPerStep", self.reward_log)
        if save_traj:
            np.save(path+name+"_trajectory", self.trajectory_log)
        if save_Q:
            self.agent.save(path+name+"_Q")

        if (save_step or save_reward) and (self.run_idx == 0):
            write_param_log(self.config.agent_params, self.config.env_params, self.config.environment, path,
                            exp_params=self.config.exp_params)
        print("File saved in", path)
        return path, name

    def online_learning(self):
        for pair in self.config.env_params.__dict__:
            space = " " * (20 - len(str(pair))) + ": "
            print(str(pair), space, str(self.config.env_params.__dict__[pair]))
        for run in range(1):
            self.reset_exp()
            self.single_run()
        self.config.agent_params = self.agent.get_settings()
        self.save_log()

    def offline_learning(self):
        # agent which will be evaluated
        eval_code = import_module("Agents.{}".format(self.config.agent))
        eval_name = self.config.agent

        path, name = self.collect_trajectory()
        q_path, q_name = self.learn_policy(path, name, eval_code, eval_name)
        self.evaluation(q_path, q_name, eval_code, eval_name, 100000)

    def collect_trajectory(self):
        # Collect trajectories with some policy
        self.config.agent = self.config.learn_from
        agent_code = import_module("Agents.{}".format(self.config.learn_from))
        self.agent = agent_code.init_agent()

        self.reset_exp()
        self.single_run()
        path, name = self.save_log(save_step=False, save_reward=False, save_traj=True, save_Q=False)
        return path, name

    def learn_policy(self, path, name, eval_code, eval_name):
        # Load saved trajectory
        simulated_env = np.load(path+name+"_trajectory.npy")
        st = simulated_env[:, :self.dim_state]
        at = simulated_env[:, self.dim_state]
        stp = simulated_env[:, self.dim_state + 1: self.dim_state * 2 + 1]
        reward = simulated_env[:, self.dim_state * 2 + 1]
        gamma = simulated_env[:, self.dim_state * 2 + 2]

        # Learn the policy
        agent_code = import_module("Agents.OfflineAgent")
        self.config.agent = eval_name
        self.config.agent_params.offline_data = at
        self.agent = agent_code.offline_agent(eval_name, self.config.agent_params)
        self.agent.set_param(self.config.agent_params)
        t = 0
        while t < len(simulated_env)-1:
            end = False
            self.agent.offline_start(st[t], t)
            while not end and t < len(simulated_env)-1:
                self.agent.offline_step(reward[t], st[t+1], gamma[t], t+1)
                if gamma[t] == 0:
                    end = True
                t += 1
                if t % 5000 == 0:
                    path, name = self.save_log(save_step=False, save_reward=False, save_traj=False, save_Q=True)
                    print("\nEvaluating agent at step", t)
                    learning_agent = self.agent
                    self.evaluation(path, name, eval_code, eval_name, 200)
                    self.agent = learning_agent
                    print("\nEvaluation ends, keep learning...")

        path, name = self.save_log(save_step=False, save_reward=False, save_traj=False, save_Q=True)
        return path, name

    def evaluation(self, path, name, eval_code, eval_name, eval_step):
        # Evaluate agent with learned policy
        self.agent_code = eval_code
        self.config.agent = eval_name
        self.config.exp_params.num_steps = eval_step # Evaluate for 500 steps
        learning_weight = self.config.agent_params.alpha
        learning_epsilon = self.config.agent_params.epsilon
        self.config.agent_params.alpha = 0 # Fixed weight
        self.config.agent_params.epsilon = 0
        self.config.agent_params.decreasing_epsilon = False # Does not use epsilon greedy in evaluation
        self.agent = eval_code.init_agent()
        self.agent.set_param(self.config.agent_params)
        self.agent.load(path+name+"_Q")

        self.reset_exp()
        self.single_run()
        self.config.agent_params.alpha = learning_weight
        self.config.agent_params.epsilon = learning_epsilon
        self.save_log(save_step=True, save_reward=True, save_traj=False, save_Q=False, eval=True)

    def single_run(self):
        print("Episode {} starts, total step {}, learning step {}/{}".format(self.count_ep, self.count_total_step,
                                                                          self.count_learning_step, self.num_steps))
        if self.control_ep:
            for _ in range(self.num_episode):
                self.single_ep()
                self.count_ep += 1
        else:
            condition = self.count_total_step
            while condition < self.num_steps:
                self.single_ep()
                self.count_ep += 1
                condition = self.count_total_step


    def single_ep(self):
        end_of_ep = False
        self.prev_state = self.env.start()
        self.prev_action = self.agent.start(self.prev_state)
        s_t = np.copy(self.prev_state)
        a_t = self.prev_action

        condition = self.count_total_step

        while (not end_of_ep) and \
                (condition < self.num_steps) and \
                (self.count_total_step - self.old_count_total_step < self.max_step_ep):

            step_info = self.env.step(self.prev_action)
            self.prev_state, reward, end_of_ep = step_info[:3]
            s_tp = np.copy(self.prev_state)
            gamma = 0 if end_of_ep else self.gamma
            self.prev_action, info = self.agent.step(reward, self.prev_state, end_of_ep)

            self.reward_log[self.count_total_step] = reward
            self.trajectory_log[self.count_total_step, :self.dim_state] = s_t
            self.trajectory_log[self.count_total_step, self.dim_state] = a_t
            self.trajectory_log[self.count_total_step, self.dim_state+1: self.dim_state*2+1] = s_tp
            self.trajectory_log[self.count_total_step, self.dim_state*2+1] = reward
            self.trajectory_log[self.count_total_step, self.dim_state*2+2] = gamma
            s_t = s_tp
            a_t = self.prev_action
            if self.learning == False and reward != 0:
                self.learning = True

            if self.learning:
                self.count_learning_step += 1
            self.count_total_step += 1

            if self.count_total_step % self.log_interval == 0:
                print("Total step {}, learning step {}/{}, time={}".format(
                    self.count_total_step, self.count_learning_step,
                    self.num_steps, time.time() - self.old_time))
                self.old_time = time.time()
                # print("check q value", self.prev_state, self.agent.check_q_value())

            condition = self.count_total_step

        self.step_log.append(self.count_total_step - self.old_count_total_step)

        if end_of_ep:
            ep_return = np.sum(np.array(self.reward_log[self.old_count_total_step: self.count_total_step]))
            print("\t\tEpisode {} ends, ep total step {}, learning step {}/{}, ep return {}, accumulate reward {}\n".format(
                self.count_ep, self.step_log[-1], self.count_learning_step, self.num_steps,
                ep_return,
                np.sum(np.array(self.reward_log))))

        self.old_count_total_step = self.count_total_step
        self.old_count_learning_step = self.count_learning_step
        return



if __name__ == '__main__':

    # parse argument
    ci = CollectInput()
    parsers = ci.control_experiment_input()
    env_name = parsers.domain.lower()
    sweeper = Sweeper('../Parameters/{}.json'.format(env_name.lower()), "control_param")
    config = sweeper.parse(parsers.sweeper_idx)

    oml = Experiment(config, parsers)
    if config.learning == "online":
        oml.online_learning()
    else:
        oml.offline_learning()
