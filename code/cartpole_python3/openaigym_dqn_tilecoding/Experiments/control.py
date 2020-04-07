import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import numpy as np
from importlib import import_module
import torch
import time
import math
from utils.collect_config import ParameterConfig, Sweeper
from utils.collect_parser import CollectInput
from utils.log_and_plot_func import write_param_log
import os


def saved_file_name(config, run_idx):
    agent_params = config.agent_params
    file_path = "{}/{}_{}_alpha{}".format(
        agent_params.exp_result_path,
        config.environment,
        config.agent,
        agent_params.alpha,
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
        # env_code = import_module(ENV_NAME[self.domain])
        env_code = import_module("Environments.{}".format(config.environment))
        self.env_name = config.environment
        self.env = env_code.init_env(run_seed)

        # self.env.set_param(config.env_params)

        num_action = 2 #self.env.num_action()
        self.dim_state = 4 #self.env.state_dim()
        state_normalize = [4.8, 8.0, 2*12*2*math.pi/360.0, 7.0] #self.env.state_range()
        setattr(config.agent_params, "num_action", num_action)
        setattr(config.agent_params, "dim_state", self.dim_state)
        setattr(config.agent_params, "state_normalize", state_normalize)
        self.config = config

        # define agent
        agent_code = import_module("Agents.{}".format(config.agent))
        self.agent = agent_code.init_agent()
        self.agent.set_param(config.agent_params)
        self.gamma = config.agent_params.gamma

        self.num_steps = config.exp_params.num_steps
        if hasattr(config.exp_params, "max_step_ep") and config.exp_params.max_step_ep != 0:
            self.max_step_ep = config.exp_params.max_step_ep
        else:
            self.max_step_ep = np.inf
        self.step_log = []
        self.reward_log = np.zeros(self.num_steps)
        self.trajectory_log = np.zeros((self.num_steps, self.dim_state*2+3))
        self.num_episode = config.exp_params.num_episodes
        self.control_ep = False if config.exp_params.num_episodes == 0 else config.exp_params.num_episodes
        self.learning = None
        self.count_ep = None
        self.count_total_step = None
        self.count_learning_step = None
        self.old_count_learning_step = None
        self.log_interval = 200

    def save_log(self):
        path, name = saved_file_name(self.config, self.run_idx)
        if not os.path.exists(path):
            os.makedirs(path)

        np.save(path+name+"_stepPerEp", self.step_log)
        np.save(path+name+"_rewardPerStep", self.reward_log)
        # np.save(path+name+"_trajectory", self.trajectory_log)
        # self.agent.save(path+name+"_Q")

        if self.run_idx == 0:
            write_param_log(self.config.agent_params, self.config.env_params, self.config.environment, path,
                            exp_params=self.config.exp_params)
        print("File saved in", path)

    def run_exp(self):
        for pair in self.config.env_params.__dict__:
            space = " " * (20 - len(str(pair))) + ": "
            print(str(pair), space, str(self.config.env_params.__dict__[pair]))
        for run in range(1):
            self.learning = False
            self.count_ep = 0
            self.count_total_step = 0
            self.count_learning_step = 0
            self.old_count_learning_step = 0
            self.old_count_total_step = 0
            self.old_time = time.time()
            self.single_run()
        self.config.agent_params = self.agent.get_settings()
        self.save_log()

    def single_run(self):
        print("Episode {} starts, total step {}, learning step {}/{}".format(self.count_ep, self.count_total_step,
                                                                          self.count_learning_step, self.num_steps))
        # print("Episode {} starts, total step {}/{}".format(self.count_ep, self.count_total_step, self.num_steps))
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
        self.env.init()
        self.prev_state = self.env.start()
        self.prev_action = self.agent.start(self.prev_state)
        s_t = np.copy(self.prev_state)
        a_t = self.prev_action

        condition = self.count_total_step

        # change epsilon based on number of episodes
        if self.config.agent_params.decreasing_epsilon:
            self.agent.epsilon = max(1.0 - 1.0 / (self.num_episode / 200) * self.count_ep,
                                     self.config.agent_params.epsilon)
            print("Changed epsilon:", self.agent.epsilon)

        # while (not end_of_ep) and (self.count_learning_step < self.num_steps):
        while (not end_of_ep) and \
                (condition < self.num_steps) and \
                (self.count_total_step - self.old_count_total_step < self.max_step_ep):
            # # change epsilon based on number of steps
            # if self.env_name == "PuddleWorld": #self.agent.learning_mode.vf_update == "DQN":
            #     self.agent.epsilon = max(1.0 - 1.0/50000 * self.count_total_step, self.config.agent_params.epsilon)
            #     # self.agent.epsilon = np.clip(self.agent.epsilon, 0.1, 1.0)
            step_info = self.env.step(self.prev_action)
            self.prev_state, reward, end_of_ep = step_info[:3]
            s_tp = np.copy(self.prev_state)
            gamma = 0 if end_of_ep else self.gamma
            self.prev_action, info = self.agent.step(reward, self.prev_state)

            self.reward_log[self.count_total_step] = reward
            # self.trajectory_log[self.count_total_step, :self.dim_state] = s_t
            # self.trajectory_log[self.count_total_step, self.dim_state] = a_t
            # self.trajectory_log[self.count_total_step, self.dim_state+1: self.dim_state*2+1] = s_tp
            # self.trajectory_log[self.count_total_step, self.dim_state*2+1] = reward
            # self.trajectory_log[self.count_total_step, self.dim_state*2+2] = gamma
            # s_t = s_tp
            # a_t = self.prev_action
            if self.learning == False and reward != 0:
                self.learning = True

            if self.learning:
                self.count_learning_step += 1
            self.count_total_step += 1

            if self.count_total_step % self.log_interval == 0:
                print("Episode {}, total step {}, learning step {}/{}, time={}".format(
                    self.count_ep, self.count_total_step, self.count_learning_step,
                    self.num_steps, time.time() - self.old_time))
                self.old_time = time.time()
                print("check q value", self.prev_state, self.agent.check_q_value())

            condition = self.count_total_step

        self.step_log.append(self.count_total_step - self.old_count_total_step)
        if end_of_ep:
            ep_return = np.sum(np.array(self.reward_log[self.old_count_total_step: self.count_total_step]))

            print("Episode {} ends, ep total step {}, learning step {}/{}, ep return {}, accumulate reward {}\n".format(
                self.count_ep, self.step_log[-1], self.count_learning_step, self.num_steps,
                ep_return,
                np.sum(np.array(self.reward_log))))
            # print("Episode {} ends, ep total step {}/{}, ep return {}, accumulate reward {}\n".format(
            #     self.count_ep, self.step_log[-1], self.num_steps,
            #     np.sum(np.array(self.reward_log[self.old_count_total_step: self.count_total_step])),
            #     np.sum(np.array(self.reward_log))))
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
    oml.run_exp()
