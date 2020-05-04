import include_parent_folder
import numpy as np
from importlib import import_module
import torch
import time
import pickle as pkl
from sklearn.neighbors import BallTree

from utils.collect_config import ParameterConfig, Sweeper
from utils.collect_parser import CollectInput
from utils.log_and_plot_func import write_param_log
import os
from Environments.SensorDriftWrapper import SensorDriftWrapper


def saved_file_name(config, run_idx, eval=False):
    learning = config.learning
    if learning == "offline" and eval:
        learning += "_eval"
        training_steps = "_training{}".format(config.exp_params.num_steps)
    else:
        training_steps = ""
    agent_params = config.agent_params
    env_params = config.env_params
    input_ = agent_params.rep_type
    input_ = input_[0].upper() + input_[1:]
    if agent_params.rep_type in ["TC", "sepTC", "sep_pair_TC"]:
        input_ += "{}x{}".format(agent_params.num_tilings, agent_params.num_tiles)
    if config.agent == "DQN":
        other_info = "_B{}_sync{}_NN{}_batch{}".format(agent_params.len_buffer,
                                             agent_params.dqn_sync,
                                             str(agent_params.nonLinearQ_node),
                                             agent_params.dqn_minibatch
                                             )
        if agent_params.dqn_sync == 1:
            other_info += "_plan{}".format(agent_params.num_planning)
    elif config.agent == "ExpectedSarsaTileCodingContinuing":
        other_info = "_lmbda{}_epsilon{}".format(agent_params.lmbda, agent_params.epsilon)
    else:
        other_info = ""
    if env_params.drift_prob > 0:
        other_info += (f'_drift_scale{env_params.drift_scale}' +
            f'_life{str(env_params.sensor_life)}_prob{env_params.drift_prob}')
    elif env_params.drift_prob < 0:
        other_info += f'_drift_scale{env_params.drift_scale}'

    file_path = "{}/{}/{}_{}{}_alpha{}_input{}".format(
        agent_params.exp_result_path,
        learning,
        config.environment,
        config.agent,
        other_info,
        agent_params.alpha,
        input_,
        training_steps
    )
    file_path += "/"
    file_name = "run" + str(run_idx)
    return file_path, file_name

class Experiment():
    def __init__(self, config, parsers):
        self.run_idx = parsers.run_idx
        self.config = config
        run_seed = self.config.exp_params.random_seed * self.run_idx
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Load an environment based on the config, and pass relevant portion of config to it.
        self.domain = parsers.domain.lower()
        env_code = import_module("Environments.{}".format(self.config.environment))
        self.env_name = self.config.environment
        if self.env_name == "OfflineEnvironment":
            self.env_simulator_training(file_path=self.config.simulator_path)

        self.env = env_code.init_env()
        if config.env_params.drift_prob != 0:
            self.env = SensorDriftWrapper(self.env)
        self.env.set_param(config.env_params)

        # Pass environment info to the agent.
        num_action = self.env.num_action()
        self.dim_state = self.env.state_dim()
        state_normalize = self.env.state_range()
        setattr(self.config.agent_params, "num_action", num_action)
        setattr(self.config.agent_params, "dim_state", self.dim_state)
        setattr(self.config.agent_params, "state_normalize", np.array(state_normalize))

        # Load an agent based on the config, and pass relevant portion of config to it.
        self.agent_code = import_module("Agents.{}".format(self.config.agent))
        self.agent = self.agent_code.init_agent()
        self.agent.set_param(self.config.agent_params)
        self.gamma = self.config.agent_params.gamma

        self.reset_exp()

    def reset_exp(self):
        self.num_steps = self.config.exp_params.num_steps
        if hasattr(self.config.exp_params, "max_step_ep") and self.config.exp_params.max_step_ep != 0:
            self.max_step_ep = self.config.exp_params.max_step_ep
        else:
            self.max_step_ep = np.inf

        # These 3 logs are used for debugging or to create logs for offline training.
        self.step_log = []
        self.reward_log = np.zeros(self.num_steps)
        self.trajectory_log = np.zeros((self.num_steps, self.dim_state*2+3))

        # Set various other parameters. Most are for the log.
        self.num_episode = self.config.exp_params.num_episodes
        self.control_ep = False if self.config.exp_params.num_episodes == 0 else self.config.exp_params.num_episodes
        self.learning = False # Mainly used in domains with a sparse reward with 1000s of steps before the first reward
        self.count_ep = 0
        self.count_total_step = 0
        self.count_learning_step = 0
        self.old_count_learning_step = 0
        self.old_count_total_step = 0
        self.log_interval = 200
        self.old_time = time.time()

    # Save one or more log files.
    # For online learning, likely just save the step and reward.
    # For offline learning, it's important to also save the trajectory.
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
        # Print parameter settings before doing anything
        for pair in self.config.env_params.__dict__:
            space = " " * (20 - len(str(pair))) + ": "
            print(str(pair), space, str(self.config.env_params.__dict__[pair]))

        self.single_run()

        # self.config.agent_params = self.agent.get_settings() # In case some settings were calculated or changed, read them because they might be saved in a log.
        self.save_log(save_Q=True)

    def offline_learning(self):
        t0 = time.time()
        # Save name of agent because collect_trajectory will change it
        eval_code = self.agent_code
        eval_name = self.config.agent

        path, name = self.collect_trajectory()
        q_path, q_name = self.learn_policy(path, name, eval_code, eval_name)
        # self.online_evaluation(q_path, q_name, eval_code, eval_name, 10000)
        self.offline_evaluation(q_path, q_name, eval_code, eval_name, 10000)
        print("Total running time:", time.time() - t0)

    def collect_trajectory(self, save=True):
        # Collect trajectories with some policy
        self.config.agent = self.config.learn_from
        agent_code = import_module("Agents.{}".format(self.config.learn_from))
        self.agent = agent_code.init_agent()

        self.reset_exp()
        self.single_run()
        if save:
            path, name = self.save_log(save_step=False, save_reward=False, save_traj=True, save_Q=False)
            return path, name

    def env_simulator_training(self, file_path=None):
        if file_path is None:
            np.random.seed(2048) # independent random seed for environment model training data
            self.collect_trajectory(save=False)
            offline_env = []
            offline_data = []
            self.trajectory_log[:, -1] = self.trajectory_log[:, -1] == 0
            for a in range(self.config.agent_params.num_action):
                idx = np.where(self.trajectory_log[:, self.dim_state] == a)[0]
                key = self.trajectory_log[idx][:, :self.dim_state]
                offline_data.append(np.copy(self.trajectory_log[idx]))
                offline_env.append(BallTree(key, leaf_size=40))
            offline_env_model = {
                "offline_data": offline_data,
                "offline_env": offline_env
            }
            path = self.config.simulator_path
            if not os.path.isdir("".join(path.split("/")[:-1])):
                os.makedirs("".join(path.split("/")[:-1]))
            with open(path, "wb") as f:
                pkl.dump(offline_env_model, f)
            print("Model saved in", path)
        else:
            with open(file_path, "rb") as f:
                offline_env_model = pkl.load(f)
        setattr(self.config.env_params, "offline_env_model", offline_env_model)

        # # test
        # terminal = np.where(self.trajectory_log[:, -1] == 1)[0]
        # for a in range(self.config.agent_params.num_action):
        #     _, idx = offline_env[a].query([self.trajectory_log[terminal[0], :self.dim_state]], k=1)
        #     print(offline_data[a][idx[0][0]])
        #     print(offline_data[a][idx[0][0]][4*2+2])
        # exit()

    def offline_env_test(self, run_idx):
        np.random.seed(config.exp_params.random_seed * run_idx)
        env_backup = self.env
        env_code = import_module("Environments.OfflineEnvironment")
        env = env_code.init_env()
        env.set_param(self.config.offline_env_model)

        max_step = 100000
        trajectory = np.zeros((max_step, self.dim_state*2+3))

        agent_code = import_module("Agents.{}".format(self.config.learn_from))
        agent = agent_code.init_agent()
        terminal = True
        count = 0
        total = 0
        while True:
            if terminal:
                print("Step", total)
                state = env.start()
                count = 0
                action = agent.start(state)
            next_state, reward, terminal, _ = env.step(action)
            seq = np.concatenate((state, np.array([action]), next_state, np.array([reward]), np.array([terminal==0])))
            trajectory[total] = seq
            action, _ = agent.step(reward, next_state)
            state = next_state
            count += 1
            total += 1
            if total >= max_step:
                break

        path = "../data/offline_env/test/"+self.config.learn_from
        if not os.path.isdir(path):
            os.makedirs(path)
        np.save("{}/run{}_rewardPerStep".format(path, run_idx), trajectory[:, self.dim_state*2+1])

        self.env = env_backup
        run_seed = config.exp_params.random_seed * self.run_idx
        np.random.seed(run_seed)

    def learn_policy(self, path, name, eval_code, eval_name):
        # Load saved trajectory
        simulated_env_1 = np.load(path+name+"_trajectory.npy")
        simulated_env = np.copy(simulated_env_1)
        for _ in range(1, self.config.agent_params.offline_repeat):
            simulated_env = np.concatenate((simulated_env, simulated_env_1), axis=0)

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
        self.config.exp_params.num_steps = len(simulated_env)
        while t < len(simulated_env)-1:
            end = False
            self.agent.offline_start(st[t], t)
            while not end and t < len(simulated_env)-1:
                if gamma[t] == 0:
                    end = True
                self.agent.offline_step(reward[t], st[t+1], end, t+1)
                t += 1
                if t % 5000 == 0:
                    path, name = self.save_log(save_step=False, save_reward=False, save_traj=False, save_Q=True)
                    print("\nEvaluating agent at step", t)
                    learning_agent = self.agent
                    # self.online_evaluation(path, name, eval_code, eval_name, 200)
                    self.offline_evaluation(path, name, eval_code, eval_name, 200, save=False)
                    self.agent = learning_agent
                    print("\nEvaluation ends, keep learning...")

        path, name = self.save_log(save_step=False, save_reward=False, save_traj=False, save_Q=True)
        return path, name

    def offline_evaluation(self, path, name, eval_code, eval_name, eval_step, save=True):
        env_backup = self.env
        env_code = import_module("Environments.OfflineEnvironment")
        self.env = env_code.init_env()
        self.env.set_param(self.config.offline_env_model)

        self.agent_code = eval_code
        self.config.agent = eval_name
        training_step = self.config.exp_params.num_steps
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
        self.config.exp_params.num_steps = training_step
        if save:
            self.save_log(save_step=True, save_reward=True, save_traj=False, save_Q=False, eval=True)
        self.env = env_backup

    def online_evaluation(self, path, name, eval_code, eval_name, eval_step):
        # Evaluate agent with learned policy. Weights are frozen (by setting alpha to zero). Epsilon is zero. (Note epsilon is non-zero in the online setting.)
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
        if self.control_ep:
            for _ in range(self.num_episode):
                self.single_ep()
                self.count_ep += 1
        else:
            condition = self.count_total_step
            while condition < self.num_steps:
                print(
                    "Episode {} starts, total step {}, learning step {}/{}".format(self.count_ep, self.count_total_step,
                                                                                   self.count_learning_step,
                                                                                   self.num_steps))
                self.single_ep()
                self.count_ep += 1
                condition = self.count_total_step

    # Run a single episode and log lots of things.
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
    t0 = time.time()
    # parse arguments
    ci = CollectInput()
    parsers = ci.control_experiment_input()
    json_name = parsers.domain.lower()

    # Load the config. Then, based on the sweeper_idx, choose one of the sweeper parameters
    # and use that to override any of the default parameters (or set the value if there
    # wasnt' a default).
    sweeper = Sweeper('../Parameters/{}.json'.format(json_name.lower()), "control_param")
    config = sweeper.parse(parsers.sweeper_idx)

    exp = Experiment(config, parsers)
    if config.learning == "online":
        exp.online_learning()
    else:
        # exp.offline_env_test(exp.run_idx)
        exp.offline_learning()

    print("Running time", time.time() - t0)