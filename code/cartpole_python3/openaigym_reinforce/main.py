#!/usr/bin/env python3

import torch
from torch import nn
from torch import optim
import gym
import matplotlib
import matplotlib.pyplot as plt
from network import network_factory
from network import PolicyNetwork
from network import ValueNetwork
import argparse
import numpy as np


# prevents type-3 fonts, which some conferences disallow.
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def make_env():
    env = gym.make('CartPole-v0')
    return env


def sliding_window(data, N):
    idx = 0
    window = np.zeros(N)
    smoothed = np.zeros(len(data))

    for i in range(len(data)):
        window[idx] = data[i]
        idx += 1

        smoothed[i] = window[0:idx].mean()

        if idx == N:
            window[0:-1] = window[1:]
            idx = N - 1

    return smoothed


def discount_returns(rewards, gamma=1.0):
    # Discounts rewards and stores their cumulative return in reverse

    r = rewards[::-1] #rewards in reverse
    G = [r[0]] #last rewards
    for i in range(1,len(r)):
        G.append(r[i] + gamma*G[-1])
    G = G[::-1]
    G = np.array(G)
    return G


def reinforce(env, policy_estimator, value_estimator, num_episodes, # value_estimator=None,
              batch_size=1, gamma=1.0):

    # Set up lists to hold results
    total_rewards = []
    batch_rewards = []
    batch_actions = []
    batch_states = []
    batch_counter = 0

    # Define optimizer
    optimizer =   optim.Adam(policy_estimator.network.parameters(),  lr=0.0025)
    optimizer_v = optim.Adam(value_estimator.network_v.parameters(), lr=0.001)

    action_space = np.arange(env.action_space.n)
    flag = 1     # 1 for train, 0 for test
    for ep in range(num_episodes):
        s_0 = env.reset()
        states = []
        rewards = []
        actions = []
        complete = False

        while complete == False:

            # Gets reward and next state

            action = policy_estimator.get_action(s_0)
            s_1, r, complete, _ = env.step(action)

            states.append(s_0)
            rewards.append(r)
            actions.append(action)
            s_0 = s_1

            # Checks if episode is over

            if complete:

                batch_counter += 1
                batch_rewards.extend(discount_returns(rewards, gamma))
                batch_states.extend(states)
                batch_actions.extend(actions)

                total_rewards.append(sum(rewards))

                # Updates after batch of episodes, here batch is 1

                if batch_counter == batch_size:
                    if flag == 1:

                        # Value update

                        state_tensor_v = torch.tensor(batch_states, dtype=torch.float32)
                        reward_tensor_v = torch.tensor(batch_rewards, dtype=torch.float32)
                        value_estimates = value_estimator.forward(state_tensor_v)
                        loss_v = torch.mean((reward_tensor_v-value_estimates.view(1,-1)[0])**2)

                        optimizer_v.zero_grad()
                        loss_v.backward(retain_graph=True)
                        optimizer_v.step()

                        # Policy update

                        state_tensor = torch.tensor(batch_states, dtype=torch.float32)
                        reward_tensor = torch.tensor(batch_rewards, dtype=torch.float32)
                        action_tensor = torch.tensor(batch_actions, dtype=torch.int32)
                        loss = - torch.mean(policy_estimator.forward(state_tensor).log_prob(action_tensor)*(reward_tensor-value_estimates.view(1,-1)[0]))
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    batch_rewards = []
                    batch_actions = []
                    batch_states = []
                    batch_counter = 0

                print("Ep: {} Average of last 100: {:.2f}".format(
                    ep + 1, np.mean(total_rewards[-100:])))

    return total_rewards

if __name__ == '__main__':

    """
    python main.py --episodes 10000
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", "-e", default=5000, type=int, help="Number of episodes to train for")
    args = parser.parse_args()

    episodes = args.episodes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    numrun = 1

    for run in range(numrun):
        env = make_env()

        in_size = env.observation_space.shape[0]
        num_actions = env.action_space.n

        network = network_factory(in_size, num_actions, env)
        network.to(device)
        pe = PolicyNetwork(network)

        ve = ValueNetwork(in_size)
        ep_returns = reinforce(env, pe, ve, episodes)

    window = 100
    plt.figure(figsize=(12,8))
    plt.plot(sliding_window(ep_returns, window))
    plt.title("Episode Return")
    plt.xlabel("Episode")
    plt.ylabel("Average Return (Sliding Window 100)")
    plt.show()

