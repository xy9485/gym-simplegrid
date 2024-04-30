from collections import defaultdict
import gymnasium as gym
import gym_simplegrid
import matplotlib
import numpy as np
import os
import shutil
# import torch
# import torchvision
import json
# from torch.utils.tensorboard import SummaryWriter
import scipy
import matplotlib.pyplot as plt




class Agent:
    def __init__(self, alpha, gamma, action_space, observation_space, eps_start, eps_end, eps_decay):
        self.action_space = action_space
        self.n_states = observation_space.n
        self.n_actions = action_space.n
        self.Q = np.zeros((self.n_states, self.n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps_start
        self.eps_min = eps_end
        self.eps_decay = eps_decay

    def choose_action(self, state):
        self.decay_eps()
        if np.random.random() > self.eps:
            action = np.argmax(self.Q[state, :])
            #if there are multiple actions with the same value, choose randomly
            if np.sum(self.Q[state, :] == self.Q[state, action]) > 1:
                action = np.random.choice(np.where(self.Q[state, :] == self.Q[state, action])[0])
        else:
            action = self.action_space.sample()
        return action

    def decay_eps(self):
        self.eps = self.eps * self.eps_decay if self.eps > self.eps_min else self.eps_min
        
    # define abstract method
    def learn(self, state, action, reward, state_):
        pass


class Q_Agent(Agent):
    def __init__(self, alpha, gamma, action_space, observation_space, eps_start, eps_end, eps_dec):
        super().__init__(alpha, gamma, action_space, observation_space, eps_start, eps_end, eps_dec)
        self.action_space = action_space
        self.observation_space = observation_space
        self.n_states = observation_space.n
        self.n_actions = action_space.n
        self.reset_Q()

        
    def learn(self, state, action, reward, state_):
        self.Q[state, action] = self.Q[state, action] + self.alpha * (reward + self.gamma * np.max(self.Q[state_, :]) - self.Q[state, action])

    def reset_Q(self):
        self.Q = np.zeros((self.n_states, self.n_actions))


def dump_to_file(data, file_name):
    with open(file_name, 'a') as f:
        f.write(json.dumps(data) + '\n')


def extract_data_logfile(
    log_path,
    key_name,
    value_name,
    smooth=10,
    max_key=True,
):

    all_keys = []
    all_values = []
    keys = []
    values = []
    last_line_repeat_idx = 0

    with open(log_path, "r") as f:
        for line in f:
            data = json.loads(line)
            if data['repeat_idx'] != last_line_repeat_idx:
                keys, values = np.array(keys), np.array(values)
                if smooth > 1 and values.shape[0] > 0:
                    K = np.ones(smooth)
                    ones = np.ones(values.shape[0])
                    values = np.convolve(values, K, "same") / np.convolve(ones, K, "same")
                all_keys.append(np.array(keys))
                all_values.append(np.array(values))
                last_line_repeat_idx = data['repeat']
                keys = []
                values = []
            if key_name in data and value_name in data:
                keys.append(data[key_name])
                values.append(data[value_name])
    
    all_keys_tmp = sorted(all_keys, key=lambda x: x[-1])
    keys = all_keys_tmp[-1] if max_key else all_keys_tmp[0]
    # threshold = keys.shape[0]

    # interpolate
    for idx, (key, value) in enumerate(zip(all_keys, all_values)):
        f = scipy.interpolate.interp1d(key, value, fill_value="extrapolate")
        all_keys[idx] = keys
        all_values[idx] = f(keys)

    all_values = np.array(all_values)
    means = np.mean(all_values, axis=0)
    half_stds = 0.5 * np.std(all_values, axis=0)

    # means, half_stds = [], []
    # for i in range(threshold):
    #     vals = []

    #     for v in all_values:
    #         if i < v.shape[0]:
    #             vals.append(v[i])
    #     if best_k is not None:
    #         vals = sorted(vals)[-best_k:]
    #     means.append(np.mean(vals))
    #     # half_stds.append(0.5 * np.std(vals))
    #     half_stds.append(np.std(vals))

    # means = np.array(means)
    # half_stds = np.array(half_stds)

    # keys = all_keys[-1][:threshold]
    assert means.shape[0] == keys.shape[0]

    return keys, means, half_stds    

def plot_data(
    keys,
    means,
    half_stds,
    max_time=None,
    label="DVQN",
    color=None,
    key_name=None,
    value_name=None,
):
    if max_time is not None:
        idxs = np.where(keys <= max_time)
        keys = keys[idxs]
        means = means[idxs]
        half_stds = half_stds[idxs]

    plt.rcParams["figure.figsize"] = (10, 7)
    plt.rcParams["figure.dpi"] = 200
    plt.rcParams["font.size"] = 20
    plt.subplots_adjust(left=0.165, right=0.99, bottom=0.16, top=0.95)
    plt.tight_layout()

    plt.plot(keys, means, label=label, color=color)
    plt.locator_params(nbins=10, axis="x")
    plt.locator_params(nbins=10, axis="y")
    # plt.ylim(0, 1050)

    plt.grid(alpha=0.8)
    # ax.title(title)
    plt.fill_between(keys, means - half_stds, means + half_stds, alpha=0.15)
    # plt.legend(loc="lower right", prop={"size": 6}).get_frame().set_edgecolor("0.1")
    # plt.legend(loc="upper left", ncol=1)
    plt.legend(ncol=1)
    plt.xlabel(key_name)
    plt.ylabel(value_name)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))


def run(obstacle_map, start_loc, goal_loc, repeat_idx, total_steps, log_path):
    env = gym.make(
        'SimpleGrid-v0', 
        obstacle_map=obstacle_map, 
        # render_mode='human',
        render_mode=None,
    )
    
    agent = Q_Agent(alpha=0.1, gamma=0.99, action_space=env.action_space, observation_space=env.observation_space, eps_start=0.1, eps_end=0.00, eps_dec=0.9999995)
        
    obs, info = env.reset(options={'start_loc':start_loc, 'goal_loc':goal_loc})
    # done = env.unwrapped.done
    done=False
    episode_reward = 0
    for step in range(total_steps):
        if done:
            print(f'repeat_idx: {repeat_idx} | step: {step} | episode reward:{episode_reward} | maxQ: {np.max(agent.Q)} | eps: {agent.eps}')
            obs, info = env.reset(options={'start_loc':start_loc, 'goal_loc':goal_loc})
            episode_reward = 0
            data = {"repeat_idx": repeat_idx, "step": step, "reward": episode_reward, "maxQ": np.max(agent.Q), "eps": agent.eps}
            dump_to_file(data, log_path)
        action = agent.choose_action(obs)
        new_obs, reward, done, _, info = env.step(action)
        agent.learn(obs, action, reward, new_obs)
        episode_reward += reward
        obs = new_obs

    env.close()           

#entry of the program
if __name__ == "__main__":
    # Load a custom map
    obstacle_map = [
            "0000000",
            "0000000",
            "0000000",
            "1110111",
            "0000000",
            "0000000",
            "0000000",
        ]
    # start_loc = (0,2)
    # goal_loc = [(6,2), (1,6)]
    start_loc = (3,3)
    goal_loc = [(0,3), (6,3)]
    total_steps = 100000
    n_repeat = 1
    # log_path = "results/2room_q_learning.log"
    # if os.path.exists(log_path):
    #     os.remove(log_path)
    # for i_repeat in range(n_repeat):
    #     run(obstacle_map, start_loc, goal_loc, repeat_idx=i_repeat, total_steps=total_steps, log_path=log_path)
    env = gym.make(
        'SimpleGrid-v0', 
        obstacle_map=obstacle_map, 
        render_mode='human',
        # render_mode=None,
        max_episodic_steps=1000,
    )
    env.reset(options={'start_loc':start_loc, 'goal_loc':goal_loc})

    done=False
    for step in range(total_steps):
        if done:
            obs, info = env.reset(options={'start_loc':start_loc, 'goal_loc':goal_loc})
        action = env.action_space.sample()
        new_obs, reward, done, _, info = env.step(action)
        obs = new_obs

    env.close() 