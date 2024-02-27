import random

from loguru import logger
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
from gymnasium.wrappers import TimeLimit

from env_hiv import HIVPatient

env = TimeLimit(env=HIVPatient(domain_randomization=False), max_episode_steps=200)


class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = int(capacity)
        self.data = []
        self.index = 0
        self.device = device

    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(
            map(lambda x: torch.Tensor(np.array(x)).to(self.device), list(zip(*batch)))
        )

    def __len__(self):
        return len(self.data)


config = {
    "nb_neurons": 128,
    "lr": 1e-4,
    "weight_decay": 2e-6,
    "gamma": 0.95,
    "buffer_size": 2e4,
    "epsilon_min": 0.01,
    "epsilon_max": 1.0,
    "epsilon_decay_period": 500 * 200,
    "epsilon_delay_decay": 8 * 200,
    "batch_size": 64,
    "monitoring_nb_trials": 0,
    "nb_gradient_steps": 1,
    "update_target_freq": 500,
}


def features(state):
    # = np.array([self.T1, self.T1star, self.T2, self.T2star, self.V, self.E]) = static
    T1, T1star, T2, T2star, V, E = state
    p1 = T1 * V
    p2 = T2 * V
    p1star = T1star * E
    p2star = T2star * E
    return np.array([p1, p1star, p2, p2star, T1, T1star, T2, T2star, V, E])


class ProjectAgent:

    def __init__(
        self,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        nb_neurons = config["nb_neurons"]
        self.gamma = config["gamma"]
        self.batch_size = config["batch_size"]

        self.network = torch.nn.Sequential(
            nn.Linear(10, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, 4),
        ).to(device)
        self.target_network = deepcopy(self.network).to(device)

        self.nb_gradient_steps = config["nb_gradient_steps"]
        self.update_target_freq = config["update_target_freq"]

        self.epsilon_max = config["epsilon_max"]
        self.epsilon_min = config["epsilon_min"]
        self.epsilon_stop = config["epsilon_decay_period"]
        self.epsilon_delay = config["epsilon_delay_decay"]
        self.epsilon_step = (self.epsilon_max - self.epsilon_min) / self.epsilon_stop

        self.memory = ReplayBuffer(config["buffer_size"], device)
        self.monitoring_nb_trials = config["monitoring_nb_trials"]

        self.criterion = torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )

    def MC_eval(self, env, nb_trials):
        MC_total_reward = []
        MC_discounted_reward = []
        for _ in range(nb_trials):
            x, _ = env.reset()
            done = False
            trunc = False
            total_reward = 0
            discounted_reward = 0
            step = 0
            while not (done or trunc):
                a = self.greedy_action(x)
                y, r, done, trunc, _ = env.step(a)
                x = y
                total_reward += r
                discounted_reward += self.gamma**step * r
                step += 1
            MC_total_reward.append(total_reward)
            MC_discounted_reward.append(discounted_reward)
        return np.mean(MC_discounted_reward), np.mean(MC_total_reward)

    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.network(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1 - D, QYmax, value=self.gamma)
            QXA = self.network(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def greedy_action(self, state):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            Q = self.network(torch.Tensor(state).unsqueeze(0).to(device))
            return torch.argmax(Q).item()

    def train(self, env, max_episode):
        logger.add("logs/train_{time}.log", rotation="50 MB", retention="3 days")
        episode_return = []
        MC_avg_total_reward = []
        MC_avg_discounted_reward = []

        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        state = features(state)
        epsilon = self.epsilon_max
        step = 0

        while episode < max_episode:
            if step % 50 == 0:
                logger.debug(
                    "Episode - {episode:3} - Step {step:5} - Epsilon {epsilon:1.3f} - Episode_cum_reward {episode_cum_reward:1.3e}",
                    episode=episode,
                    step=step,
                    epsilon=epsilon,
                    episode_cum_reward=episode_cum_reward,
                )
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon - self.epsilon_step)

            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = self.greedy_action(state)

            # step
            next_state, reward, done, trunc, _ = env.step(action)
            next_state = features(next_state)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward

            # train
            for _ in range(self.nb_gradient_steps):
                self.gradient_step()
            # update target network if needed
            if step % self.update_target_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())

            # next transition
            step += 1
            if done or trunc:
                if self.monitoring_nb_trials > 0:
                    MC_dr, MC_tr = self.MC_eval(env, self.monitoring_nb_trials)
                    MC_avg_total_reward.append(MC_tr)
                    MC_avg_discounted_reward.append(MC_dr)
                    episode_return.append(episode_cum_reward)
                    logger.info(
                        "Episode - {episode:3} - Epsilon {epsilon:1.3f} - Batch size {len:1.2e} - Episode return {episode_cum_reward:1.3e} - MC total reward {MC_tr:1.3e} - MC discounted reward {MC_dr:1.3e}",
                        episode=episode,
                        epsilon=epsilon,
                        len=len(self.memory),
                        episode_cum_reward=episode_cum_reward,
                        MC_tr=MC_tr,
                        MC_dr=MC_dr,
                    )
                else:
                    episode_return.append(episode_cum_reward)
                    logger.info(
                        "Episode - {episode:3} - Epsilon {epsilon:1.3f} - Batch size {len:1.2e} - Episode return {episode_cum_reward:1.3e}",
                        episode=episode,
                        epsilon=epsilon,
                        len=len(self.memory),
                        episode_cum_reward=episode_cum_reward,
                    )
                episode += 1
                state, _ = env.reset()
                state = features(state)
                episode_cum_reward = 0
            else:
                state = next_state

        return episode_return, MC_avg_discounted_reward, MC_avg_total_reward

    def act(self, observation, use_random=False):
        obs = features(observation)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            Q = self.network(torch.Tensor(obs).unsqueeze(0).to(device))
            return torch.argmax(Q).item()

    def save(self, path):
        # path = "models/agent.pth"
        torch.save(self.network.state_dict(), path)

    def load(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.load_state_dict(
            torch.load("models/agent.pth", map_location=torch.device(device))
        )
