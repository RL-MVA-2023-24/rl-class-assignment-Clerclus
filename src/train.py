import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.wrappers import TimeLimit
from torch.distributions import Categorical

from env_hiv import HIVPatient

N = 14
env = gym.vector.AsyncVectorEnv(
    [
        lambda: TimeLimit(
            env=HIVPatient(domain_randomization=False), max_episode_steps=200
        ),
        lambda: TimeLimit(
            env=HIVPatient(domain_randomization=False), max_episode_steps=200
        ),
        lambda: TimeLimit(
            env=HIVPatient(domain_randomization=False), max_episode_steps=200
        ),
        lambda: TimeLimit(
            env=HIVPatient(domain_randomization=False), max_episode_steps=200
        ),
        lambda: TimeLimit(
            env=HIVPatient(domain_randomization=False), max_episode_steps=200
        ),
        lambda: TimeLimit(
            env=HIVPatient(domain_randomization=False), max_episode_steps=200
        ),
        lambda: TimeLimit(
            env=HIVPatient(domain_randomization=False), max_episode_steps=200
        ),
        lambda: TimeLimit(
            env=HIVPatient(domain_randomization=False), max_episode_steps=200
        ),
        lambda: TimeLimit(
            env=HIVPatient(domain_randomization=False), max_episode_steps=200
        ),
        lambda: TimeLimit(
            env=HIVPatient(domain_randomization=False), max_episode_steps=200
        ),
        lambda: TimeLimit(
            env=HIVPatient(domain_randomization=False), max_episode_steps=200
        ),
        lambda: TimeLimit(
            env=HIVPatient(domain_randomization=True), max_episode_steps=200
        ),
        lambda: TimeLimit(
            env=HIVPatient(domain_randomization=True), max_episode_steps=200
        ),
        lambda: TimeLimit(
            env=HIVPatient(domain_randomization=True), max_episode_steps=200
        ),
    ],
    shared_memory=False,
)


def feature(state):
    T1, T1star, T2, T2star, V, E = state
    p1 = T1 * V
    p2 = T2 * V
    p1star = T1star * E
    p2star = T2star * E
    return np.array([p1, p1star, p2, p2star, T1, T1star, T2, T2star, V, E])


def features(states):
    return np.apply_along_axis(feature, 1, states)


class policyNetwork(nn.Module):
    def __init__(self, nb_neurons=128):
        super().__init__()
        T1Upper = 1e6
        T1starUpper = 5e4
        T2Upper = 3200.0
        T2starUpper = 80.0
        VUpper = 2.5e5
        EUpper = 353200.0
        p1upper = 2.5000e11
        p1starupper = 1.7660e10
        p2upper = 8.0000e08
        p2starupper = 2.8256e07
        upper = torch.tensor(
            np.array(
                [
                    p1upper,
                    p1starupper,
                    p2upper,
                    p2starupper,
                    T1Upper,
                    T1starUpper,
                    T2Upper,
                    T2starUpper,
                    VUpper,
                    EUpper,
                ]
            )
        ).float()
        self.register_buffer(name="upper", tensor=upper)

        state_dim = 10
        n_action = 4
        self.fc1 = nn.Linear(state_dim, nb_neurons)
        self.fc2 = nn.Linear(nb_neurons, nb_neurons)
        # self.fc3 = nn.Linear(nb_neurons, nb_neurons)
        self.fc4 = nn.Linear(nb_neurons, n_action)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(dim=0)
        x = x / self.upper
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        # x = self.fc3(x)
        # x = F.relu(x)
        action_scores = self.fc4(x)
        return F.softmax(action_scores, dim=1)

    def sample_action(self, x):
        probabilities = self.forward(x)
        action_distribution = Categorical(probabilities)
        return action_distribution.sample().item()

    def sample_action_and_log_prob(self, x):
        probabilities = self.forward(x)
        action_distribution = Categorical(probabilities)
        action = action_distribution.sample()
        log_prob = action_distribution.log_prob(action)
        return action, log_prob


config = {
    "nb_neurons": 128,
    "lr": 1e-2,
    "weight_decay": 0,  # 2e-6,
    "gamma": 0.99,
}


class ProjectAgent:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy_network = policyNetwork(config["nb_neurons"]).to(self.device)

        self.scalar_dtype = next(policy_network.parameters()).dtype
        self.policy = policy_network
        self.gamma = config["gamma"]
        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=20, gamma=0.9
        )

    def one_gradient_step(self, env):
        states = []
        actions = []
        log_probs = []
        x, _ = env.reset()
        x = features(x)
        rewards = []
        episode_cum_reward = np.zeros(N)
        while True:
            a, log_prob = self.policy.sample_action_and_log_prob(
                torch.tensor(x).to(self.device)
            )
            a = a.detach().cpu().numpy()
            log_probs.append(log_prob)
            y, r, done, trunc, _ = env.step(a)
            y = features(y)
            states.append(x)
            actions.append(a)
            rewards.append(r)
            episode_cum_reward += r
            x = y
            if done[0] or trunc[0]:
                new_returns = []
                G_t = np.zeros(N)
                for r in reversed(rewards):
                    G_t = r + self.gamma * G_t
                    new_returns.append(G_t)
                returns = np.flip(np.array(new_returns), axis=0)
                break
        # loss
        returns = torch.tensor(returns.copy()).to(self.device)
        returns = (returns - returns.mean(axis=1)[:, None]) / (
            returns.std(axis=1)[:, None] + 1e-6
        )
        returns = returns.reshape(-1)
        log_probs = torch.cat(log_probs).reshape(-1)
        loss = -(returns * log_probs).mean()
        # gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return np.mean(episode_cum_reward), loss.item()

    def train(self, env, nb_rollouts):
        avg_sum_rewards = []
        losses = []
        for ep in range(nb_rollouts):
            reward, loss = self.one_gradient_step(env)
            print(f"Epoch {ep:3} - Reward {reward:1.3e}")
            losses.append(loss)
            avg_sum_rewards.append(reward)
        return avg_sum_rewards, losses

    def act(self, observation, use_random=False):
        obs = feature(observation)
        with torch.no_grad():
            a = self.policy(torch.tensor(obs).to(self.device).float())
            return torch.argmax(a).item()

    def save(self, path):
        # path = "models/agent.pth"
        torch.save(self.policy.state_dict(), path)

    def load(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.load_state_dict(
            torch.load("models/agent.pth", map_location=torch.device(device))
        )
