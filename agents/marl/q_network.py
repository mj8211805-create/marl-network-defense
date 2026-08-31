"""Vectorized Deep Q-Network and Replay Buffer implemented with high-performance NumPy."""

import numpy as np
import pickle
from typing import Tuple, List, Dict, Any, Optional
from config import config


class ReplayBuffer:
    """Fixed-size circular experience replay memory buffer."""

    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(capacity, dtype=np.int64)
        self.rew_buf = np.zeros(capacity, dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done_buf = np.zeros(capacity, dtype=np.float32)

    def store(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = action
        self.rew_buf[self.ptr] = reward
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            self.obs_buf[idxs],
            self.act_buf[idxs],
            self.rew_buf[idxs],
            self.next_obs_buf[idxs],
            self.done_buf[idxs]
        )


class VectorizedDQN:
    """Multi-Layer Perceptron Q-Network with Adam Optimizer and Backprop."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64, lr: float = 0.001):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.lr = lr

        # He / Kaiming Normal Initialization
        self.W1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim), dtype=np.float32)
        self.W2 = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, hidden_dim), dtype=np.float32)
        self.W3 = np.random.randn(hidden_dim, output_dim).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.b3 = np.zeros((1, output_dim), dtype=np.float32)

        # Adam Optimizer Momentums
        self.m = {k: np.zeros_like(v) for k, v in self._get_params().items()}
        self.v = {k: np.zeros_like(v) for k, v in self._get_params().items()}
        self.t = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8

    def _get_params(self) -> Dict[str, np.ndarray]:
        return {
            "W1": self.W1, "b1": self.b1,
            "W2": self.W2, "b2": self.b2,
            "W3": self.W3, "b3": self.b3
        }

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Computes Q-values for input observations: Shape (batch_size, output_dim)."""
        if X.ndim == 1:
            X = X.reshape(1, -1)

        z1 = np.dot(X, self.W1) + self.b1
        a1 = np.maximum(0, z1)  # ReLU
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = np.maximum(0, z2)  # ReLU
        Q = np.dot(a2, self.W3) + self.b3
        return Q

    def train_step(self, obs: np.ndarray, actions: np.ndarray, targets: np.ndarray) -> float:
        """Vectorized forward and backprop pass with Adam gradient updates."""
        batch_size = obs.shape[0]

        # Forward Pass with cache
        z1 = np.dot(obs, self.W1) + self.b1
        a1 = np.maximum(0, z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = np.maximum(0, z2)
        Q = np.dot(a2, self.W3) + self.b3

        # Compute Loss: MSE on selected actions
        pred_q = Q[np.arange(batch_size), actions]
        td_error = pred_q - targets
        loss = float(np.mean(td_error ** 2))

        # Output Layer Gradients
        dQ = np.zeros_like(Q)
        dQ[np.arange(batch_size), actions] = (2.0 / batch_size) * td_error

        dW3 = np.dot(a2.T, dQ)
        db3 = np.sum(dQ, axis=0, keepdims=True)

        # Layer 2 Gradients
        da2 = np.dot(dQ, self.W3.T)
        dz2 = da2 * (z2 > 0)
        dW2 = np.dot(a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        # Layer 1 Gradients
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * (z1 > 0)
        dW1 = np.dot(obs.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        grads = {
            "W1": dW1, "b1": db1,
            "W2": dW2, "b2": db2,
            "W3": dW3, "b3": db3
        }

        # Adam Optimizer Update
        self.t += 1
        params = self._get_params()
        for k in params:
            g = np.clip(grads[k], -5.0, 5.0)  # Gradient clipping
            self.m[k] = self.beta1 * self.m[k] + (1.0 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1.0 - self.beta2) * (g ** 2)

            m_hat = self.m[k] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1.0 - self.beta2 ** self.t)

            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        return loss

    def copy_weights_from(self, source_net: "VectorizedDQN", tau: float = 1.0):
        """Copies or soft-updates weights from another Q-Network."""
        params_self = self._get_params()
        params_src = source_net._get_params()
        for k in params_self:
            params_self[k] = tau * params_src[k] + (1.0 - tau) * params_self[k]


class DQNAgent:
    """Individual Deep Q-Learning Agent with Experience Replay & Target Network."""

    def __init__(self, name: str, obs_dim: int, action_dim: int, lr: float = 0.001, gamma: float = 0.95):
        self.name = name
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma

        self.eval_net = VectorizedDQN(obs_dim, action_dim, hidden_dim=64, lr=lr)
        self.target_net = VectorizedDQN(obs_dim, action_dim, hidden_dim=64, lr=lr)
        self.target_net.copy_weights_from(self.eval_net)

        self.buffer = ReplayBuffer(capacity=config.MEMORY_SIZE, obs_dim=obs_dim)
        self.steps_trained = 0

    def select_action(self, observation: np.ndarray, epsilon: float = 0.0) -> int:
        """Epsilon-greedy action selection."""
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.action_dim)
        q_values = self.eval_net.forward(observation)
        return int(np.argmax(q_values[0]))

    def remember(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool):
        self.buffer.store(obs, action, reward, next_obs, done)

    def train_step(self, batch_size: int = 64) -> Optional[float]:
        if self.buffer.size < batch_size:
            return None

        obs, acts, rews, next_obs, dones = self.buffer.sample_batch(batch_size)

        # Double DQN / Bellman Target Calculation
        next_q_eval = self.eval_net.forward(next_obs)
        best_next_actions = np.argmax(next_q_eval, axis=1)

        next_q_target = self.target_net.forward(next_obs)
        target_q_vals = next_q_target[np.arange(batch_size), best_next_actions]

        targets = rews + (1.0 - dones) * self.gamma * target_q_vals

        loss = self.eval_net.train_step(obs, acts, targets)
        self.steps_trained += 1

        if self.steps_trained % config.TARGET_UPDATE_STEPS == 0:
            self.target_net.copy_weights_from(self.eval_net, tau=0.1)

        return loss
