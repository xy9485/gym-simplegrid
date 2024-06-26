from typing import Dict
import numpy as np


class ReplayBuffer:
    def __init__(self, obs_dim: int, size: int, batch_size: int = 32):
        self.obs_dim = obs_dim
        self.size = size
        self.batch_size = batch_size

        self.reset()
        
    def reset(self):
        if self.obs_dim == 1:
            self.obs_buf = np.zeros([self.size], dtype=np.int32)
            self.next_obs_buf = np.zeros([self.size], dtype=np.int32)
        else:
            self.obs_buf = np.zeros([self.size, self.obs_dim], dtype=np.float32)
            self.next_obs_buf = np.zeros([self.size, self.obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([self.size], dtype=np.int32)
        self.rews_buf = np.zeros([self.size], dtype=np.float32)
        self.done_buf = np.zeros(self.size, dtype=np.float32)
        self.prob_acts_buf = np.zeros(self.size, dtype=np.float32)
        self.max_size, self.batch_size = self.size, self.batch_size        
        self.ptr, self.size = 0, 0

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray, 
        rew: float, 
        next_obs: np.ndarray, 
        done: bool,
        prob_act: float
    ):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.prob_acts_buf[self.ptr] = prob_act
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs],
                    prob_acts=self.prob_acts_buf[idxs])

    def __len__(self) -> int:
        return self.size