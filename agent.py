import numpy as np
import random
import torch
import gc

class ReplayBuffer(object):
    def  __init__(self,buffer_size,state_dim) -> None:
        self.buffer_size = buffer_size
        self.data = [[] for _ in range(buffer_size)]
        self.state_dim = state_dim
        self.head=0
        self.full=False

    def update(self,record):
        # del self.data[self.head]
        # gc.collect()
        self.data[self.head]=record
        self.head += 1
        if self.head == self.buffer_size:
            self.head = 0 # warp around
            self.full = True
        return
    
    def sample(self,batch_size):
        high = self.head
        if self.full:
            high = self.buffer_size
        rand_id=np.random.randint(0,high,batch_size)
        if high==0:
            return None
        batch = [self.data[rand_id[j]] for j in range(batch_size)]
        batch = {
            "s": torch.tensor([batch[j][0] for j in range(batch_size)]).float(),
            "a": torch.tensor([batch[j][1] for j in range(batch_size)]).float(),
            "r": torch.tensor([batch[j][2] for j in range(batch_size)]).float(),
            "sp":torch.tensor([batch[j][3] for j in range(batch_size)]).float(),
            "terminated":torch.tensor([batch[j][4] for j in range(batch_size)]).float()
        }
        return batch
    
class StateBuffer(object):
    def  __init__(self,buffer_size) -> None:
        self.buffer_size = buffer_size
        self.data = []

    def update(self,record):
        if self.is_full():
            self.data.pop(0)
        self.data.append(record)
        return
    
    def is_full(self):
        return len(self.data)==self.buffer_size
    
    def get_image(self):
        assert self.is_full()
        return self.data

    def clean(self):
        del self.data
        gc.collect()
        self.data=[]
        return

def action(pi_a_s):
    """
    Return the action based on pi(a|s).
    """
    pro=random.random()
    for a in range(len(pi_a_s)):
        pro -= pi_a_s[a]
        if pro <= 0.:
            return a

@torch.no_grad()
def eps_greedy(q,s,action_space,epsilon):
    """
    Args:
        q: the q function approximator
        s: the current state.
        action_space: action number (discrete)
    Returns:
        pi_a_s: p(a|s)
    """
    s=np.array(s).reshape(1,-1)
    A=len(action_space)
    pi_a_s=np.ones(A)*(epsilon/A)
    q_s=[]
    for a in range(A):
        q_s.append(q(s,action_space[a]).item())
    q_s=np.array(q_s)
    pi_a_s[np.argmax(q_s)]+=1-epsilon
    return pi_a_s

@torch.no_grad()
def greedy(q,s,action_space):
    """
    Args:
        q: the q function approximator
        s: [B,S] the current state.
        action_space: action number (discrete)
    Returns:
        max_a q(s,a), argmax_a q(s,a)
    """
    B=s.shape[0]
    A=len(action_space)
    q_s=[]
    for a in range(A):
        a = torch.tensor(action_space[a]).reshape(1,1).repeat(B,1)
        q_s.append(q(s,a).reshape(-1,1))
    q_s=torch.cat(q_s,dim=-1)
    return torch.max(q_s,dim=-1)

def epsilon_decay(init_eps,last_eps,e,term=50):
    if e<term:
        return last_eps + (1-e/term)*(init_eps-last_eps)
    else:
        return last_eps