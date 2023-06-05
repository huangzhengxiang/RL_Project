import numpy as np
import random
import torch
import gc

class ReplayBuffer(object):
    def  __init__(self,buffer_size,state_dim) -> None:
        self.buffer_size = buffer_size
        self.data = []
        self.state_dim = state_dim
        self.head=0
        self.initiated=False
        self.full=False
        
    def _init(self,record):
        s_shape = [self.buffer_size]
        s_shape.extend(list(torch.tensor(record[0]).shape))
        all_s = torch.zeros(s_shape,dtype=float)
        a_shape = [self.buffer_size]
        a_shape.extend(list(torch.tensor(record[1]).shape))
        all_a = torch.zeros(a_shape,dtype=float)
        r_shape = [self.buffer_size]
        r_shape.extend(list(torch.tensor(record[2]).shape))
        all_r = torch.zeros(r_shape,dtype=float)
        all_sp = torch.zeros(s_shape,dtype=float)
        t_shape = [self.buffer_size]
        t_shape.extend(list(torch.tensor(record[4]).shape))
        all_t = torch.zeros(t_shape,dtype=float)
        self.data = [all_s,all_a,all_r,all_sp,all_t]

    def update(self,record):
        if not self.initiated:
            self.initiated = True
            self._init(record)
        for j in range(5):
            self.data[j][self.head,...]=torch.tensor(record[j],dtype=float)[...]
        self.head += 1
        if self.head == self.buffer_size:
            self.head = 0 # warp around
            self.full = True
        del record
        gc.collect()
        return
    
    def sample(self,batch_size):
        high = self.head
        if self.full:
            high = self.buffer_size
        rand_id=np.random.randint(0,high,batch_size)
        batch = torch.tensor(rand_id)
        batch_dict = {
            "s": torch.tensor([batch[j][0] for j in range(batch_size)]).float(),
            "a": torch.tensor([batch[j][1] for j in range(batch_size)]).float(),
            "r": torch.tensor([batch[j][2] for j in range(batch_size)]).float(),
            "sp":torch.tensor([batch[j][3] for j in range(batch_size)]).float(),
            "terminated":torch.tensor([batch[j][4] for j in range(batch_size)]).float()
        }
        del batch
        return batch_dict
    
class StateBuffer(object):
    def  __init__(self,buffer_size) -> None:
        self.buffer_size = buffer_size
        self.data = []

    def update(self,record):
        if self.is_full():
            self.data.pop(0)
        self.data.append(record.copy())
        return
    
    def is_full(self):
        return len(self.data)==self.buffer_size
    
    def get_image(self):
        assert self.is_full()
        return self.data.copy()

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