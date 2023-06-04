import torch
import torch.nn as nn
import random
from torch.distributions import MultivariateNormal
from agent import ReplayBuffer, StateBuffer
import os
import torch.multiprocessing as mp
import gymnasium as gym
import cv2
import numpy as np

act_funcs={
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "leaky": nn.LeakyReLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "softplus": nn.Softplus,
    "sigmoid": nn.Sigmoid,
    "logsigmoid": nn.LogSigmoid,
    "softmax": nn.Softmax,
    "logsoftmax": nn.LogSoftmax
}

class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, act_func="sigmoid", last_act=None) -> None:
        super().__init__()
        assert isinstance(hid_dim,list)
        hid_layers=len(hid_dim)
        self.dims=[in_dim]
        self.dims.extend(hid_dim)
        self.dims.append(out_dim)
        self.hid_layers=hid_layers
        self.module_list=[]
        self.module_list = [
            nn.Sequential(
                nn.Linear(self.dims[j],self.dims[j+1]),
                act_funcs[act_func]()
            ) 
            for j in range(hid_layers)
        ]
        if last_act is not None:
            self.module_list.append(nn.Sequential(
                nn.Linear(self.dims[-2],self.dims[-1]),
                act_funcs[last_act]()
            ))
        else:
            self.module_list.append(nn.Linear(self.dims[-2],self.dims[-1]))
        self.mlp=nn.ModuleList(self.module_list)

    def forward(self, x):
        for j in range(self.hid_layers+1):
            x = self.mlp[j](x)
        return x

class ConvDQNet(nn.Module):
    def __init__(self,in_dim,hid_dim,fc_dim,out_dim,act_func="sigmoid",last_act=None,device="cpu") -> None:
        super().__init__()
        assert len(hid_dim)==3
        self.device=device
        self.cnn1=nn.Sequential(
            nn.Conv2d(
                in_channels=in_dim,
                out_channels=hid_dim[0],
                kernel_size=8,
                stride=4
            ),
            act_funcs[act_func]()
        )
        self.cnn2=nn.Sequential(
            nn.Conv2d(
                in_channels=hid_dim[0],
                out_channels=hid_dim[1],
                kernel_size=4,
                stride=2
            ),
            act_funcs[act_func]()
        )
        self.cnn3=nn.Sequential(
            nn.Conv2d(
                in_channels=hid_dim[1],
                out_channels=hid_dim[2],
                kernel_size=3,
                stride=1
            ),
            act_funcs[act_func]()
        )
        if last_act is None:
            self.fc=nn.Sequential(
                nn.Flatten(),
                nn.Linear(hid_dim[-1]*49,fc_dim),
                act_funcs[act_func](),
                nn.Linear(fc_dim,out_dim)
            )
        else:
            self.fc=nn.Sequential(
                nn.Flatten(),
                nn.Linear(hid_dim[-1]*49,fc_dim),
                act_funcs[act_func](),
                nn.Linear(fc_dim,out_dim),
                act_funcs[last_act]()
            )

    def forward(self,s):
        B, C, H, W = s.shape[:4]
        if not isinstance(s,torch.Tensor):
            s = torch.tensor(s)
        s = s.float().reshape(B,C,H,W)
        # only encode state and predicts Q values of all actions.
        x = s.to(device=self.device).float()
        action = self.fc(self.cnn3(self.cnn2(self.cnn1(x))))
        return action

class DQNet(nn.Module):
    def __init__(self,in_dim,hid_dim,out_dim,act_func="sigmoid",last_act=None,device="cpu") -> None:
        super().__init__()
        self.device=device
        self.mlp=MLP(in_dim,hid_dim,out_dim,act_func,last_act)

    def forward(self,s,a=None):
        B = s.shape[0]
        if not isinstance(s,torch.Tensor):
            s = torch.tensor(s).reshape(B,-1)
        s = s.float().reshape(B,-1)
        if a is None:
            # only encode state and predicts Q values of all actions.
            x = s.to(device=self.device).float()
            return self.mlp(x)
        else:
            if not isinstance(a,torch.Tensor):
                a = torch.tensor(a).reshape(B,-1)
            a = a.float().reshape(B,-1)
            x = torch.cat([s,a],dim=-1).to(device=self.device).float()
            return self.mlp(x).reshape(-1)
    
class DVNet(nn.Module):
    def __init__(self,in_dim,hid_dim,out_dim,act_func="sigmoid",last_act=None,device="cpu") -> None:
        super().__init__()
        self.device=device
        self.mlp=MLP(in_dim,hid_dim,out_dim,act_func,last_act)

    def forward(self,s):
        B = s.shape[0]
        if not isinstance(s,torch.Tensor):
            s = torch.tensor(s).reshape(B,-1)
        x = s.reshape(B,-1).to(device=self.device).float()
        return self.mlp(x).reshape(-1)

def get_model_name(model_type,action_space):
    if model_type=="":
        if len(action_space.shape) == 1:
            # discrete problem
            model_type = "DQN"
        else:
            # continuous problem
            model_type = "DDPG"
    return model_type

def model_parser(model_type,config,state_dim,action_space):
    if (model_type=="DDPG"):
        return DDPG(config,state_dim,action_space)
    elif (model_type=="A2C"):
        return A2C(config,state_dim,action_space)
    elif (model_type=="A3C"):
        return A3C(config,state_dim,action_space)
    elif (model_type=="DQN"):
        return BaseDQN(config,state_dim,action_space)
    else:
        print("Unsupported Model Error!")
        
def preproccess(figure):
    """_summary_

    Args:
        figure (list): (210,160,3)

    Returns:
        torch.Tensor: resized to (84,84,1)
    """
    figure = cv2.resize(np.array(figure,dtype=np.uint8),(84,84))
    figure = np.array(cv2.cvtColor(figure,cv2.COLOR_BGR2GRAY)).reshape(84,84,1).tolist()
    return figure
    

class ContinuousControl(object):
    """
    An Abstract Class for All Continuous Control Problems.
    """
    def __init__(self,config,state_dim,action_space) -> None:
        pass

    def action(self,s,t,e,episode):
        return None
    
    def update(self,record):
        return None
    
    def need_train(self,frame,stopped,e):
        return None
    
    def train(self, batch_size, gamma):
        return
    
    def save(self,dir_path,prefix=""):
        return

    def load(self,dir_path,prefix=""):
        return
    
    def set_train(self):
        return
    
    def set_test(self):
        return
    
    def need_sync(self):
        return False
    
    def sync(self):
        pass
    
    def episode_end(self):
        pass
    
class DeterministicPolicyNet(nn.Module):
    def __init__(self,in_dim,hid_dim,out_dim,act_func="sigmoid",last_act=None,mapping=None,device="cpu") -> None:
        super().__init__()
        self.device=device
        self.mlp=MLP(in_dim,hid_dim,out_dim,act_func,last_act)
        if mapping is not None:
            self.mapping = mapping
        else:
            self.mapping = self.default_mapping

    def forward(self,s):
        B = s.shape[0]
        if not isinstance(s,torch.Tensor):
            s = torch.tensor(s).reshape(B,-1)
        s = s.reshape(B,-1).to(device=self.device).float()
        return self.mapping(self.mlp(s))
    
    def default_mapping(Input):
        return Input
    
class StochasticPolicyNet(nn.Module):
    def __init__(self,in_dim,hid_dim,out_dim,act_func="sigmoid",last_act=None,mapping=None,device="cpu",eps=1e-4) -> None:
        super().__init__()
        self.device=device
        self.eps=eps
        self.mlp=MLP(in_dim,hid_dim[:-2],hid_dim[-2],act_func,act_func)
        self.mu=MLP(hid_dim[-2],[hid_dim[-1]],out_dim,last_act=last_act)
        self.lower=MLP(hid_dim[-2],[hid_dim[-1]],out_dim,last_act="softplus")
        if mapping is not None:
            self.mapping = mapping
        else:
            self.mapping = self.default_mapping

    def forward(self,s):
        B = s.shape[0]
        if not isinstance(s,torch.Tensor):
            s = torch.tensor(s).reshape(B,-1)
        s = s.reshape(B,-1).to(device=self.device).float()
        feature = self.mlp(s)
        mu = self.mapping(self.mu(feature)).float()
        n =mu.shape[1]
        lower = (self.lower(feature)+self.eps).reshape(B,-1,1).repeat(1,1,n) \
            * torch.eye(n).reshape(1,n,n).repeat(B,1,1).to(device=self.device).float()
        return mu, lower
    
    def default_mapping(Input):
        return Input

class DDPG(ContinuousControl):
    def __init__(self,config,state_dim,action_space) -> None:
        self.config=config
        self.state_dim=state_dim
        if not isinstance(action_space,torch.Tensor):
            self.action_space=torch.tensor(action_space)
        else:
            self.action_space=action_space
        input_size_sa=state_dim + action_space.shape[0]
        output_size_a=action_space.shape[0]
        self.buffer=ReplayBuffer(config["buffer_size"], state_dim)
        self.DQNet=DQNet(input_size_sa,self.config["dqnet"],1,"relu",None)
        self.targetDQNet=DQNet(input_size_sa,self.config["dqnet"],1,"relu",None)
        self.targetDQNet.load_state_dict(self.DQNet.state_dict().copy())
        self.policyNet=DeterministicPolicyNet(self.state_dim,self.config["policynet"],output_size_a,"relu","tanh",self._mapping)
        self.targetPolicyNet=DeterministicPolicyNet(self.state_dim,self.config["policynet"],output_size_a,"relu","tanh",self._mapping)
        self.targetPolicyNet.load_state_dict(self.policyNet.state_dict().copy())
        self.mseLoss=nn.MSELoss()
        self.DQNOptimizer=torch.optim.Adam(self.DQNet.parameters(),
                                           lr=self.config["actor_lr"],
                                           weight_decay=self.config["weight_decay"])
        self.policyOptimizer=torch.optim.Adam(self.policyNet.parameters(),lr=self.config["critic_lr"])
        self.train_count=0
        self.begin_noise = self.config["begin_noise"]
        self.end_noise = self.config["end_noise"]
        
    def noise_scheduler(self,t,e,episode):
        if e is not None and episode is not None:
            # training
            return (self.end_noise + (self.begin_noise-self.end_noise)*(1-e/(episode // 2))) if e < episode // 2 else self.end_noise
        else:
            return 0.

    @torch.no_grad()
    def action(self,s,t,e,episode):
        noise = self.noise_scheduler(t,e,episode)
        opt_a = self.policyNet(torch.tensor(s).reshape(1,-1))
        explore_a = opt_a + torch.randn_like(opt_a) * noise
        clipped_a = torch.clip(explore_a,min=self.action_space[:,0].reshape(1,-1),max=self.action_space[:,1].reshape(1,-1))
        return clipped_a.numpy().reshape(-1)
    
    def _mapping(self,out):
        centralized = out + (self.action_space[:,1]+self.action_space[:,0])/2
        mapped = centralized * ((self.action_space[:,1]-self.action_space[:,0])/2)
        return mapped
        
    def update(self,record):
        self.buffer.update(record)
        return
    
    def need_train(self, frame, stopped, e):
        return (frame % self.config["skip_frames_backward"]==0)
    
    def train(self, batch_size, gamma):
        self.train_count+=1
        # train the model
        # generate target
        self.DQNOptimizer.zero_grad()
        self.policyOptimizer.zero_grad()
        sample = self.buffer.sample(batch_size=batch_size)
        mask = 1-sample["terminated"]
        with torch.no_grad():
            ap = self.targetPolicyNet(sample["sp"])
            TD_target = sample["r"] + mask*gamma*self.targetDQNet(sample["sp"],ap)
        # regress DQN
        Q_hat = self.DQNet(sample["s"],sample["a"])
        loss = self.mseLoss(Q_hat,TD_target)
        loss.backward()
        self.DQNOptimizer.step()

        self.DQNOptimizer.zero_grad()
        self.policyOptimizer.zero_grad()
        # optimize PolicyNet
        Q_value = - self.DQNet(sample["s"],self.policyNet(sample["s"])).mean()
        Q_value.backward()
        self.policyOptimizer.step()
        return loss.item()
    
    def need_sync(self):
        if (self.train_count==self.config["skip_frames_sync"]):
            self.train_count=0
            return True
        else:
            return False
    
    def sync(self):
        self.targetDQNet.load_state_dict(self.DQNet.state_dict().copy())
        self.targetPolicyNet.load_state_dict(self.policyNet.state_dict().copy())
    
    def save(self,dir_path,prefix=""):
        os.makedirs(dir_path,exist_ok=True)
        torch.save(self.DQNet.state_dict(),os.path.join(dir_path,prefix+"DQNet.pkl"))
        torch.save(self.policyNet.state_dict(),os.path.join(dir_path,prefix+"policyNet.pkl"))
        return

    def load(self,dir_path,prefix=""):
        self.DQNet.load_state_dict(torch.load(os.path.join(dir_path,prefix+"DQNet.pkl")))
        self.policyNet.load_state_dict(torch.load(os.path.join(dir_path,prefix+"policyNet.pkl")))
        self.sync()
        return
    
    def set_train(self):
        self.begin_noise = self.config["begin_noise"]
        self.end_noise = self.config["end_noise"]
        self.DQNet.train()
        self.policyNet.train()
    
    def set_test(self):
        self.begin_noise = self.end_noise = 0.
        self.DQNet.eval()
        self.policyNet.eval()

class BaseDQN(ContinuousControl):
    def __init__(self,config,state_dim,action_space) -> None:
        self.config=config
        self.variant=self.config["variant"]
        self.obsType="ram" if isinstance(state_dim, int) else "rgb"
        self.state_dim=state_dim
        if not isinstance(action_space,torch.Tensor):
            self.action_space=torch.tensor(action_space)
        else:
            self.action_space=action_space
        self.device = "cpu" if self.obsType=="ram" else "cuda"
        if self.obsType=="ram":
            self.buffer=ReplayBuffer(config["buffer_size"], self.state_dim)
            self.DQNet=DQNet(self.state_dim,self.config["mlp"],self.action_space.shape[0],"relu",None)
            self.targetDQNet=DQNet(self.state_dim,self.config["mlp"],self.action_space.shape[0],"relu",None)
            self.targetDQNet.load_state_dict(self.DQNet.state_dict().copy())
        else:
            self.buffer=ReplayBuffer(config["buffer_size"], self.state_dim)
            self.DQNet=ConvDQNet(self.config["history"],self.config["cnn"],self.config["final"],self.action_space.shape[0],"relu",None,"cuda")
            self.targetDQNet=ConvDQNet(self.config["history"],self.config["cnn"],self.config["final"],self.action_space.shape[0],"relu",None,"cuda")
            self.DQNet.cuda()
            self.targetDQNet.cuda()
            self.targetDQNet.load_state_dict(self.DQNet.state_dict().copy())
        self.mseLoss=nn.MSELoss()
        self.DQNOptimizer=torch.optim.Adam(self.DQNet.parameters(),
                                           lr=self.config["lr"],
                                           weight_decay=self.config["weight_decay"])
        self.train_count=0
        self.batch_size = self.config["batch_size"]
        self.batch_action=self.action_space.reshape(1,-1).repeat(self.batch_size,1).to(self.device)
        self.action_buffer=StateBuffer(self.config["history"])
        self.s_buffer=StateBuffer(self.config["history"])
        self.sp_buffer=StateBuffer(self.config["history"])
        self.begin_noise = self.config["begin_noise"]
        self.end_noise = self.config["end_noise"]
        
    def noise_scheduler(self,t,e,episode):
        if e is not None and episode is not None:
            # training
            return (self.end_noise + (self.begin_noise-self.end_noise)*(1-e/(episode // 2))) if e < episode // 2 else self.end_noise
        else:
            return 0.05 # to prevent that it cannot begin the game

    @torch.no_grad()
    def eps_greedy(q_s: torch.Tensor, action_space: torch.Tensor, epsilon: float):
        """
        Args:
            q_s: the q function approximator (the default sample size is 1)
            action_space: action number (discrete)
            epsilon (float): the exploration rate 
        Returns:
            pi_a_s: p(a|s)
        """
        A=len(action_space)
        pi_a_s=torch.ones(A)*(epsilon/A)
        pi_a_s[q_s.reshape(-1).argmax().item()]+=1-epsilon
        pro=random.random()
        for a in range(len(pi_a_s)):
            pro -= pi_a_s[a]
            if pro <= 0.:
                return a
            
    @torch.no_grad()
    def greedy(q_s: torch.Tensor):
        """
        Args:
            q_a: the q function approximator.
        Returns:
            max_a q(s,a)
        """
        return torch.max(q_s,dim=-1)

    @torch.no_grad()
    def action(self,s,t,e,episode):
        if t < self.config["skip_first_frames"]:
            # for the first few frames, no action
            if self.obsType=="rgb":
                self.action_buffer.update(preproccess(s))
            return self.action_space[0]
        if self.obsType=="rgb":
            self.action_buffer.update(preproccess(s))
            s = self.action_buffer.get_image()
        state_shape=list(s.shape if isinstance(s,np.ndarray) or isinstance(s,torch.Tensor) else np.array(s).shape)
        state_shape.insert(0,1)
        eps = self.noise_scheduler(t,e,episode)
        return self.action_space[BaseDQN.eps_greedy(self.DQNet(torch.tensor(s).reshape(state_shape)),
                               self.action_space,
                               eps)]
    
        
    def update(self,record):
        # random drop
        if random.random()>(1/(self.config["history"])):
            return
        if self.obsType=="rgb":
            self.s_buffer.update(preproccess(record[0]))
            self.sp_buffer.update(preproccess(record[3]))
            if self.s_buffer.is_full():
                record[0]=self.s_buffer.get_image()
                record[3]=self.sp_buffer.get_image()
                self.buffer.update(record) 
        else:
            self.buffer.update(record)
        del record
        return
    
    def episode_end(self):
        self.action_buffer.clean()
        self.s_buffer.clean()
        self.sp_buffer.clean()
    
    def need_train(self, frame, stopped, e):
        return (frame % self.config["skip_frames_backward"]==0)
    
    def train(self, batch_size, gamma):
        batch_size = self.batch_size
        self.train_count+=1
        # train the model
        # generate target
        self.DQNOptimizer.zero_grad()
        sample = self.buffer.sample(batch_size=batch_size)
        # preprocess
        mask = 1-sample["terminated"]
        sample["r"] = sample["r"].to(self.device)
        sample["a"] = sample["a"].to(self.device)
        mask = mask.to(self.device)
        
        with torch.no_grad():
            if self.variant=="ddqn":
                TD_target = sample["r"] + mask*gamma*torch.gather(self.targetDQNet(sample["sp"]),
                                                                dim=-1,
                                                                index=torch.gather(self.batch_action,
                                                                                    dim=1,
                                                                                    index=BaseDQN.greedy(self.DQNet(sample["sp"]))[1].reshape(-1,1))).reshape(-1)
            else:
                TD_target = sample["r"] + mask*gamma*BaseDQN.greedy(self.targetDQNet(sample["sp"]))[0]        
        
        # regress DQN
        Q_hat = torch.gather(self.DQNet(sample["s"]),
                             dim=-1,
                             index=sample["a"].to(torch.int64).reshape(-1,1)).reshape(-1)
        loss = self.mseLoss(Q_hat,TD_target)
        loss.backward()
        self.DQNOptimizer.step()
        return loss.item()
    
    def need_sync(self):
        if (self.train_count==self.config["skip_frames_sync"]):
            self.train_count=0
            return True
        else:
            return False
    
    def sync(self):
        self.targetDQNet.load_state_dict(self.DQNet.state_dict().copy())
    
    def save(self,dir_path,prefix=""):
        os.makedirs(dir_path,exist_ok=True)
        torch.save(self.DQNet.state_dict(),os.path.join(dir_path,prefix+"DQNet.pkl"))
        return

    def load(self,dir_path,prefix=""):
        self.DQNet.load_state_dict(torch.load(os.path.join(dir_path,prefix+"DQNet.pkl")))
        self.sync()
        return
    
    def set_train(self):
        self.begin_noise = self.config["begin_noise"]
        self.end_noise = self.config["end_noise"]
        self.DQNet.train()
    
    def set_test(self):
        self.begin_noise = self.end_noise = 0.
        self.DQNet.eval()

class A2C(ContinuousControl):
    def __init__(self,config,state_dim,action_space) -> None:
        self.config=config
        self.state_dim=state_dim
        if not isinstance(action_space,torch.Tensor):
            self.action_space=torch.tensor(action_space)
        else:
            self.action_space=action_space
        input_size_s=state_dim
        output_size_a=action_space.shape[0]
        self.DVNet=DVNet(input_size_s,[32,16],1,"relu",None)
        self.policyNet=StochasticPolicyNet(self.state_dim,[32,32],output_size_a,"relu","tanh",self._mapping)
        self.DVNOptimizer=torch.optim.Adam(self.DVNet.parameters(),lr=self.config["lr"])
        self.policyOptimizer=torch.optim.Adam(self.policyNet.parameters(),lr=self.config["lr"])
        self.multivariateNormal=None
        if "gamma" in self.config:
            self.gamma=self.config["gamma"]
        else:
            self.gamma=0.99
        self.time_step=0.
        self.skip_frames_backward=self.config["skip_frames_backward"]
        self.recordBuffer = []

    def action(self,s,t,e,episode):
        mu, lower = self.policyNet(torch.tensor(s).reshape(1,-1))
        self.multivariateNormal=MultivariateNormal(loc=mu,scale_tril=lower)
        a = self.multivariateNormal.sample()
        return a.numpy().reshape(-1)
    
    def _mapping(self,out):
        centralized = out + (self.action_space[:,1]+self.action_space[:,0])/2
        mapped = centralized * ((self.action_space[:,1]-self.action_space[:,0])/2)
        return mapped
        
    def update(self,record):
        record[1]=(self.multivariateNormal.log_prob(
            torch.tensor(record[1]).to(device="cpu").reshape(self.action_space.shape[0],self.action_space.shape[0])))
        self.recordBuffer.append(record)
        self.time_step += 1.
        return
    
    def calc_grad(self):
        policyGradient=torch.tensor([0.]).to(device="cpu").float()
        valueGradient=torch.tensor([0.]).to(device="cpu").float()
                
        with torch.no_grad():
            R = 0. if self.recordBuffer[-1][-1] else self.DVNet(torch.tensor(self.recordBuffer[-1][3]).reshape(1,-1))
        for i in reversed(range(int(self.time_step))):
            R = self.recordBuffer[i][2] + self.gamma * R
            advantage=R-self.DVNet(torch.tensor(self.recordBuffer[i][0]).reshape(1,-1))
            policyGradient += float(advantage.item()) * self.recordBuffer[i][1]
            valueGradient += 0.5 * (advantage ** 2)
        return (valueGradient / self.time_step) - policyGradient
    
    def need_train(self, frame, stopped, e):
        return stopped or (self.time_step == self.skip_frames_backward)
    
    def train(self, batch_size, gamma):
        # train the model
        # generate target
        
        self.DVNOptimizer.zero_grad()
        self.policyOptimizer.zero_grad()

        # policyGradient shall be performed upon gradient ascent!
        loss = self.calc_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.DVNet.parameters(),10)
        torch.nn.utils.clip_grad_norm_(self.policyNet.parameters(),10)
        
        self.DVNOptimizer.step()
        self.policyOptimizer.step()
        
        self.time_step = 0.
        self.recordBuffer = []
        return loss.item()
    
    def sync_with_global(self,DVNet,policyNet):
        """
        Must input the copy of the state dicts.
        """
        self.DVNet.load_state_dict(DVNet) 
        self.policyNet.load_state_dict(policyNet)

    def get_grad(self):
        # train the model
        # generate target

        self.DVNet.zero_grad()
        self.policyNet.zero_grad()

        # policyGradient shall be performed upon gradient ascent!
        loss = self.calc_grad()
        loss.backward()

        tempDVNGradient = []
        tempPolicyGradient = []
        for selfParam in self.DVNet.parameters():
            tempDVNGradient.append(selfParam.grad.clone())
        for selfParam in self.policyNet.parameters():
            tempPolicyGradient.append(selfParam.grad.clone())

        grad = []
        grad.append(tempDVNGradient)
        grad.append(tempPolicyGradient)
        
        self.time_step = 0.
        self.recordBuffer = []
        return loss.item(), grad
    
    def save(self,dir_path,prefix=""):
        os.makedirs(dir_path,exist_ok=True)
        torch.save(self.DVNet.state_dict(),os.path.join(dir_path,prefix+"DVNet.pkl"))
        torch.save(self.policyNet.state_dict(),os.path.join(dir_path,prefix+"policyNet.pkl"))
        return

    def load(self,dir_path,prefix=""):
        self.DVNet.load_state_dict(torch.load(os.path.join(dir_path,prefix+"DVNet.pkl")))
        self.policyNet.load_state_dict(torch.load(os.path.join(dir_path,prefix+"policyNet.pkl")))
        return
    
    def set_train(self):
        self.DVNet.train()
        self.policyNet.train()
        return
    
    def set_test(self):
        self.DVNet.eval()
        self.policyNet.eval()
        return



class A3C(ContinuousControl):
    def __init__(self,config,state_dim,action_space) -> None:
        self.config=config
        self.state_dim=state_dim
        if not isinstance(action_space,torch.Tensor):
            self.action_space=torch.tensor(action_space)
        else:
            self.action_space=action_space
        self.episode=self.config["episode"]
        self.world_name=self.config["world_name"]
        self.maxT=self.config["maxT"]
        self.gamma=self.config["gamma"]
        self.num_workers=self.config["num_workers"]
        self._initialize()
        self.process=[]

    def _initialize(self):
        input_size_s=self.state_dim
        output_size_a=self.action_space.shape[0]
        self.DVNet=DVNet(input_size_s,[32,16],1,"relu",None)
        self.policyNet=StochasticPolicyNet(self.state_dim,[32,32],output_size_a,"relu","tanh",self._mapping)
        self.DVNOptimizer=torch.optim.Adam(self.DVNet.parameters(),lr=self.config["lr"])
        self.policyOptimizer=torch.optim.Adam(self.policyNet.parameters(),lr=self.config["lr"])

    def _mapping(self,out):
        centralized = out + (self.action_space[:,1]+self.action_space[:,0])/2
        mapped = centralized * ((self.action_space[:,1]-self.action_space[:,0])/2)
        return mapped
    
    def train(self,batch_size,gamma):
        episode = self.episode
        paramQueues = []
        gradQueues = mp.Queue(10*self.num_workers)
        notFinised = [1 for _ in range(self.num_workers)]
        for j in range(self.num_workers):
            paramQueues.append(mp.Queue(10))
            model=model_parser("A2C",self.config,self.state_dim,self.action_space)
            env=gym.make(self.world_name,self.maxT,render_mode=None)
            initial_weights=[self.DVNet.state_dict().copy(),self.policyNet.state_dict().copy()]
            p = mp.Process(target=A3C._train,
                           args=(j,
                                 initial_weights,
                                 paramQueues[j],
                                 gradQueues,
                                 env,
                                 model,
                                 episode))
            p.start()
            self.process.append(p)
            
        cnt = 0
        best_score = -100000.
        reward_list = []
        while notFinised:
            cnt += 1
            gradDict = gradQueues.get()
            self.DVNOptimizer.zero_grad()
            self.policyOptimizer.zero_grad()
            id, gradList, sign = gradDict
            for k,net in enumerate((self.DVNet.parameters(),self.policyNet.parameters())):
                for selfParam, childParam in zip(net,gradList[k]):
                    selfParam._grad=childParam

            torch.nn.utils.clip_grad_norm_(self.DVNet.parameters(),10)
            torch.nn.utils.clip_grad_norm_(self.policyNet.parameters(),10)

            self.DVNOptimizer.step()
            self.policyOptimizer.step()
            paramQueues[id].put([self.DVNet.state_dict().copy(),
                                      self.policyNet.state_dict().copy()])
            
            notFinised[id] = sign
            
            if cnt % (100 * self.maxT) == 0:
                avg_score=self._test()
                print("Average Reward: {:.3f}".format(avg_score))
                reward_list.append(avg_score)
                if avg_score > best_score:
                    best_score = avg_score
                    model.save(dir_path=os.path.join(".","ckpts","A3C"))
            
            if sum(notFinised)==0:
                print("All children finished!")
                break

        for p in self.process:
            p.join()
            
        return reward_list

    def save(self,dir_path,prefix=""):
        os.makedirs(dir_path,exist_ok=True)
        torch.save(self.DVNet.state_dict(),os.path.join(dir_path,prefix+"DVNet.pkl"))
        torch.save(self.policyNet.state_dict(),os.path.join(dir_path,prefix+"policyNet.pkl"))
        return

    def load(self,dir_path,prefix=""):
        self.DVNet.load_state_dict(torch.load(os.path.join(dir_path,prefix+"DVNet.pkl")))
        self.policyNet.load_state_dict(torch.load(os.path.join(dir_path,prefix+"policyNet.pkl")))
        return
    
    def _test(self):
        from vis import test
        model=model_parser("A2C",self.config,self.state_dim,self.action_space)
        model.sync_with_global(self.DVNet.state_dict().copy(),self.policyNet.state_dict().copy())
        model.set_test()
        avg_score=test(self.world_name,model,self.action_space,maxT=self.maxT,test_times=30,render_mode=None)
        return avg_score
    
    def _train(id,init_weights,paramQueue,gradQueue,env,model,episode):

        # initializes the weights.
        model.sync_with_global(init_weights[0],init_weights[1])

        for e in range(episode):
            # set model to train!
            model.set_train()

            s, info = env.reset()
            frame = 0
            t = 0.
            total_reward = 0.
            loss_list = []
            while True:
                t+=1
                # agent policy that uses the observation and info
                a = model.action(s,t,e,episode)
                # get the s_{t+1}, r_t, end or not from the env
                sp, r, terminated, truncated, info = env.step(a)
                # update buffer
                model.update([s.tolist(),a.tolist(),r,sp.tolist(),terminated])
                # update state
                s=sp
                frame += 1
                total_reward += r

                if model.need_train(frame,terminated or truncated,e):
                    loss, grad = model.get_grad()
                    loss_list.append(loss)
                    gradQueue.put([id,grad,int((e!=episode-1) or (not (terminated or truncated)))]) # The last sign is the finished sign!

                    new_weights = paramQueue.get()

                    model.sync_with_global(new_weights[0],new_weights[1])

                # logging
                if terminated or truncated:
                    s, info = env.reset()
                    print("Episode: {}, Loss: {:.3f}, Terminated Steps: {}, Total Reward: {:.3f}".format(e,sum(loss_list)/len(loss_list),t,total_reward))
                    break