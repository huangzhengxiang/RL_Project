import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from agent import ReplayBuffer
import os
import torch.multiprocessing as mp
import gymnasium as gym

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
    def __init__(self,in_dim,hid_dim,out_dim,act_func="sigmoid",last_act=None,device="cpu") -> None:
        super().__init__()
        self.device=device
        self.mlp=MLP(in_dim,hid_dim,out_dim,act_func,last_act)

    def forward(self,s,a):
        B = s.shape[0]
        if not isinstance(s,torch.Tensor):
            s = torch.tensor(s).reshape(B,-1)
        if not isinstance(a,torch.Tensor):
            a = torch.tensor(a).reshape(B,-1)
        s = s.float().reshape(B,-1)
        a = a.float().reshape(B,-1)
        x = torch.cat([s,a],dim=-1).to(device=self.device).float()
        return self.mlp(x).reshape(-1)

class DQNet(nn.Module):
    def __init__(self,in_dim,hid_dim,out_dim,act_func="sigmoid",last_act=None,device="cpu") -> None:
        super().__init__()
        self.device=device
        self.mlp=MLP(in_dim,hid_dim,out_dim,act_func,last_act)

    def forward(self,s,a):
        B = s.shape[0]
        if not isinstance(s,torch.Tensor):
            s = torch.tensor(s).reshape(B,-1)
        if not isinstance(a,torch.Tensor):
            a = torch.tensor(a).reshape(B,-1)
        s = s.float().reshape(B,-1)
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

def model_parser(model_type,config,state_dim,action_space):
    if (model_type=="DDPG"):
        return DDPG(config,state_dim,action_space)
    elif (model_type=="A2C"):
        return A2C(config,state_dim,action_space)
    elif (model_type=="A3C"):
        return A3C(config,state_dim,action_space)
    else:
        print("Unsupported Model Error!")

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
        self.DQNet=DQNet(input_size_sa,[32,16],1,"relu",None)
        self.targetDQNet=DQNet(input_size_sa,[32,16],1,"relu",None)
        self.targetDQNet.load_state_dict(self.DQNet.state_dict().copy())
        self.policyNet=DeterministicPolicyNet(self.state_dim,[32,16],output_size_a,"relu","tanh",self._mapping)
        self.targetPolicyNet=DeterministicPolicyNet(self.state_dim,[32,16],output_size_a,"relu","tanh",self._mapping)
        self.targetPolicyNet.load_state_dict(self.policyNet.state_dict().copy())
        self.mseLoss=nn.MSELoss()
        self.DQNOptimizer=torch.optim.Adam(self.DQNet.parameters(),lr=self.config["lr"])
        self.policyOptimizer=torch.optim.Adam(self.policyNet.parameters(),lr=self.config["lr"])
        self.train_count=0
        self.noise = self.config["noise_sigma"]

    @torch.no_grad()
    def action(self,s,t,e,episode):
        # noise = (self.noise) / 2 if episode is not None and  (e >= episode / 3) else self.noise
        noise = self.noise
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
            ap = self.targetPolicyNet(sample["s"])
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
        self.noise = self.config["noise_sigma"]
        self.DQNet.train()
        self.policyNet.train()
    
    def set_test(self):
        self.noise = 0.
        self.DQNet.eval()
        self.policyNet.eval()

class BaseDQN(ContinuousControl):
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
        self.DQNet=DQNet(input_size_sa,[32,16],1,"relu",None)
        self.targetDQNet=DQNet(input_size_sa,[32,16],1,"relu",None)
        self.targetDQNet.load_state_dict(self.DQNet.state_dict().copy())
        self.policyNet=DeterministicPolicyNet(self.state_dim,[32,16],output_size_a,"relu","tanh",self._mapping)
        self.targetPolicyNet=DeterministicPolicyNet(self.state_dim,[32,16],output_size_a,"relu","tanh",self._mapping)
        self.targetPolicyNet.load_state_dict(self.policyNet.state_dict().copy())
        self.mseLoss=nn.MSELoss()
        self.DQNOptimizer=torch.optim.Adam(self.DQNet.parameters(),lr=self.config["lr"])
        self.policyOptimizer=torch.optim.Adam(self.policyNet.parameters(),lr=self.config["lr"])
        self.train_count=0
        self.noise = self.config["noise_sigma"]

    @torch.no_grad()
    def action(self,s,t,e,episode):
        # noise = (self.noise) / 2 if episode is not None and  (e >= episode / 3) else self.noise
        noise = self.noise
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
            ap = self.targetPolicyNet(sample["s"])
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
        self.noise = self.config["noise_sigma"]
        self.DQNet.train()
        self.policyNet.train()
    
    def set_test(self):
        self.noise = 0.
        self.DQNet.eval()
        self.policyNet.eval()

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