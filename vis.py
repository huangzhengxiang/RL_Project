import gymnasium as gym
import torch
from agent import greedy
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model import model_parser
import yaml

config_dir = os.path.join(".","config")
vis_dir = os.path.join(".","best_ckpt")

def genSpace(env: gym.Env):
    """
    Generate the state space and the action space!
    """
    if len(env.observation_space.shape)==1:
        # The input is a single vector!
        state_dim=env.observation_space.shape[0]
    else:
        # The input is a picture!
        state_dim=env.observation_space.shape
    if type(env.action_space) == gym.spaces.Discrete:
        # The action is discrete!
        action_space = np.arange(env.action_space.n)
    else:
        # The action is continuous! Doesn't consider inf condition, but not a big deal!
        action_space = np.zeros([env.action_space.shape[0],2])
        action_space[:,0] = env.action_space.low
        action_space[:,1] = env.action_space.high
    return state_dim, action_space

def Qmean(model,observation_space,action_space) -> float:
    N = observation_space.shape[0]
    mean_a = []
    for a in range(len(action_space)):
        mean_a.append(model(observation_space,torch.tensor(action_space[a]).reshape(1,-1).repeat(N,1)).flatten().mean().item())
    return sum(mean_a) / len(mean_a)

def Vmean(model,observation_space,action_space) -> float:
    return all_v(model,observation_space,action_space).flatten().mean()

def all_v(model,observation_space,action_space):
    return greedy(model,observation_space,action_space)[0]

def vis_V(v_vector,n_vector):
    for v in v_vector:
        plt.plot(v,linewidth=1)
    plt.legend(n_vector)
    plt.savefig("./demo/vis_V.png")
    plt.close()

def vis_V_heatmap(v_vector,n_vector):
    for j in range(len(v_vector)):
        v = v_vector[j]
        sns.heatmap(v.reshape(181,15).T)
        plt.savefig("./demo/{}.png".format(n_vector[j]))
        plt.close()
@torch.no_grad()
def test(env_id,model,action_space,maxT=1000,test_times=10,render_mode=None) -> float:
    env = gym.make(env_id,maxT,render_mode=render_mode)
    score = []
    model.set_test()
    for _ in range(test_times):
        s, info = env.reset()
        total_reward = 0.
        for t in range(maxT):
            # agent policy that uses the observation and info
            a = model.action(s,t,None,None)
            # get the s_{t+1}, r_t, end or not from the env
            sp, r, terminated, truncated, info = env.step(a)
            # update state
            s=sp
            total_reward += r
            if terminated or truncated:
                # if terminated:
                #     print("termination steps: {}".format(t))
                # else:
                #     print("Truncated!")
                model.episode_end()
                score.append(total_reward)
                break
    env.close()
    return sum(score)/len(score)

if __name__=="__main__":
    # initialization
    parser=argparse.ArgumentParser()
    parser.add_argument("--test_times",type=int,default=10)
    parser.add_argument("--render_mode",type=str,default=None)
    args=parser.parse_args()
    test_times=args.test_times
    render_mode=args.render_mode
    # test the agent
    env_list = os.listdir(vis_dir)

    for world_name in env_list:
        model_list = os.listdir(os.path.join(vis_dir,world_name))
        # 3. build world
        env=gym.make(world_name,render_mode=None)
        state_dim, action_space = genSpace(env)
        
        for model_path in model_list:
            model_name = model_path.split("_")[0]
            maxT = 1000 if model_name=="DDPG" or model_name=="A2C" or model_name=="A3C" else 10000
            env=gym.make(world_name,maxT,render_mode=None)
            config_path = os.path.join(config_dir,world_name,"{}.yaml".format(model_name))
            with open(config_path,"rt") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
                f.close()
            # 4. initialize model
            if (model_name=="A3C"):
                model_name="A2C"
            model=model_parser(model_name,config,state_dim,action_space)
            model.load(os.path.join(vis_dir,world_name,model_path))
            avg_score=test(world_name,model,action_space,maxT,test_times=test_times,render_mode=render_mode)
            print("model: {}".format(os.path.join(vis_dir,world_name,model_path)))
            print("average reward: {:.2f}".format(avg_score))