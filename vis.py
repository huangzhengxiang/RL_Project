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
                score.append(total_reward)
                break
    env.close()
    return sum(score)/len(score)

if __name__=="__main__":
    # initialization
    parser=argparse.ArgumentParser()
    parser.add_argument("--world",type=str,default="Pendulum-v1")
    parser.add_argument("--test_times",type=int,default=10)
    parser.add_argument("--render_mode",type=str,default=None)
    args=parser.parse_args()
    maxT=200
    world_name=args.world
    test_times=args.test_times
    render_mode=args.render_mode
    # test the agent
    model_list = os.listdir(vis_dir)
    v_list = []

    # 3. build world
    env=gym.make(world_name,maxT,render_mode=None)
    state_dim=3
    action_space=np.array([[-2.,2.]]) # continuous space

    loc_space = (torch.tensor([range(0,180+1)])*0.01-1.2).reshape(-1,1,1).repeat(1,14+1,1)
    speed_space = (torch.tensor([range(0,14+1)])*0.01-0.07).reshape(1,-1,1).repeat(180+1,1,1)
    observation_space = torch.cat([loc_space,speed_space],dim=-1).reshape(-1,2)

    for model_path in model_list:
        model_name = model_path.split("_")[0]
        config_path = os.path.join(config_dir,"{}.yaml".format(model_name))
        with open(config_path,"rt") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            f.close()
        # 4. initialize model
        if (model_name=="A3C"):
            model_name="A2C"
        model=model_parser(model_name,config,state_dim,action_space)
        model.load(os.path.join(vis_dir,model_path))
        # q_mean=Qmean(dqnet,observation_space,action_space)
        # v_mean=Vmean(dqnet,observation_space,action_space)
        # v_list.append(all_v(dqnet,observation_space,action_space).numpy())
        # print("average q value: {:.3f}, v value: {:.3f}".format(q_mean,v_mean))
        avg_score=test(world_name,model,action_space,maxT,test_times=test_times,render_mode=render_mode)
        print("model: {}".format(os.path.join(vis_dir,model_path)))
        print("average steps: {:.2f}".format(avg_score))
    # vis_V(v_list,["ddqn","dqn"])
    # vis_V_heatmap(v_list,["ddqn","dqn"])