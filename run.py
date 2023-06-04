import gymnasium as gym
import torch
from model import model_parser, get_model_name
from vis import test, genSpace
import argparse
import os
import numpy as np
import yaml

SEED=8192
config_dir = os.path.join(".","config")
logger_dir = os.path.join(".","logger")

if __name__=="__main__":
    # 0. random seeds
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(SEED)
    # 1. parser 
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name",type=str,default="Hopper-v2")
    parser.add_argument("--episode",type=int,default=3000)
    parser.add_argument("--maxT",type=int,default=1000)
    parser.add_argument("--gamma",type=float,default=0.99)
    parser.add_argument("--batch_size",type=int,default=64)
    parser.add_argument("--mtype",type=str,default="",help="DQN or DDPG or A2C or A3C")
    args=parser.parse_args()
    world_name=args.env_name
    episode=args.episode
    maxT=args.maxT
    gamma=args.gamma
    batch_size=args.batch_size
    mtype=args.mtype
    
    # 2. build world
    env=gym.make(world_name,maxT,render_mode=None)
    state_dim, action_space = genSpace(env)
    
    # 3. config file and logger file
    mtype=get_model_name(mtype, action_space)
    print(mtype)
    config_path = os.path.join(config_dir,world_name,"{}.yaml".format(mtype))
    with open(config_path,"rt") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(config)
    os.makedirs(os.path.join(logger_dir,world_name),exist_ok=True)
    logger_path = os.path.join(logger_dir,world_name,"{}.txt".format(mtype))
    logger = open(logger_path,"wt")

    # 4. initialize model
    config["gamma"]=gamma
    config["world_name"]=world_name
    config["maxT"]=1000 if mtype=="DDPG" or mtype=="A2C" or mtype=="A3C" else maxT # The maxT for continous control is always 1000.
    config["episode"]=episode
    model=model_parser(mtype,config,state_dim,action_space)
    maxT = config["maxT"]
    env=gym.make(world_name,maxT,render_mode=None)

    # 5. A3C parallel training is handled inside its class
    if mtype=="A3C":
        model.set_train()
        reward_list = model.train(None,None)
        [print("{:.2f}".format(reward),file=logger) for reward in reward_list]
        logger.close()
        model.load(dir_path=os.path.join(".","ckpts",world_name,"A3C"))
        best_reward = model._test()
        print("Best Reward for A3C: {:.3f}".format(best_reward))
        print("Finish Training!")
        exit()

    # 5. train the agent
    best_score = -1000000.
    loss_list = []
    step_list = []
    reward_list = []
    try:
        for e in range(episode):
            # set model to train!
            model.set_train()

            s, info = env.reset()
            frame = 0
            avg_loss = 0.
            itr = 0.
            total_reward = 0.
            for t in range(maxT):
                # agent policy that uses the observation and info
                a = model.action(s,t,e,episode)
                # get the s_{t+1}, r_t, end or not from the env
                sp, r, terminated, truncated, info = env.step(a)
                # update buffer
                model.update([s.tolist(),
                            a.tolist() if isinstance(a,np.ndarray) else a,
                            r,sp.tolist(),terminated])
                # update state
                s=sp
                frame += 1
                total_reward += r

                if model.need_train(frame,terminated or truncated,e):
                    loss = model.train(batch_size, gamma)
                    avg_loss += loss
                    itr += 1
                    # For every synT steps, synchronize 2 nets.
                    if model.need_sync():
                        model.sync()

                # logging
                if terminated or truncated:
                    s, info = env.reset()
                    model.episode_end()
                    print("Episode: {}, Loss: {:.4f}, Terminated Steps: {}, Total Reward: {:.3f}".format(e,avg_loss/itr,t,total_reward))
                    assert avg_loss/itr < 100_000 # weight decay dominate the loss
                    loss_list.append(avg_loss/itr)
                    step_list.append(t)
                    reward_list.append(total_reward)
                    break
            
            if e % 20 == 0:
                model.set_test()
                avg_score=test(world_name,model,action_space,maxT=maxT,test_times=2,render_mode=None)
                print("Episode: {}, Average Reward: {:.3f}".format(e,avg_score))
                if avg_score > best_score:
                    best_score = avg_score
                    model.save(dir_path=os.path.join(".","ckpts",world_name,mtype))
    finally:
        try:
            # 4. save the logger
            env.close()
            [print("{} {:.2f} {} {:.2f}".format(j,loss_list[j],step_list[j],reward_list[j]),file=logger) for j in range(episode)]

        finally:

            # 5. test the agent
            model.load(dir_path=os.path.join(".","ckpts",world_name,mtype))
            avg_score=test(world_name,model,action_space,maxT=maxT,test_times=10,render_mode=None)
            print("average score: {:.2f}".format(avg_score))