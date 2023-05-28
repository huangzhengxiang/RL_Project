import gymnasium as gym
import torch
from model import model_parser
from vis import test
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
    parser.add_argument("--world",type=str,default="Pendulum-v1")
    parser.add_argument("--episode",type=int,default=1000)
    parser.add_argument("--maxT",type=int,default=200)
    parser.add_argument("--gamma",type=float,default=0.99)
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--mtype",type=str,default="DDPG",help="DDPG or A2C or A3C")
    args=parser.parse_args()
    world_name=args.world
    episode=args.episode
    maxT=args.maxT
    gamma=args.gamma
    batch_size=args.batch_size
    mtype=args.mtype
    # 2. config file and logger file
    config_path = os.path.join(config_dir,"{}.yaml".format(args.mtype))
    with open(config_path,"rt") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(config)
    os.makedirs(logger_dir,exist_ok=True)
    logger_path = os.path.join(logger_dir,"{}.txt".format(mtype))
    logger = open(logger_path,"wt")

    # 3. build world
    env=gym.make(world_name,maxT,render_mode=None)
    state_dim=3
    action_space=np.array([[-2.,2.]]) # continuous space

    # 4. initialize model
    config["gamma"]=gamma
    config["world_name"]=world_name
    config["maxT"]=maxT
    config["episode"]=episode
    model=model_parser(mtype,config,state_dim,action_space)

    # 5. A3C parallel training is handled inside its class
    if mtype=="A3C":
        model.set_train()
        reward_list = model.train(None,None)
        [print("{:.2f}".format(reward),file=logger) for reward in reward_list]
        logger.close()
        model.load(dir_path=os.path.join(".","ckpts","A3C"))
        best_reward = model._test()
        print("Best Reward for A3C: {:.3f}".format(best_reward))
        print("Finish Training!")
        exit()

    # 5. train the agent
    best_score = -1000000.
    loss_list = []
    step_list = []
    reward_list = []
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
            model.update([s.tolist(),a.tolist(),r,sp.tolist(),terminated])
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
                print("Episode: {}, Loss: {:.4f}, Terminated Steps: {}, Total Reward: {:.3f}".format(e,avg_loss/itr,t,total_reward))
                loss_list.append(avg_loss/itr)
                step_list.append(t)
                reward_list.append(total_reward)
                break
        
        if e % 40 == 0:
            model.set_test()
            avg_score=test(world_name,model,action_space,maxT=maxT,test_times=20,render_mode=None)
            print("Episode: {}, Average Reward: {:.3f}".format(e,avg_score))
            if avg_score > best_score:
                best_score = avg_score
                model.save(dir_path=os.path.join(".","ckpts",mtype))

    # 4. save the logger
    env.close()
    [print("{} {:.2f} {} {:.2f}".format(j,loss_list[j],step_list[j],reward_list[j]),file=logger) for j in range(episode)]


    # 5. test the agent
    model.load(dir_path=os.path.join(".","ckpts",mtype))
    avg_score=test(world_name,model,action_space,maxT=maxT,test_times=100,render_mode=None)
    print("average score: {:.2f}".format(avg_score))