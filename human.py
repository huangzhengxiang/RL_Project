import gymnasium as gym
import argparse
import os
import numpy as np
import yaml
import pygame

descList = [
    "NOOP",
    "FIRE",
    "UP",
    "RIGHT",
    "LEFT",
    "DOWN",
    "UPRIGHT",
    "UPLEFT",
    "DOWNRIGHT",
    "DOWNLEFT",
    "UPFIRE",
    "RIGHTFIRE",
    "LEFTFIRE",
    "DOWNFIRE",
    "UPRIGHTFIRE",
    "UPLEFTFIRE",
    "DOWNRIGHTFIRE",
    "DOWNLEFTFIRE"
    ]

keyList = [
    pygame.K_s,
    pygame.K_SPACE,
    pygame.K_w,
    pygame.K_d,
    pygame.K_a,
    pygame.K_x,
    pygame.K_e,
    pygame.K_q,
    pygame.K_c,
    pygame.K_z,
    pygame.K_KP_8,
    pygame.K_KP_6,
    pygame.K_KP_4,
    pygame.K_KP_2,
    pygame.K_KP_9,
    pygame.K_KP_7,
    pygame.K_KP_3,
    pygame.K_KP_1
]

SEED=8192
maxT=1000
logger_dir = os.path.join(".","logger")
def get_action() -> int:
    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                for key in keyList:
                    if event.key==key:
                        return keyList.index(key)
            else:
                if hasattr(event,"text"):
                    for key in keyList:
                        if key < 255:
                            if event.text==chr(key):
                                return keyList.index(key)

if __name__=="__main__":
    # 0. random seeds
    np.random.seed(SEED)
    # 1. parser 
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name",type=str,default="Pendulum-v1")
    args=parser.parse_args()
    world_name=args.env_name
    # 2. config file and logger file
    os.makedirs(logger_dir,exist_ok=True)
    logger_path = os.path.join(logger_dir,"{}.txt".format("human"))
    logger = open(logger_path,"wt")

    # 3. build world
    env=gym.make(id=world_name,render_mode="human")
    state_dim=3
    action_space=np.array([[-2.,2.]]) # continuous space

    # 5. train the agent
    best_score = -1000000.
    loss_list = []
    step_list = []
    reward_list = []
    for e in range(1):
        # set model to train!

        s, info = env.reset()
        frame = 0
        avg_loss = 0.
        itr = 0.
        total_reward = 0.
        while True:
            # agent policy that uses the observation and info
            a = get_action()
            # get the s_{t+1}, r_t, end or not from the env
            sp, r, terminated, truncated, info = env.step(a)
            # update state
            s=sp
            frame += 1
            total_reward += r

            # logging
            if terminated or truncated:
                s, info = env.reset()
                print("Episode: {}, Total Reward: {:.3f}".format(e,total_reward))
                reward_list.append(total_reward)
                break

    # 4. save the logger
    env.close()
    [print("{} {:.2f}".format(j,reward_list[j]),file=logger) for j in range(1)]