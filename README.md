# RL Project
<b>Author: 黄正翔 (Huang, Zhengxiang)</b>

This is the final project for Reinforcement Learning (RL)

Demo video clips:

Ant

<img src="video\Ant-v2\DDPG.gif" width=420>

HalfCheetah

<img src="video\HalfCheetah-v2\DDPG.gif" width=420>

Hopper

<img src="video\Hopper-v2\DDPG.gif" width=420>

Humanoid

<img src="video\Humanoid-v2\DDPG.gif" width=420>

VideoPinball

<img src="video\VideoPinball-ramNoFrameskip-v4\DQN.gif" width=420 height=320>

Boxing

<img src="video\BoxingNoFrameskip-v4\DQN.gif" width=420 height=320>


### 0. Environment
I used OPENAI gymnasium as the environment, since gym is not officially supported now.

~~~
pip install gymnasium
pip install gymnasium[atari]
pip install gymnasium[accept-rom-license]
pip install gymnasium[mujoco]
~~~

You also need opencv
~~~
pip install opencv-python
~~~

The tutorial I referred to is on https://gymnasium.farama.org/environments/classic_control/pendulum/

Also, I referred to https://gymnasium.farama.org/tutorials/training_agents/blackjack_tutorial/ in order to build an agent.

### 1. train the agent
For the discrete discision space games, we recommend to use DQN.
Train the agent with DQN.
~~~
python run.py --env_name VideoPinball-ramNoFrameskip-v4
python run.py --env_name BreakoutNoFrameskip-v4
python run.py --env_name PongNoFrameskip-v4
python run.py --env_name BoxingNoFrameskip-v4
~~~
~~~
python run.py --env_name Hopper-v2
python run.py --env_name Humanoid-v2
python run.py --env_name HalfCheetah-v2
python run.py --env_name Ant-v2
~~~

### 2. visualize the results
See the results illustration.
~~~
python vis.py --render_mode human --test_times 1
~~~

Calculate the average score.

~~~
python vis.py --test_times 10
~~~

### 3. "Human Expert"
BoxingNoFrameskip-v4: 5
BreakoutNoFrameskip-v4: 30
PongNoFrameskip-v4: -4.5
VideoPinball-ramNoFrameskip-v4: 5210
~~~
python human.py --env_name VideoPinball-ramNoFrameskip-v4
python human.py --env_name BreakoutNoFrameskip-v4
python human.py --env_name PongNoFrameskip-v4
python human.py --env_name BoxingNoFrameskip-v4
~~~

### 4. Other Utils

~~~
python vis.py --render_mode rgb_array --test_times 1 --env_name VideoPinball-ramNoFrameskip-v4
python vis.py --render_mode rgb_array --test_times 1 --env_name BoxingNoFrameskip-v4
python vis.py --env_name BreakoutNoFrameskip-v4 --render_mode human --test_times 1
python vis.py --env_name Humanoid-v2 --render_mode human --test_times 1
python run.py --env_name VideoPinball-ramNoFrameskip-v4 --load
~~~  
