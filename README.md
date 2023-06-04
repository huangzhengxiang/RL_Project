# RL_Project
This is the final project for Reinforcement Learning (RL)
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

Train the agent with A3C.

~~~
python main.py --mtype A3C --episode 1000
~~~

Maybe you can try A2C if you wish.

~~~
python main.py --mtype A2C --episode 3000
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

### "Human Expert"
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