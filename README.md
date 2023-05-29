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

The tutorial I referred to is on https://gymnasium.farama.org/environments/classic_control/pendulum/

Also, I referred to https://gymnasium.farama.org/tutorials/training_agents/blackjack_tutorial/ in order to build an agent.

### 1. train the agent

Train the agent with DDPG.
~~~
python main.py --mtype DDPG --episode 1000 --batch_size 64
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
python vis.py --render_mode human
~~~

Calculate the average score.

~~~
python vis.py --test_times 100
~~~

### "Human Expert"
BreakoutNoFrameskip-v4: 30
~~~
python human.py --env_name BreakoutNoFrameskip-v4
~~~