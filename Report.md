[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Implementation of Learning Algorithm

It has been shown in `Navigation.ipynb` that using an agent which takes action randomly will not solve the problem. A more intelligent agent is therefore needed to achieve an average score of +13 over 100 consecutive episodes. We employ Q-Learning that ims at finding an optimal policy, i.e., a policy that maximizes the reward for the agent through interacting with the environment and recording observations. 

### Future Direction
- It would be interesting to implement a double DQN, a dueling DQN, and/or prioritized experience replay and compare with the current implementation to identify the best approach.
- It would also be interesting to implement the agent that can learn from raw pixels. 

