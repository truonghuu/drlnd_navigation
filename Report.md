[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

 In this project, we use Deep Q-Network to develop an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

### Implementation of Q-Learning Algorithm

It has been shown in `Navigation.ipynb` that using an agent which takes action randomly does not solve the problem. A more intelligent agent is therefore needed to achieve an average score of +13 over 100 consecutive episodes. We employ Q-Learning that aims at finding an optimal policy, i.e., a policy that maximizes the scores of the agent.   

The main component of Q-Learning algorithm is the `Q-table` where each element `Q(s,a)` is the maximum expected future reward for a pair of state `s` and action `a`. In other words, given the initial state `s`, `Q(s,a)` is the maximum expected future reward if the agent takes action `a`. 

At the begining of learning, `Q-table` is initialized as a zero matrix, as the agent has not interacted with the environment. The values in `Q-table` will be updated in each learning episode using the updating function as defined below.

<img src="https://github.com/truonghuu/drlnd_navigation/blob/master/figures/q_function.png" width="50%" align="top-left" alt="" title="q_function.png" />
Q-Function ([Source](https://towardsdatascience.com/a-beginners-guide-to-q-learning-c3e2a30a653c))

In this function, learning rate (alpha) defines how much we accept the new value vs the old value. Discount factor (gamma) is used to balance the immediate and future reward. `R(s,a)` is the reward that the agent receives when taking an action at a certain state. 
 
#### Epsilon-greedy Algorithm

At the beginning, the agent randomly chooses actions since it does not know anything about the environment. As it interacts with the environment, it learns some actions that maximize its expected rewards. Thus, it is reasonably that the agent will keep choosing the same action for a state as that action maximizes its expected reward. However, the agent could fall into a local optimum that prevents it from exploring other actions that might result in a higher expected reward. 

To address this problem, we employ an **𝛆-greedy algorithm**. At every state of the environment, the agent takes a random action with some probability epsilon `𝛜` and takes the action that maximizes the expected reward with probability (1-𝛜). To allow the agent to explore the environment at the beginning and exploit its experience when it is more confident, the value of epsilon is gradually reduced over time.

The logic of 𝛆-greedy algorithm is implemented as part of the `agent.act()` method [here](https://github.com/truonghuu/drlnd_navigation/blob/master/dqn_agent.py) in `dqn_agent.py` of the source code.


#### Deep Q-Network (DQN)

For most problems, it is impractical to represent Q-Function as a table that contains values for each pair of state and action. On one hand, it is because of the large number of states and actions of the problems. On the other hand, most of problems have continuous values for environment states. To overcome this problem, we train a neural network with parameter w to estimate Q-values, i.e., `F(s,a,w) ≈ Q(s,a)`. This is basiscally a regression problem in which the input is the state of the environment and output is the estimated Q-values. The loss function of the training is defined as mean squared error of the estimated Q-value and the target Q-value, which is defined as `R(s,a) + gamma * max Q'(s', a')`. By minimizing the loss function, we train the model to converge to the maximum expected reward.

In actual implementation, we use 2 neural networks for learning: one is for the prediction network and one for the target network. The target network has the same architecture as the prediction network but with paramters updated less frequently compared to the prediction one. In other words, after every K training iterations (K is a hyperparameter), the parameters of the prediction network are copied to the target network.
   
The neural network architecture used for this project can be found [here](https://github.com/truonghuu/drlnd_navigation/blob/master/model.py) in the `model.py` file of the source code. It is a fully-connected neural network that contains an input layer of 37 nodes, 2 hidden layers with 64 nodes each and an output  with 4 nodes, corresponding to 4 actions.


#### Experience Replay

However, unlike supervised learning where input data is independent and identically distributed, in reinforcement learning, the input and target change over time. In other words, we train an agent to estimate the Q-values but the Q-values are changing as the agent knows the environment better.

Experience replay allows the agent to learn from past experience.

Each experience is stored in a replay buffer as the agent interacts with the environment. The replay buffer contains a collection of experience tuples with the state, action, reward, and next state `(s, a, r, s')`. The agent then samples from this buffer as part of the learning step. Experiences are sampled randomly, so that the data is uncorrelated. This prevents action values from oscillating or diverging catastrophically, since a naive Q-learning algorithm could otherwise become biased by correlations between sequential experience tuples.

Also, experience replay improves learning through repetition. By doing multiple passes over the data, our agent has multiple opportunities to learn from a single experience tuple. This is particularly useful for state-action pairs that occur infrequently within the environment.

The implementation of the replay buffer can be found [here](https://github.com/truonghuu/drlnd_navigation/blob/master/dqn_agent.py) in the `dqn_agent.py` file of the source code.

### Results

Through different runs, it is observed that the agent achieves an average score of +13 over 100 consecutive episodes after a total of 327 episodes of training where the first 100 episodes are consider as ram up period. The figure below shows the evolution of average score with respect to number of episodes run.

<img src="https://github.com/truonghuu/drlnd_navigation/blob/master/figures/score_evolution.png" width="50%" align="top-left" alt="" title="Experiment Results" />

### Future Direction

- It would be interesting to explore the performance of a bigger/deeper neural network for the agent. 
- It would be interesting to implement a double DQN, a dueling DQN, and/or prioritized experience replay and compare with the current implementation to identify the best implementation.
- It would also be interesting to implement the agent that can learn from raw pixels. 

