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

### Implementation of Learning Algorithm

It has been shown in `Navigation.ipynb` that using an agent which takes action randomly does not solve the problem. A more intelligent agent is therefore needed to achieve an average score of +13 over 100 consecutive episodes. We employ Q-Learning that aims at finding an optimal policy, i.e., a policy that maximizes the scores of the agent.   

The main component of Q-Learning is the Q-Function that calculates the expected reward `R` for all possible actions `A` in all possible states `S` (i.e., `Q: A x S --> R`). Based on Q-Function, we can then define the optimal policy `π*` as the action that maximizes the Q-function for a given state across all possible states. The optimal Q-function `Q*(s,a)` maximizes the total expected reward for an agent starting in state `s` and choosing action `a`, then following the optimal policy for each subsequent state.

#### Epsilon Greedy Algorithm

One challenge with the Q-function above is choosing which action to take while the agent is still learning the optimal policy. Should the agent choose an action based on the Q-values observed thus far? Or, should the agent try a new action in hopes of earning a higher reward? This is known as the **exploration vs. exploitation dilemma**.

To address this, we employ an **𝛆-greedy algorithm**. This algorithm allows the agent to systematically manage the exploration vs. exploitation trade-off. The agent "explores" by picking a random action with some probability epsilon `𝛜`. However, the agent continues to "exploit" its knowledge of the environment by choosing actions based on the policy with probability (1-𝛜).

Furthermore, the value of epsilon is purposely decayed over time, so that the agent favors exploration during its initial interactions with the environment, but increasingly favors exploitation as it gains more experience. The starting and ending values for epsilon, and the rate at which it decays are three hyperparameters that are later tuned during experimentation.

The logic of 𝛆-greedy algorithm is implemented as part of the `agent.act()` method [here](https://github.com/truonghuu/drlnd_navigation/blob/master/dqn_agent.py) in `dqn_agent.py` of the source code.


#### Deep Q-Network (DQN)

With Deep Q-Learning, a deep neural network is used to approximate the Q-function. Given a network `F`, finding an optimal policy is a matter of finding the best weights `w` such that `F(s,a,w) ≈ Q(s,a)`.

The neural network architecture used for this project can be found [here](https://github.com/truonghuu/drlnd_navigation/blob/master/model.py) in the `model.py` file of the source code. The network contains three fully connected layers with 64, 64, and 4 nodes, respectively.

The input layer has 37 nodes, which is the size of a state. We employ the experience replay approach to feed in the deep neural network a number of past experience (i.e., batch).

#### Experience Replay

Experience replay allows the agent to learn from past experience.

Each experience is stored in a replay buffer as the agent interacts with the environment. The replay buffer contains a collection of experience tuples with the state, action, reward, and next state `(s, a, r, s')`. The agent then samples from this buffer as part of the learning step. Experiences are sampled randomly, so that the data is uncorrelated. This prevents action values from oscillating or diverging catastrophically, since a naive Q-learning algorithm could otherwise become biased by correlations between sequential experience tuples.

Also, experience replay improves learning through repetition. By doing multiple passes over the data, our agent has multiple opportunities to learn from a single experience tuple. This is particularly useful for state-action pairs that occur infrequently within the environment.

The implementation of the replay buffer can be found [here](https://github.com/truonghuu/drlnd_navigation/blob/master/dqn_agent.py) in the `dqn_agent.py` file of the source code.

### Results

Through different runs, it is observed that the agent achieves an average score of +13 over 100 consecutive episodes after a total of 327 episodes of training where the first 100 episodes are consider as ram up period. The figure below shows the evolution of average score with respect to number of episodes run.

<img src="https://github.com/truonghuu/drlnd_navigation/blob/master/score_evolution.png" width="50%" align="top-left" alt="" title="Experiment Results" />

### Future Direction

- It would be interesting to explore the performance of a bigger/deeper neural network for the agent. 
- It would be interesting to implement a double DQN, a dueling DQN, and/or prioritized experience replay and compare with the current implementation to identify the best implementation.
- It would also be interesting to implement the agent that can learn from raw pixels. 

