# 🤖 Stock Trading Bot

Simple Stock trading bot using offline Reinforcement Learning for decision making.

Environment and some functions were modified from library [FinRL: Financial Reinforcement Learning](https://github.com/AI4Finance-Foundation/FinRL). 

Currently **works only with SB3 and SB3-contrib**.

## 🦾 **Project main features**
- supports both discrete and continious action space RL algorithms
- restoration of training process (from saved model)
- iterative training feature with changable seeds (for systems with low RAM)
- changable custom features for RL (called INDICATORS)
- RL algorithms from [Stable-Baselines-3](https://stable-baselines3.readthedocs.io/en/master/) & [Stable-Baselines-3 Contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib) (PPO LSTM, TRPO, etc.) are available for usage
- works with any Python version >= 3.6

## ⛓ **Package - requirements, problems and hints**
- Requirement - dataset must contain next exact column names  ---> ['date', 'open', 'high', 'low', 'close', 'volume']
- Problem     - vanishing gradients in netx methods           ---> DDPG, TD3, SAC
- Hint        - use RNN with LSTM layers to gain more perfomance

## 💻 Installation and Running 
Download the package:
```
!wget https://raw.github.com/asa-eve/Trading_Bot_RL/main/code_examples/main_iterative_training.ipynb
```
Use code that is located in 'code examples' folder.

## 🌧 **Trend vs Trade** or **"How to make learning process faster"**
Depending on dataset you might find the problem of finding optimal solution - it would be displayed through 'time required to train' and jumps in agent's perfomance metric - that means agent tries to choose between 2 strategies:
- active Trading
- passive Trending

With 'Trading' we want for agent to focus more on the number of trades and money, while with 'Trending' agent's task is only to maximize the total earnings. 

In 1st case we want to define reward to help agent trade more actively:
```
self.reward = (end_total_asset - begin_total_asset) * self.reward_scaling  *  (self.trades / self.day)   *  <coefficient_of_trading>
```
For "LSTM PPO" from "SB3 contrib" it gave a huge growth of perfomance at early stages of the training process. That coefficient could be tuned (by Optuna) in order to find optimal model for the particular dataset.

In 'Trending' case the reward look simplier:
```
self.reward = (end_total_asset - begin_total_asset) * self.reward_scaling
```
The training process might rapidly slow and after some time (depending on the data - could be days or even weeks) it will come to the conclusion of using 'trend'. After that if no additional 'coefficients' were added to the reward for keeping trend - an agent might start to 'trade' if the environment isn't changing. 

**Exploration vs. Exploitation** - you should also keep in mind that the task might challenge an agent in finding "optimal solution" - so there are some hyperparemeters that should be kept in mind:
- entropy coefficient
    - could help to EXPLORE environment more (by adding random actions)
- clipping
    - could help to EXPLOIT the strategy (from 0 to 1)
- regularization
    - sometimes it's hard to get a good perfomance on 'train' and 'test' - meaning generalization is bad - adding L2 or dropout regularization could help
- batch size
    - larger batch size - faster learning, more memory usage, more stable and smooth training and convergence - but also reduced exploration and chances of overfitting


## 🧠 Things to keep in mind while using
- Iteration length
  - short is not enough - but long might creates 'overfit' to the train (mostly to the noise)
  - **consider something average - start with long and slowly decrease**
  - full learning process of a single model might take hours, days or even weeks (depending on the task)
- Performance metrics
  - consider checking 'perfomance' on validation - to get grasp on how good the model generalizes the data
  - but don't forget - metrics are good but **total reward your best guide**
- Hyperparameters
  - hyperparameter tuning might not strongly affect the outcome perfomance
  - the most important parameters - learning rate, neural networks, batch memory storages size, parameters of "Explore vs Exploit" (clipping, entropy)
- Reward function
  - the most influential thing in RL
  - think carefully on how to define it - the agent always has a ways to surprise you in finding ways to cheat it
- State space (features choice)
  - it's important to consider what really affect the result and what's not
  - remove as much 'noisy' features as you can - unnecessary features greatly increase the complexity of the agent learning task
- RL algorithms
  - try out most popular algorithms (PPO, A2C, DQN, DDPG) - there's no need to check every single one of them


## 📃 File Structure
```
Trading_Bot_RL
├── trading_bot_rl (main folder)
│   ├── agent.py
│   ├── env.py
│   └── functions
│   	├── callbacks.py
│   	├── data_preprocessing.py
│   	├── env_functions.py
│   	├── yahoodownloader.py
│   	└── general.py
├── code_examples
├── datasets
├── trained_models
├── setup.py
├── requirements.txt
└── README.md
```
