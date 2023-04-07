# 🤖 Trading_Bot_RL

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

## ⛓ **Some requirements**
- Datasets
  - column names - exact names -> ['date', 'open', 'high', 'low', 'close', 'volume']

## 🌧 **Current build**
- ✔ Simple stock trading 
- - [x] Custom policy networks (RNN, CNN)
- - [x] Resolved problem with gradients in DDPG, TD3, SAC
- - [x] Threshold exploit (risk management)
- - [x] Optuna Tuning
- - [x] Forecasting Model Code
- - [x] Multiple Stock Trading


## 💻 Installation and Running 
Download the package:
```
!wget https://raw.github.com/asa-eve/Trading_Bot_RL/main/code_examples/main_iterative_training.ipynb
```
Use code that is located in 'code examples' folder.

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
  - the most important parameters - learning rate, neural networks, batch memory storages size
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
