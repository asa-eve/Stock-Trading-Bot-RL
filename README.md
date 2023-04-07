# 🤖 Trading_Bot_RL

Simple system for stock trading using Reinforcement Learning as decision-maker. 

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
Download and run 1 of example codes (via command):
```
!wget https://raw.github.com/asa-eve/Trading_Bot_RL/main/code_examples/main_iterative_training.ipynb
```

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
