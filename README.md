# 🤖 Trading_Bot_RL

Simple system for stock trading using Reinforcement Learning as decision-maker. 

Environment and some functions were modified from library [FinRL: Financial Reinforcement Learning](https://github.com/AI4Finance-Foundation/FinRL). 

## 🦾 **Project main features**
- simple to install, fast to run, easy to understand
- free choice of agent's action space (discrete or continious)
- RL algorithms from [Stable-Baselines-3](https://stable-baselines3.readthedocs.io/en/master/) & [Stable-Baselines-3 Contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib) (PPO LSTM, TRPO, etc.) are available for usage
- iterative training feature (for systems with low RAM) with changable seeds
- ability to choose your own INDICATORS for RL model features
- no requirement for active updates & works with Python >= 3.6

## ⛓ **Some requirements**
- for datasets - presence of next features with exact names ['date', 'open', 'high', 'low', 'close', 'volume']

## 🌧 **Current progress**
- ✔ Simple stock trading 
- - [x] Optuna hyperparameter tuning added
- - [x] Forecasting model training feature added
- - [x] Multiple stock trading


## 💻 Installation and Running 
Download and run 1 of example codes (via command):
```
!wget https://raw.github.com/asa-eve/Trading_Bot_RL/main/code_examples/main_iterative_training.ipynb
```

## 📃 File Structure
```
Trading_Bot_RL
├── trading_bot_rl (main folder)
│   ├── functions
│   	├── callbacks.py
│   	├── data_preprocessing.py
│   	├── env_functions.py
|     ├── yahoodownloader.py
│   	└── general.py
│   ├── agent.py
│   ├── env.py
├── code_examples
│   ├── main_iterative_training.ipynb
├── datasets
│   ├── ...
├── trained_models
│   ├── <model_name>
│   	├── ... 
├── setup.py
├── requirements.txt
└── README.md
```
