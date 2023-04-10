# 🤖 Stock Trading Bot

Simple Stock trading bot using offline Reinforcement Learning for decision making.

Environment and some functions were modified from library [FinRL: Financial Reinforcement Learning](https://github.com/AI4Finance-Foundation/FinRL). 

Currently **works only with SB3 and SB3-contrib**.

For more detailed [explanations and clues on how to solve Stock Trading task using RL.](https://github.com/asa-eve/Stock-Trading-Bot-RL/tree/main/trading_bot_rl#-trend-vs-trade-or-how-to-make-learning-process-faster)

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

Every model will be saved and taken from 'dataset' folder with next names of both model folder and model archive:
```
# MODEL FOLDER NAME
---
dataset name example = ^GSPC_ta_my_features
dataset forecasts name example = ^GSPC_ta_my_features_with_forecasts_LSTM_1_120

model_folder_name = <dataset_name>__<dataset_forecasts_prefix>__<data % on which testing done>__<Normalization and Encoding used>
example: ^GSPC_ta_my_features__with_forecasts_LSTM_1_120__Test15__NormEncdFalse
---

# FILE MODEL NAME
---
model_file_name = <model_name>__<length of iteration>__<length of training in 1 iteration>__<lr>__<seed fixed or no>
example: LSTM_PPO__iterationEp30__trainedEp30__lr0.0001__seed1
---
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
