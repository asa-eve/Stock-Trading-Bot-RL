import os
from stable_baselines3.common.utils import set_random_seed
import pandas as pd
import numpy as np

#-------------------------------------------------------------------------------------------------------------------------------------------------------

def calculate_sharpe(df_account_value):
    # daily_return
    perf_data = df_account_value["account_value"].pct_change(1)

    if perf_data.std() !=0:
      sharpe = (252**0.5)*perf_data.mean()/ \
            perf_data.std()
      return sharpe
    else:
      return 0

#-------------------------------------------------------------------------------------------------------------------------------------------------------

def set_seed(seed_value):
    os.environ['PYTHONHASHSEED']=str(seed_value)
    set_random_seed(seed_value) # random, numpy, tensorflow seeds
    
#-------------------------------------------------------------------------------------------------------------------------------------------------------
    
def check_and_make_directories(directories):
    for directory in directories:
        if not os.path.exists("./" + directory):
            os.makedirs("./" + directory)
            