from trading_bot_rl.env import StockTradingEnv

# -------------------------------------------------------------------------------------------------------------------------------------------------------

def env_reinit(data, env_kwargs={}, turbulence_threshold=None, risk_indicator_col=None):
    if turbulence_threshold != None and risk_indicator_col != None:
        e_gym = StockTradingEnv(df=data, turbulence_threshold=turbulence_threshold,
                                risk_indicator_col=risk_indicator_col, **env_kwargs)
    else:
        e_gym = StockTradingEnv(df=data, **env_kwargs)
    env_, _ = e_gym.get_sb_env()
    return e_gym, env_


# -------------------------------------------------------------------------------------------------------------------------------------------------------

def turbulence_threshold_define(data, quantile):
    #quantile = 0.997
    return data.turbulence.quantile(quantile)