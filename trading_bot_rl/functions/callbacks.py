from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback

#-------------------------------------------------------------------------------------------------------------------------------------------------------        
# save freq (steps)
def checkpoint_callback(eval_save_freq, save_path='./models_saved_checkpointcallback/'):
    return CheckpointCallback(save_freq=eval_save_freq, save_path=save_path)

#-------------------------------------------------------------------------------------------------------------------------------------------------------        

# deterministic (deterministic actions for evaluation)
def eval_callback(eval_env, eval_steps_freq, best_model_save_path='./models_saved_evalcallback/'):
    return EvalCallback(eval_env, best_model_save_path=best_model_save_path,
                             log_path=best_model_save_path, eval_freq=eval_steps_freq,
                             deterministic=False, render=False)

#-------------------------------------------------------------------------------------------------------------------------------------------------------        

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        try:
            self.logger.record(key="train/reward", value=self.locals["rewards"][0])
        except BaseException:
            self.logger.record(key="train/reward", value=self.locals["reward"][0])
        return True