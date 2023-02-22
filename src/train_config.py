from ray.tune.registry import register_env
from mask_model import register_actor_mask_model
from sagemaker_rl.ray_launcher import SageMakerRayLauncher
from ray import tune





register_actor_mask_model()


class MyLauncher(SageMakerRayLauncher):

    def register_env_creator(self):
        env_name='mytradingmodel' 
        env_config={}
        from trading import mytradingenv
        tune.register_env(env_name, lambda env_name: mytradingenv(env_name, env_config=env_config))  #register trading env
        
        
    def get_experiment_config(self):
        return {
            "training": {
                   "env": "mytradingmodel",
                   "run": "PPO",                     # Use PPO algorithm
                   "stop":{"episodes_total":500000}, # 500k training episodes
                   "config": {
                      "use_pytorch": False,
                      "gamma": 0.99,
                      "kl_coeff": 1.0,
                      "num_sgd_iter": 20,
                      "lr": 0.0001,
                      "sgd_minibatch_size": 1000,
                      "train_batch_size": 25000,
                      "monitor": True,  
                      "model": {
                          "custom_model": "trading_mask"  # Use custom action masking model                        
                            },
                      "num_workers": (self.num_cpus-1),
                      "num_gpus": self.num_gpus,
                      "batch_mode": "truncate_episodes",
                       "explore":True,
                       "exploration_config":{
                           "type":"StochasticSampling",  
                       },
                     },
                     "checkpoint_freq": 1, 
                  }
             }
    
    
if __name__ == "__main__":
    MyLauncher().train_main()    
    