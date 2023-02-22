import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from gym.spaces import Box, Dict, Discrete
import tensorflow as tf
import numpy as np

class ParametricActionsModel(TFModelV2):
    
    def __init__(self, obs_space, action_space, num_outputs,
        model_config, name, *args, **kwargs):
        
        super(ParametricActionsModel, self).__init__(obs_space,
            action_space, num_outputs, model_config, name, *args, **kwargs)
        
        self.true_obs_shape = (7,)

        self.action_embed_model = FullyConnectedNetwork(Box(np.finfo(np.float32).min,np.finfo(np.float32).max,shape=self.true_obs_shape),
                                  action_space,
                                  num_outputs,
                                  model_config,
                                  name,
                                                       )             # action embedding model
        self.register_variables(self.action_embed_model.variables())
        
    
    def forward(self, input_dict, state, seq_lens):
        
        action_mask= tf.cast(tf.concat(input_dict["obs"]["action_mask"], axis=1), tf.float32)  # action mask values
        
        action_embedding,_ = self.action_embed_model({"obs":input_dict["obs"]["trading_state"]}) # action embeddings
        
        logit_mod = tf.maximum(tf.math.log(action_mask),tf.float32.min)                          # moidfiers to action logits
        
        return (action_embedding+logit_mod), state
    
    def value_function(self):
        return self.action_embed_model.value_function()
    
    
        
def register_actor_mask_model():
    ModelCatalog.register_custom_model("trading_mask", ParametricActionsModel)    #Register the masking model