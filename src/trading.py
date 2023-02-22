import numpy as np
import math
import gym
from gym.spaces import Box, Dict, Tuple, Discrete




def state_bounds_gen():
    
        state_bound_dict={
        
            
            "asset1":[-np.finfo(np.float32).max,np.finfo(np.float32).max],
            "asset2":[-np.finfo(np.float32).max,np.finfo(np.float32).max],
            "asset3":[-np.finfo(np.float32).max,np.finfo(np.float32).max],
            "balance":[0,np.finfo(np.float32).max],
            "price1":[-np.finfo(np.float32).max,np.finfo(np.float32).max],
            "price2":[-np.finfo(np.float32).max,np.finfo(np.float32).max],
            "price3":[-np.finfo(np.float32).max,np.finfo(np.float32).max]
        
        }
        return state_bound_dict    


class mytradingenv(gym.Env):
    
    def __init__(self,*args, **kwargs):
    

        self.buy_price=np.array([0.03, 0.045, 0.035])        # transaction cost per unit bought for three asset classes
        self.sell_price=np.array([0.025, 0.035, 0.03])       # transaction cost per unit sold for three asset classes
        self.mu=np.array([40,35,48])                         # Mean initial asset price
        self.var=np.array([4,2,7])                           # Variance of asset prices
        self.tvec=np.arange(20)                              # Length of each episode=20
        self.sig=np.zeros((3,len(self.tvec)))
        self.sig[0,:]=self.mu[0]+0.4*self.tvec+4*np.cos(2*math.pi*self.tvec/16)  #Functions used to model mean asset prices over time
        self.sig[1,:]=self.mu[1]+0.1*self.tvec
        self.sig[2,:]=self.mu[2]+0.3*self.tvec-6*np.sin(2*math.pi*self.tvec/7)
        
        state_bounds=state_bounds_gen()
        low,high= map(np.array,zip(*state_bounds.values()))  # Minimun and maximum values for the state variables         
        
        self.action_space = Tuple([Discrete(11),Discrete(11),Discrete(11)])  #Action space consisting of three discrete actions
        
        self.observation_space=Dict({"action_mask":Tuple([Box(0,1,shape=(11,)),Box(0,1,shape=(11,)),Box(0,1,shape=(11,))]),
                                     "trading_state":Box(low,high,dtype=np.float32)})  # Dictionary space consisting of trading state 
                                                                                       # and action mask
      
    
    def reset(self):
        
        self.assets=np.zeros(3,dtype=np.float32)        # Assets owned at the beginning
        self.balance=1000                               # Initial cash balance
        self.t_step=0
        self.prices=[np.random.normal(mu,var) for mu,var in zip(self.mu,self.var)]  # Sampling marker prices for the assets
        self.state=np.hstack([self.assets, self.balance, self.prices])        # Initial state
        self.total_assets=self.balance                                        # Total portfolio value
        self.update_mask()                                                    # Updating action mask values
        
        reset_state={
            "action_mask":list(np.float32(self.action_mask)),                 # Initial state  
            "trading_state":np.float32(self.state)
        }
        
        return reset_state
    
    
    def step(self,action):    
        self.t_step+=1
        
        for index, a in enumerate (action):
            print("action is",a)
            print("price is",self.prices[index])
            quant=abs(a-5)                                              # Number of assets traded/10
            if a<5:                                                     # Condition: Asset sale ?
                if 10*quant*self.sell_price[index]>self.balance:        # Condition: sale cost > Balance ? 
                    quant=np.floor(self.balance/(10*self.sell_price[index]))    
                self.assets[index]-=10*quant                               # Asset update
                self.balance=self.balance+10*quant*(self.prices[index]-self.sell_price[index]) # Balance update
            if a>5:
                if 10*quant*(self.buy_price[index]+self.prices[index])>self.balance:          # Condition: Buy cost > Balance ?
                    quant=np.floor(self.balance/(10*(self.buy_price[index]+self.prices[index])))
                self.assets[index]+=10*quant                               # Asset update
                self.balance=self.balance-10*quant*(self.prices[index]+self.sell_price[index]) # Balance update
            else:
                continue
        
        self.prices=np.array([np.random.normal(mu,var) for mu,var in zip(self.sig[:,self.t_step],self.var)]) # New asset prices
        self.state=np.hstack([self.assets,self.balance, self.prices])                                        # New state
        self.total_assets=self.balance+np.dot(self.assets,self.prices)                                       # Total portfolio value
        self.update_mask()                                                                                   # Mask update
       
        obs={
            "action_mask": list(np.float32(self.action_mask)),
            "trading_state":np.float32(self.state)
            
        }
       
        if self.t_step==len(self.tvec)-1:
            reward=self.total_assets        # reward = Total portfolio value at the end of the episode
        else:
            reward=0
        done=True if self.t_step==len(self.tvec)-1 else False
        return obs, reward, done, {}
        
        
    def update_mask(self):
        
        self.action_mask=[np.array([1.0]*x.n) for x in self.action_space.spaces]  # Set all masks to 1 
       
        if self.balance<1:                                                        # If balance < 1, set buy masks to zero (C4)
            for jj in range(len(self.action_mask)):
                self.action_mask[jj][6:]=[0.0]*5
           
        self.action_mask[2][6:]=[0.0]*5 if (self.prices[2]*self.assets[2]/self.total_assets)>1/3 else [1.0]*5  #(C3)
        
        self.action_mask[1][6:]=[0.0]*5 if (self.prices[1]*self.assets[1]/self.total_assets)>2/3 else [1.0]*5  #(C2)
        
        for k in range(3):
            cap=int(min(5,self.assets[k]/10))
            self.action_mask[k][:5]=[0.0]*(5-cap)+[1.0]*cap                                          # (C1)
       
        
    


