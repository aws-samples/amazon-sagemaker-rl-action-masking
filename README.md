# Portfolio Optimization through Multidimensional Action Optimization using Amazon SageMaker RL
## Introduction

The objective of a reinforcement learning (RL)  problem is to train an agent that, given an observation from its environment, will choose the optimal action that maximizes cumulative reward. Solving a business problem with RL involves specifying the agent’s environment, the space of actions, the structure of observations and the right reward function for the target business outcome. An RL agent learns by exploring the state space and taking random actions. However, there are several scenarios where some of the actions may not be admissible depending on the state. For e.g., consider an autonomous car that has ten possible speed levels.  This car may only be allowed to choose from a subset of its speed levels when traversing a school neighborhood. In this case, sampling the full action space is inefficient. Action masking is an approach to avoid sampling inadmissible actions. Here we show how to perform action masking an train an RL agent using Amazon SageMaker RL and Ray RLlib.  We consider a portfolio optimization problem that involves a three dimensional action vector and four constraints. Masking is implemented using a parametric action model from RLlib and the agent is trained using proximal policy optimization (PPO) algorithm.

## Getting Started

1. Create a SageMaker Notebook Instance

Training an RL agent using this repository requires a SageMaker notebook instance. For details on how to create a notebook instance, see the aws documentation.

2. Execute `Training_Notebook.ipynb`

Use this Jupyter notebook to execute the training steps in an interactive manner. The environment and masking model files that this notebook uses are located in the src folder

## Verifying Masking

To verify that action masking is working as expected, i.e., blocking actions with mask=0, use `Test_Notebook.ipynb` file located in `src` folder.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

