# Rainbow-RL-Agent
Ultimate version of Reinforcement Learning **Rainbow Agent** with Tensorflow 2 from paper "Rainbow: Combining Improvements in Deep Reinforcement Learning".
My version can handle Recurrent Neural Nets and Multi Parallelized Environments.

The Rainbow Agent is a DQN agent with strong improvments :
- **DoubleQ-learning** : Adding a Target Network that is used in the loss function and upgrade once every `tau` steps. See paper [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
- **Distributional RL** : Approximating the probability distributions of the Q-values instead of the Q-values themself. See paper : [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887)
- **Prioritizedreplay** : Sampling method that prioritize experiences with big *Temporal Difference(TD) errors* (~loss) at the beginning of a training. See paper : [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
- **Dueling Networks**: Divide neural net stream into two branches, an action stream and a value stream. Both of them combined formed the Q-action values. See paper : [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1509.06461)
- **Multi-step learning** : Making Temporal Difference bigger than classic DQN (where TD = 1). See paper [Multi-step Reinforcement Learning: A Unifying Algorithm](https://arxiv.org/abs/1703.01327)
- **NoisyNets** : Replace classic epsilon-greedy exploration/exploitation with noise in the Neural Net. [Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295)

## How to use ?

### Install
Import the rainbow agent
```python
git clone https://github.com/ClementPerroud/Rainbow-Agent rainbow
```
###Import
```python
from rainbow.agent import Rainbow
```

### Usage
```python
agent = Rainbow(
    simultaneous_training_env = 5,
    
    #Distributional
    distributional= True,
    v_min= -200,
    v_max = 250,
    nb_atoms= 51,
    
    # Prioritized Replay
    prioritized_replay = True,
    prioritized_replay_alpha= 0.5,
    prioritized_replay_beta_function = lambda episode, step : min(1, 0.5 + 0.5*step/150_000),
    
    # General
    multi_steps = 3,
    nb_states = 6,
    nb_actions = 4,
    gamma = 0.99,
    replay_capacity = 1E8,
    tau = 2000,
    
    # Model
    window= 15,
    units = [16,16, 16],
    dropout= 0.2,
    adversarial= True,
    noisy= True,
    learning_rate = 3*2.5E-4,

    batch_size= 128,
    train_every = 10,
    # epsilon_function = lambda episode, step : max(0.001, (1 - 5E-5)** step), # Useless if noisy == True
    name = "Rainbow",
)
```

