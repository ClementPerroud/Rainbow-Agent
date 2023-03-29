import tensorflow as tf
import numpy as np
import dill
import pathlib
import types

def save_agent(agent, path):
    print(f"Saving agent to {path}")
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    
    agent.model.save_weights(path + "/model.h5",)
    agent.target_model.save_weights(path + "/target_model.h5",)
    np.savez_compressed(path + "/memory",
                        states_memory = agent.memory.states_memory,
                        actions_memory = agent.memory.actions_memory,
                        rewards_memory = agent.memory.rewards_memory,
                        states_prime_memory = agent.memory.states_prime_memory,
                        done_memory = agent.memory.done_memory,
                        probabilities = agent.memory.probabilities)
                        
    model = agent.model
    target_model = agent.target_model
    states_memory = agent.memory.states_memory
    actions_memory = agent.memory.actions_memory
    rewards_memory = agent.memory.rewards_memory
    states_prime_memory = agent.memory.states_prime_memory
    done_memory = agent.memory.done_memory
    probabilities = agent.memory.probabilities


    agent.model = None
    agent.target_model = None
    agent.memory.states_memory = agent.memory.actions_memory = agent.memory.rewards_memory = agent.memory.states_prime_memory = agent.memory.done_memory = agent.memory.probabilities_memory = None

    with open(path + "/agent.pkl", "wb") as file:
        dill.dump(agent, file)

    agent.model = model
    agent.target_model = target_model
    agent.target_model = target_model
    agent.memory.states_memory = states_memory
    agent.memory.actions_memory = actions_memory
    agent.memory.rewards_memory = rewards_memory
    agent.memory.states_prime_memory = states_prime_memory
    agent.memory.done_memory = done_memory
    agent.memory.probabilities = probabilities

def load_agent(path, retrain = True, verbose = True):
    if verbose: print(f"Loading agent {path}")
    with open(path + "/agent.pkl", "rb") as file:
        agent = dill.load(file)

    agent.model = agent.build_model()
    agent.model.load_weights(path + '/model.h5')
    agent.model.build(agent.input_shape)   
    
    if retrain:
        agent.target_model = agent.build_model(trainable = False)
        agent.target_model.load_weights(path + '/target_model.h5')
        agent.target_model.build(agent.input_shape)
        
        memories = np.load(path + "/memory.npz")
        agent.memory.states_memory = memories["states_memory"]
        agent.memory.actions_memory = memories["actions_memory"]
        agent.memory.rewards_memory = memories["rewards_memory"]
        agent.memory.states_prime_memory = memories["states_prime_memory"]
        agent.memory.done_memory = memories["done_memory"]
        agent.memory.probabilities = memories["probabilities"]
    else:
        def end_training(agent, *args, **kwargs):
            print("The model cannot be trained anymore nor store experiences")
        agent.train = agent.store_experience =  types.MethodType(end_training, agent)
    return agent
    