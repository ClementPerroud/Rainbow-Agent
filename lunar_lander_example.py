
import gym
from agent import Rainbow

env = gym.make("LunarLander-v2")

rainbow = Rainbow(
    #Distributional
    distributional= True,
    v_min= -150,
    v_max = 170,
    nb_atoms= 51, 
    # Prioritized Replay
    prioritized_replay =True,
    prioritized_replay_alpha= 0.5,
    prioritized_replay_beta_function = lambda episode, step : min(1, 0.5 + 0.5*step/100_000),
    # General
    multi_steps = 3,
    nb_states = 8,
    nb_actions = 4,
    gamma = 0.99,
    replay_capacity = 1E6,
    tau = 3000,
    
    # Model
    window= 5,
    units = [64,64],
    dropout= 0,
    adversarial= False,
    noisy= False,
    learning_rate = 1E-3,

    batch_size= 64,
    train_every = 4,
    epsilon_function = lambda episode, step : max(0.01, (1 - 5E-5)** step), #lambda episode, step : max(0.05, 0.9999 ** step), # Ignore if noisy is True
    name = "",
)



for _ in range(400):
    obs, info = env.reset()
    done, truncated = False, False
    while not done and not truncated:
        action = rainbow.e_greedy_pick_action_or_random(obs)
        next_obs, reward, done, truncated, info = env.step(action)
        rainbow.store_replay(obs, action, reward, next_obs, done, truncated)
        rainbow.train()
        obs = next_obs




batch_indexes, states, actions, rewards, states_prime, dones, importance_weights = rainbow.replay_memory.sample(
    1024,
    rainbow.prioritized_replay_beta_function(rainbow.episode_count, rainbow.steps)
)
results = rainbow.model(states)

action_colors=["blue", "orange","purple","red"]
fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize=(16,9))
for action in range(4):
    for i in range(256):
        axes[action%2, action//2%2].plot(rainbow.zs, results[i, action, :], color = action_colors[action], alpha = 0.2)


