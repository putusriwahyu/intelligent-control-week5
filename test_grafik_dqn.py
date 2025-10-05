import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random
import matplotlib.pyplot as plt
import os
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

ENV_NAME = "CartPole-v1"
EPISODES = 1000
MAX_STEPS = 500
LEARNING_RATE = 0.001
GAMMA = 0.95
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 32
MEMORY_SIZE = 2000
REWARD_SAVE_PATH = "rewards_cartpole.npy"

# Inisialisasi Environment
env = gym.make(ENV_NAME)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
memory = deque(maxlen=MEMORY_SIZE)

# Bangun Model DQN 
model = keras.Sequential([
    keras.layers.Dense(24, input_shape=(state_size,), activation="relu"),
    keras.layers.Dense(24, activation="relu"),
    keras.layers.Dense(action_size, activation="linear")
])
model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE))

# Fungsi Pilih Aksi
def select_action(state, epsilon):
    if np.random.rand() <= epsilon:
        return np.random.choice(action_size)
    q_values = model.predict(state, verbose=0)
    return np.argmax(q_values[0])

#  Training 
scores = []
for episode in range(EPISODES):
    state = env.reset()
    if isinstance(state, tuple):  # Gym >=0.26
        state = state[0]
    state = np.reshape(state, [1, state_size])
    total_reward = 0

    for step in range(MAX_STEPS):
        action = select_action(state, EPSILON)
        result = env.step(action)

        if len(result) == 5:
            next_state, reward, terminated, truncated, _ = result
            done = terminated or truncated
        else:
            next_state, reward, done, _ = result

        next_state = np.reshape(next_state, [1, state_size])
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        if done:
            break

    # Simpan total reward episode ini
    scores.append(total_reward)

    # Training model dari memori replay
    if len(memory) > BATCH_SIZE:
        minibatch = random.sample(memory, BATCH_SIZE)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += GAMMA * np.amax(model.predict(next_state, verbose=0)[0])
            target_f = model.predict(state, verbose=0)
            target_f[0][action] = target
            model.fit(state, target_f, epochs=1, verbose=0)

    # Kurangi epsilon
    if EPSILON > EPSILON_MIN:
        EPSILON *= EPSILON_DECAY

    print(f"Episode {episode+1}/{EPISODES} | Reward = {total_reward:.2f} | Epsilon = {EPSILON:.3f}")

# Simpan hasil reward ke file
np.save(REWARD_SAVE_PATH, np.array(scores))
print(f"\nTraining selesai! Reward disimpan ke {REWARD_SAVE_PATH}")

# ---------- Grafik (Gaya Sama Seperti LunarLander) ----------
plt.figure(figsize=(10,5))
plt.plot(scores, color='steelblue', alpha=0.6, label='Reward per Episode')

window = 20
if len(scores) > window:
    moving = np.convolve(scores, np.ones(window)/window, mode='valid')
    plt.plot(range(window-1, len(scores)), moving, color='orange', linewidth=2,
             label=f'Rata-rata {window} episode')

plt.title(f"Performa DQN pada {ENV_NAME}")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)
plt.show()
