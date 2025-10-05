import gym
import numpy as np

# ✅ Patch untuk kompatibilitas NumPy >= 1.24 (karena Gym lama masih pakai np.bool8)
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import tensorflow as tf
from tensorflow import keras
from collections import deque
import random
import matplotlib.pyplot as plt

# Inisialisasi lingkungan OpenAI Gym (misalnya, CartPole)
env = gym.make("CartPole-v1")

# Parameter DRL
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
learning_rate = 0.001
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 32
memory = deque(maxlen=2000)

# Membangun model Deep Q-Network (DQN)
model = keras.Sequential([
    keras.layers.Dense(24, input_shape=(state_size,), activation="relu"),
    keras.layers.Dense(24, activation="relu"),
    keras.layers.Dense(action_size, activation="linear")
])
model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=learning_rate))

# Fungsi memilih aksi berdasarkan eksplorasi dan eksploitasi
def select_action(state, epsilon):
    if np.random.rand() <= epsilon:
        return np.random.choice(action_size)  # Eksplorasi
    q_values = model.predict(state, verbose=0)
    return np.argmax(q_values[0])  # Eksploitasi

# Simpan skor tiap episode
scores = []

# Proses training
for episode in range(1000):
    state = env.reset()
    if isinstance(state, tuple):  # gym versi baru return (obs, info)
        state = state[0]
    state = np.reshape(state, [1, state_size])

    total_reward = 0
    for time in range(500):
        action = select_action(state, epsilon)
        step_result = env.step(action)

        # Kompatibilitas gym lama vs baru
        if len(step_result) == 5:  # gym >=0.26
            next_state, reward, terminated, truncated, _ = step_result
            done = terminated or truncated
        else:  # gym lama
            next_state, reward, done, _ = step_result

        next_state = np.reshape(next_state, [1, state_size])
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        if done:
            break

    scores.append(total_reward)

    # Update jaringan saraf (Training)
    if len(memory) > batch_size:
        minibatch = random.sample(memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += gamma * np.amax(model.predict(next_state, verbose=0)[0])
            target_f = model.predict(state, verbose=0)
            target_f[0][action] = target
            model.fit(state, target_f, epochs=1, verbose=0)

    # Kurangi epsilon (exploration decay)
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    # ✅ Print progres tiap episode
    print(f"Episode: {episode+1}, Skor: {total_reward}, Epsilon: {epsilon:.4f}")

print("Training selesai!")

# 📊 Grafik performa
plt.plot(scores, label="Skor per Episode")

# Moving average biar lebih halus
window = 50
if len(scores) >= window:
    moving_avg = np.convolve(scores, np.ones(window)/window, mode="valid")
    plt.plot(range(window-1, len(scores)), moving_avg, label=f"Rata-rata {window} episode")

plt.xlabel("Episode")
plt.ylabel("Skor (Steps)")
plt.title("Performa DQN pada CartPole")
plt.legend()
plt.show()
