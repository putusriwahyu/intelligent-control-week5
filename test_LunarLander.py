# ============================================================
# 🚀 DQN Cepat + Resume Training + Grafik Reward (Versi Aman TF 2.15+)
# ============================================================
# ✅ Kompatibel Gym baru, bisa lanjut training, dan otomatis simpan data
# ============================================================

import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random
import os
import time
import matplotlib.pyplot as plt

# ✅ Patch NumPy untuk Gym lama
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

# ---------- Hyperparameter ----------
ENV_NAME = "LunarLander-v2"  # ganti ke "Acrobot-v1" kalau Box2D error
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

LEARNING_RATE = 0.0005
GAMMA = 0.99
BUFFER_SIZE = 100000
BATCH_SIZE = 128
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE_EVERY = 1000
MAX_EPISODES = 1000
MAX_STEPS = 1000
MIN_REPLAY_SIZE = 1000
TRAIN_EVERY = 10
EVAL_EVERY = 50

# ✅ Format baru .keras lebih aman untuk resume training
MODEL_SAVE_PATH = "dqn_best_resume.keras"
REWARD_SAVE_PATH = "rewards_resume.npy"

# ---------- Replay Buffer ----------
class ReplayBuffer:
    def __init__(self, max_size=BUFFER_SIZE):
        self.buffer = deque(maxlen=max_size)

    def add(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))

    def sample(self, batch_size=BATCH_SIZE):
        batch = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*batch))

    def __len__(self):
        return len(self.buffer)

# ---------- Build Q-network ----------
def build_model(state_size, action_size):
    model = keras.Sequential([
        keras.layers.Input(shape=(state_size,)),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dense(action_size, activation="linear")
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mean_squared_error"
    )
    return model

# ---------- Gym wrapper ----------
def env_reset(env, seed=None):
    res = env.reset(seed=seed) if seed is not None else env.reset()
    return res[0] if isinstance(res, tuple) else res

def env_step(env, action):
    res = env.step(action)
    if len(res) == 5:
        s, r, term, trunc, info = res
        return s, r, term or trunc, info
    return res

# ---------- Epsilon-greedy ----------
def select_action(model, state, epsilon, n_actions):
    if np.random.rand() <= epsilon:
        return np.random.randint(n_actions)
    q_values = model.predict(state[np.newaxis, :], verbose=0)
    return np.argmax(q_values[0])

# ---------- Training (with resume support) ----------
def train(resume=True):
    env = gym.make(ENV_NAME)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    q_net = build_model(state_size, action_size)
    target_net = build_model(state_size, action_size)

    start_ep = 1
    rewards_history = []

    # 🔁 Load model & reward jika resume
    if resume and os.path.exists(MODEL_SAVE_PATH):
        q_net = keras.models.load_model(MODEL_SAVE_PATH, compile=False)
        q_net.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                      loss="mean_squared_error")
        target_net.set_weights(q_net.get_weights())
        if os.path.exists(REWARD_SAVE_PATH):
            rewards_history = list(np.load(REWARD_SAVE_PATH))
            start_ep = len(rewards_history) + 1
        print(f"Lanjut training dari episode {start_ep} (model sebelumnya ditemukan)")
    else:
        target_net.set_weights(q_net.get_weights())
        print("Mulai training baru dari awal")

    replay = ReplayBuffer()
    epsilon = EPSILON_START * (EPSILON_DECAY ** len(rewards_history))
    total_steps = 0
    best_avg_reward = -np.inf

    # Prefill replay buffer
    state = env_reset(env, seed=SEED)
    while len(replay) < MIN_REPLAY_SIZE:
        a = env.action_space.sample()
        ns, r, done, _ = env_step(env, a)
        replay.add(state, a, r, ns, done)
        state = env_reset(env) if done else ns

    for ep in range(start_ep, MAX_EPISODES + 1):
        state = env_reset(env, seed=SEED + ep)
        ep_reward = 0

        for step in range(MAX_STEPS):
            total_steps += 1
            a = select_action(q_net, state, epsilon, action_size)
            ns, r, done, _ = env_step(env, a)
            replay.add(state, a, r, ns, done)
            state = ns
            ep_reward += r

            if len(replay) >= BATCH_SIZE and total_steps % TRAIN_EVERY == 0:
                s_b, a_b, r_b, ns_b, d_b = replay.sample(BATCH_SIZE)
                q_next = target_net.predict(ns_b, verbose=0)
                q_values = q_net.predict(s_b, verbose=0)
                for i in range(BATCH_SIZE):
                    q_values[i, a_b[i]] = r_b[i] if d_b[i] else r_b[i] + GAMMA * np.max(q_next[i])
                q_net.train_on_batch(s_b, q_values)

            if total_steps % TARGET_UPDATE_EVERY == 0:
                target_net.set_weights(q_net.get_weights())

            if done:
                break

        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        rewards_history.append(ep_reward)

        # Simpan progress tiap 10 episode
        if ep % 10 == 0:
            np.save(REWARD_SAVE_PATH, rewards_history)
            q_net.save(MODEL_SAVE_PATH, include_optimizer=False)
            print(f"Episode {ep:4d} | Reward: {ep_reward:7.2f} | Eps: {epsilon:.3f}")

        # Evaluasi tiap beberapa episode
        if ep % EVAL_EVERY == 0:
            avg = evaluate(q_net, env, 3)
            print(f"  🔹 Eval Avg Reward: {avg:.2f}")
            if avg > best_avg_reward:
                best_avg_reward = avg
                q_net.save(MODEL_SAVE_PATH, include_optimizer=False)
                print(f"Model terbaik diperbarui (avg {best_avg_reward:.2f})")

    env.close()
    np.save(REWARD_SAVE_PATH, rewards_history)
    print("Training selesai. Semua data disimpan.")

    # ---------- Grafik ----------
    plt.figure(figsize=(8,5))
    plt.plot(rewards_history, label="Reward per Episode", alpha=0.7)
    if len(rewards_history) > 20:
        moving = np.convolve(rewards_history, np.ones(20)/20, mode="valid")
        plt.plot(range(19, len(rewards_history)), moving, color='orange', label="Rata-rata 20 episode", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"Performa DQN (Resume Mode) pada {ENV_NAME}")
    plt.legend()
    plt.grid(True)
    plt.show()

# ---------- Evaluation ----------
def evaluate(model, env, episodes=5, render=False):
    total = 0
    for _ in range(episodes):
        s = env_reset(env)
        done = False
        ep_r = 0
        while not done:
            a = np.argmax(model.predict(s[np.newaxis, :], verbose=0)[0])
            s, r, done, _ = env_step(env, a)
            ep_r += r
            if render:
                env.render()
        total += ep_r
    return total / episodes

# ---------- Run ----------
if __name__ == "__main__":
    start = time.time()
    train(resume=True)  # ubah ke False kalau mau mulai dari nol
    print(f"Total waktu: {(time.time()-start)/60:.2f} menit")

    if os.path.exists(MODEL_SAVE_PATH):
        model = keras.models.load_model(MODEL_SAVE_PATH, compile=False)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                      loss="mean_squared_error")
        avg = evaluate(model, gym.make(ENV_NAME), episodes=5)
        print(f"Rata-rata reward akhir (5 episode): {avg:.2f}")
