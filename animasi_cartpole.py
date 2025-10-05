import gym
import numpy as np
from tensorflow import keras

# ✅ Pastikan gym dan numpy kompatibel
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

# Muat model hasil training
model = keras.models.load_model("dqn_best_resume.keras")

# Inisialisasi environment LunarLander
env = gym.make("LunarLander-v2")

EPISODES = 20  # jumlah episode pengujian
scores = []

for ep in range(1, EPISODES + 1):
    state = env.reset()
    if isinstance(state, tuple):  # untuk gym >= 0.26
        state = state[0]
    done = False
    total_reward = 0

    while not done:
        # Prediksi aksi terbaik dari model
        action = np.argmax(model.predict(state[np.newaxis, :], verbose=0)[0])
        
        # Step environment
        result = env.step(action)
        if len(result) == 5:  # gym >= 0.26
            next_state, reward, terminated, truncated, _ = result
            done = terminated or truncated
        else:  # gym lama
            next_state, reward, done, _ = result
        
        total_reward += reward
        state = next_state

    scores.append(total_reward)
    print(f"Episode {ep:2d}: Score = {total_reward:7.2f}")

env.close()

# Rata-rata hasil
print("\n=== Hasil Uji Agen DQN pada LunarLander ===")
print(f"Rata-rata Reward dari {EPISODES} episode: {np.mean(scores):.2f}")
print(f"Reward maksimum: {np.max(scores):.2f}")
print(f"Reward minimum: {np.min(scores):.2f}")
