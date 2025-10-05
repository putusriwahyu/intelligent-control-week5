import numpy as np
import matplotlib.pyplot as plt

# 🔹 Load data reward hasil training
rewards = np.load("rewards_resume.npy")

# 🔹 Tampilkan ringkasan data
print(f"Total episode: {len(rewards)}")
print(f"Reward maksimum: {np.max(rewards):.2f}")
print(f"Reward rata-rata: {np.mean(rewards):.2f}")

# 🔹 Buat grafik
plt.figure(figsize=(10,5))
plt.plot(rewards, label="Reward per Episode", color='royalblue', alpha=0.7)

# Moving average biar halus
if len(rewards) > 20:
    window = 20
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode="valid")
    plt.plot(range(window-1, len(rewards)), moving_avg, label=f"Rata-rata {window} episode", color='orange', linewidth=2)

plt.title("Performa DQN pada LunarLander-v2")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
