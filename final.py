import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 23

# Time vector for a single heartbeat
sampling_rate = 500  # Hz
beat_duration = 1  # seconds
time = np.linspace(0, beat_duration, sampling_rate)

# Normal ECG (N)
def normal_ecg(t):
    p_wave = 0.25 * np.sin(2 * np.pi * 5 * t) * np.exp(-30 * t)
    qrs = -1 * np.exp(-100 * (t - 0.03)**2) + 2.5 * np.exp(-50 * (t - 0.05)**2)
    t_wave = 0.35 * np.sin(2 * np.pi * 1 * t) * np.exp(-10 * (t - 0.2))
    return p_wave + qrs + t_wave

# Left bundle branch block (L)
def left_bundle_block_ecg(t):
    return normal_ecg(t) + 0.5 * np.exp(-30 * (t - 0.15)**2)  # Add delay in depolarization

# Right bundle branch block (R)
def right_bundle_block_ecg(t):
    return normal_ecg(t) - 0.3 * np.exp(-20 * (t - 0.1)**2)  # Slightly early depolarization

# Atrial premature beat (A)
def atrial_premature_beat_ecg(t):
    return normal_ecg(t) * (1 + 0.5 * np.sin(2 * np.pi * 8 * t))  # High-frequency noise in P wave

# Ventricular premature beat (V)
def ventricular_premature_beat_ecg(t):
    return -1.5 * np.exp(-100 * (t - 0.1)**2)  # Sharp abnormal QRS spike

# Generate signals for each type
ecg_N = normal_ecg(time)
ecg_L = left_bundle_block_ecg(time)
ecg_R = right_bundle_block_ecg(time)
ecg_A = atrial_premature_beat_ecg(time)
ecg_V = ventricular_premature_beat_ecg(time)

# Plot the waveforms
plt.figure(figsize=(12, 10))

# Normal (N)
plt.subplot(5, 1, 1)
plt.plot(time, ecg_N, label="N (Normal)", color='blue')
plt.title("Normal (N) ECG")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

# Left bundle branch block (L)
plt.subplot(5, 1, 2)
plt.plot(time, ecg_L, label="L (Left Bundle Branch Block)", color='green')
plt.title("Left Bundle Branch Block (L) ECG")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

# Right bundle branch block (R)
plt.subplot(5, 1, 3)
plt.plot(time, ecg_R, label="R (Right Bundle Branch Block)", color='orange')
plt.title("Right Bundle Branch Block (R) ECG")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

# Atrial premature beat (A)
plt.subplot(5, 1, 4)
plt.plot(time, ecg_A, label="A (Atrial Premature Beat)", color='purple')
plt.title("Atrial Premature Beat (A) ECG")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

# Ventricular premature beat (V)
plt.subplot(5, 1, 5)
plt.plot(time, ecg_V, label="V (Ventricular Premature Beat)", color='red')
plt.title("Ventricular Premature Beat (V) ECG")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
