import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 23

# Function to generate a synthetic ECG-like signal
def generate_ecg_signal(length=700, noise_level=0.2):
    time = np.linspace(0, 1, length)
    
    # ECG-like waveform (approximated PQRST complex)
    ecg_wave = (
        np.sin(2 * np.pi * 1 * time) * 1.2 +  # Baseline fluctuation
        np.exp(-((time - 0.1) * 100) ** 2) * 2 +  # P-wave
        -np.exp(-((time - 0.2) * 200) ** 2) * 5 +  # Q-wave
        np.exp(-((time - 0.3) * 150) ** 2) * 8 +  # R-wave
        -np.exp(-((time - 0.4) * 200) ** 2) * 5 +  # S-wave
        np.exp(-((time - 0.6) * 100) ** 2) * 2    # T-wave
    )
    
    # Add Gaussian noise
    noise = np.random.normal(0, noise_level, length)
    noisy_ecg = ecg_wave + noise
    return time, ecg_wave, noisy_ecg

# Moving average for denoising
def moving_average(signal, window_size=10):
    return np.convolve(signal, np.ones(window_size) / window_size, mode='same')

# Butterworth low-pass filter for denoising
def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# Generate synthetic ECG signal
length = 700
time, clean_ecg, noisy_ecg = generate_ecg_signal(length=length, noise_level=0.3)

# Apply noise reduction techniques
denoised_ecg_moving_avg = moving_average(noisy_ecg, window_size=15)  # Moving average
denoised_ecg_butter = butter_lowpass_filter(noisy_ecg, cutoff=10, fs=700, order=4)  # Butterworth filter

# Plotting
plt.figure(figsize=(12, 10))

# Original clean ECG
plt.subplot(4, 1, 1)
plt.plot(time, clean_ecg, label="Clean ECG Signal", color='blue')
plt.title("Clean ECG Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()

# Noisy ECG
plt.subplot(4, 1, 2)
plt.plot(time, noisy_ecg, label="Noisy ECG Signal", color='orange')
plt.title("Noisy ECG Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()

# Denoised using Moving Average
plt.subplot(4, 1, 3)
plt.plot(time, denoised_ecg_moving_avg, label="Denoised ECG (Moving Average)", color='green')
plt.title("Denoised ECG Signal (Moving Average)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()

# Denoised using Butterworth Filter
plt.subplot(4, 1, 4)
plt.plot(time, denoised_ecg_butter, label="Denoised ECG (Butterworth Filter)", color='red')
plt.title("Denoised ECG Signal (Butterworth Filter)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()

plt.tight_layout()
plt.show()
