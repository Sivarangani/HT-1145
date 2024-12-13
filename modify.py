import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import butter, filtfilt
import csv
import os
import numpy as np
from numpy.linalg import svd
# Your CVSS-NLMS and SLPF functions (from the previous answer)
# -------------------
def cvss_nlms_filter(signal, desired_signal, mu_init=0.01, alpha=0.98, step_size_range=(0.01, 1.0)):
    step_size = mu_init
    filtered_signal = np.zeros_like(signal)
    for n in range(1, len(signal)):
        error = desired_signal[n] - filtered_signal[n-1]
        norm_factor = np.dot(signal[n-1], signal[n-1]) + 1e-5
        step_size = np.clip(alpha * step_size, step_size_range[0], step_size_range[1])
        filtered_signal[n] = filtered_signal[n-1] + step_size * error / norm_factor * signal[n]
    return filtered_signal


def slpf_filter(signal, rank_threshold=0.9, window_size=500):
    denoised_signal = np.zeros_like(signal)
    half_window = window_size // 2

    for start in range(0, len(signal), half_window):
        end = min(start + window_size, len(signal))
        window_signal = signal[start:end]

        if len(window_signal) < window_size:
            break  # Skip the last window if it's too small

        # Create Hankel matrix
        hankel_matrix = np.array([window_signal[i: i + half_window] for i in range(half_window)])

        # Perform SVD
        U, S, Vt = svd(hankel_matrix, full_matrices=False)

        # Rank selection based on cumulative energy
        cumulative_energy = np.cumsum(S) / np.sum(S)
        r = np.argmax(cumulative_energy > rank_threshold) + 1

        # Low-rank approximation
        low_rank_matrix = np.dot(U[:, :r], np.dot(np.diag(S[:r]), Vt[:r, :]))
        denoised_signal[start:end] = np.mean(low_rank_matrix, axis=0)

    return denoised_signal

def butterworth_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y
# -------------------

# File reading
path = 'mitbih_database'
window_size = 180
maximum_counting = 10000

classes = ['N', 'L', 'R', 'A', 'V']
n_classes = len(classes)
count_classes = [0]*n_classes

X = list()
y = list()

# Read files
filenames = next(os.walk(path))[2]
records = []
annotations = []
filenames.sort()

# Segregate filenames and annotations
for f in filenames:
    filename, file_extension = os.path.splitext(f)
    if file_extension == '.csv':
        records.append(path + "/" + filename + file_extension)
    else:
        annotations.append(path + "/" + filename + file_extension)

print(records[0])
print(annotations[0])

# Sampling frequency
fs = 360  # Assuming 360 Hz for MIT-BIH dataset

# Records
for r in range(0, len(records)):
    signals = []

    # Read CSV signals
    with open(records[r], 'rt') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        row_index = -1
        for row in spamreader:
            if row_index >= 0:
                signals.insert(row_index, int(row[1]))
            row_index += 1

    # Preprocess Signals: Apply Butterworth, CVSS-NLMS, SLPF
    signals = stats.zscore(signals)
    baseline_removed = butterworth_filter(signals, lowcut=0.5, highcut=50, fs=fs, order=4)
    desired_signal = np.zeros_like(signals)  # Assuming zero as baseline
    nlms_filtered = cvss_nlms_filter(baseline_removed, desired_signal)
    denoised_signal = slpf_filter(nlms_filtered)

    # Plot example denoised signal
    if r == 1:
        plt.title(records[1] + " wave after denoising (CVSS-NLMS + SLPF)")
        plt.plot(denoised_signal[0:700])
        plt.show()

    # Read annotations
    with open(annotations[r], 'r') as fileID:
        data = fileID.readlines()
        beat = list()
        example_beat_printed = False

        for d in range(1, len(data)):
            splitted = data[d].split(' ')
            splitted = filter(None, splitted)
            next(splitted)  # Skip time
            pos = int(next(splitted))  # Sample ID
            arrhythmia_type = next(splitted)  # Type

            if arrhythmia_type in classes:
                arrhythmia_index = classes.index(arrhythmia_type)
                count_classes[arrhythmia_index] += 1
                if window_size <= pos < (len(denoised_signal) - window_size):
                    beat = denoised_signal[pos - window_size:pos + window_size]
                    if r == 1 and not example_beat_printed:
                        plt.title("A Beat from " + records[1])
                        plt.plot(beat)
                        plt.show()
                        example_beat_printed = True

                    X.append(beat)
                    y.append(arrhythmia_index)

# Convert X and y to arrays
X = np.array(X)
y = np.array(y)

print("Processed beats shape:", X.shape)
print("Labels shape:", y.shape)
