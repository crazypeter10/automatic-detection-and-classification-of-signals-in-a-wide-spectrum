import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# Generăm un semnal simplu (sinusoidal + zgomot variabil)
def generate_signal(freq, noise_level=None, duration=0.01, sampling_rate=2_000_000):
    if noise_level is None:
        noise_level = np.random.uniform(0.1, 0.5)  # zgomot variabil
    t = np.linspace(0, duration, int(sampling_rate * duration))
    signal = np.sin(2 * np.pi * freq * t) + noise_level * np.random.randn(len(t))
    return t, signal

# Funcție pentru a salva spectrograma
def save_spectrogram(signal, label, file_name, sampling_rate=2_000_000):
    freqs, times, Sxx = spectrogram(signal, fs=sampling_rate)
    plt.figure(figsize=(5, 5))
    plt.pcolormesh(times, freqs, 10 * np.log10(Sxx))
    plt.ylim(0, 1_000_000)  # Limitează la 1 MHz pentru claritate
    plt.axis('off')
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    plt.close()

# Generăm date și salvăm spectrograme pentru fiecare categorie
def generate_and_save_data(data_dir, num_samples=100):
    categories = ["sinusoidal", "noisy", "interference"]
    for category in categories:
        os.makedirs(os.path.join(data_dir, category), exist_ok=True)

    for i in range(num_samples):
        if i % 3 == 0:
            freq = np.random.uniform(1, 1_000_000)  # Intervalul frecvenței între 1 Hz și 1 MHz
            _, signal = generate_signal(freq)
            save_spectrogram(signal, "sinusoidal", f"{data_dir}/sinusoidal/{i}.png")
        
        elif i % 3 == 1:
            freq = np.random.uniform(1, 1_000_000)
            _, signal = generate_signal(freq, noise_level=np.random.uniform(0.3, 0.7))
            save_spectrogram(signal, "noisy", f"{data_dir}/noisy/{i}.png")
        
        else:
            freq1, freq2 = np.random.uniform(1, 1_000_000, 2)
            t, signal1 = generate_signal(freq1)
            _, signal2 = generate_signal(freq2)
            signal = signal1 + signal2
            if np.random.rand() > 0.5:
                _, signal3 = generate_signal(np.random.uniform(1, 1_000_000))
                signal += signal3  # adăugăm un al treilea semnal pentru complexitate
            save_spectrogram(signal, "interference", f"{data_dir}/interference/{i}.png")

# Generăm datele de antrenament și test cu noile îmbunătățiri
if __name__ == "__main__":
    generate_and_save_data("data/train", num_samples=200)   # mai multe mostre pentru antrenament
    generate_and_save_data("data/test", num_samples=50)     # mostre pentru testare
