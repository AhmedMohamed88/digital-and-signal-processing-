import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog


def signals(choice, amplitude, analog_frequence, sampling_frequence, phaseshift):
    t = np.arange(0, 1, 1 / sampling_frequence)
    if choice == 'sin':
        signal = amplitude * np.sin(2 * np.pi * analog_frequence * t + phaseshift)
    elif choice == 'cos':
        signal = amplitude * np.cos(2 * np.pi * analog_frequence * t + phaseshift)
    else:
        return None
    indices = np.arange(len(signal))
    return indices, signal



def signal_continuous(indices, samples):
    plt.plot(indices, samples)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Continuous Signal')
    plt.show()


def signal_discrete(indices, samples):
    plt.stem(indices, samples)
    plt.xlabel('Sample Index (n)')
    plt.ylabel('Amplitude')
    plt.title('Discrete Signal')
    plt.show()


def sine_wave():
    amplitude = float(amplitudetextbox.get())
    analog_frequence = float(analogfrequencetextbox.get())
    sampling_frequence = float(samplingfrequencetextbox.get())
    phaseshift = float(phaseshifttextbox.get())

    indices, signal = signals('sin', amplitude, analog_frequence, sampling_frequence, phaseshift)
    if indices is not None and signal is not None:
        signal_continuous(indices, signal)
        signal_discrete(indices, signal)


def cosine_wave():
    amplitude = float(amplitudetextbox.get())
    analog_frequence = float(analogfrequencetextbox.get())
    sampling_frequence = float(samplingfrequencetextbox.get())
    phaseshift = float(phaseshifttextbox.get())

    indices, signal = signals('cos', amplitude, analog_frequence, sampling_frequence, phaseshift)
    if indices is not None and signal is not None:
        signal_continuous(indices, signal)
        signal_discrete(indices, signal)


def load_data(file_path):
    file_path="signal1.txt"
    # Load the data skipping the first 3 rows
    data = np.loadtxt(file_path, skiprows=3)
    if data.shape[1] >= 2:
        indices = data[:, 0]  # First column as indices
        samples = data[:, 1]  # Second column as samples
        signal_continuous(indices, samples)
        signal_discrete(indices, samples)
    else:
        print("The file must have at least two columns.")


window = tk.Tk()
window.title("Signal Generator")


tk.Label(window, text="Enter amplitude", fg='black', font=('Arial', 15)).grid(row=0, column=0, padx=5, pady=10)
amplitudetextbox = tk.Entry(window, fg='black', font=('Arial', 15))
amplitudetextbox.grid(row=0, column=1)

tk.Label(window, text="Enter analog frequency", fg='orange', font=('Arial', 15)).grid(row=1, column=0, padx=5, pady=10)
analogfrequencetextbox = tk.Entry(window, fg='black', font=('Arial', 15))
analogfrequencetextbox.grid(row=1, column=1)

tk.Label(window, text="Enter sampling frequency", fg='green', font=('Arial', 15)).grid(row=2, column=0, padx=5, pady=10)
samplingfrequencetextbox = tk.Entry(window, fg='black', font=('Arial', 15))
samplingfrequencetextbox.grid(row=2, column=1)

tk.Label(window, text="Enter phase shift", fg='red', font=('Arial', 15)).grid(row=3, column=0, padx=5, pady=10)
phaseshifttextbox = tk.Entry(window, fg='black', font=('Arial', 15))
phaseshifttextbox.grid(row=3, column=1)

sinbutton = tk.Button(window, command=sine_wave, text="Sine Wave", fg='blue', font=('Arial', 15))
sinbutton.grid(row=5, column=0, sticky=tk.W, padx=5, pady=10)

cosbutton = tk.Button(window, command=cosine_wave, text="Cosine Wave", fg='blue', font=('Arial', 15))
cosbutton.grid(row=5, column=1, sticky=tk.W, padx=5, pady=10)

loabutton = tk.Button(window, command=load_data("signal1.txt"), text="read data", fg='blue', font=('Arial', 15))
loabutton.grid(row=5, column=3, sticky=tk.W, padx=5, pady=10)

signal_output_box = tk.Text(window, height=10, width=50, font=('Arial', 12))
signal_output_box.grid(row=6, column=0, columnspan=2, padx=5, pady=10)

window.mainloop()