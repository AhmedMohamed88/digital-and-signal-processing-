import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk 
from tkinter import filedialog
import cmath
import math
from math import ceil
import os
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk ,Toplevel, Label, Button
from tkinter import messagebox , simpledialog
from tkinter import filedialog
from scipy.signal import convolve


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
def load_data():
    file_path = filedialog.askopenfile()
    data = np.loadtxt(file_path, skiprows=3)
    if data.shape[1] >= 2:
        indices = data[:, 0]
        samples = data[:, 1]
        signal_continuous(indices, samples)
        signal_discrete(indices, samples)
    else:
        print("The file must have at least two columns.")
def task1():

    
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


    def sine_wave():
        amplitude = float(amplitudetextbox.get())
        analog_frequence = float(analogfrequencetextbox.get())
        sampling_frequence = float(samplingfrequencetextbox.get())
        phaseshift = float(phaseshifttextbox.get())

        # Generate the signal
        indices, signal = signals('sin', amplitude, analog_frequence, sampling_frequence, phaseshift)
       
        if indices is not None and signal is not None:
            # Display the signal
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




    def SignalSamplesAreEqual(file_original):
        # Ask user for generated file
        file_generated = filedialog.askopenfilename()
        
        indices_1 = []
        samples_1 = []
        indices_2 = []
        samples_2 = []
        
        # Read original file
        with open(file_original, 'r') as f:
            # Skip the first four lines (assuming headers)
            for _ in range(4):
                next(f)

            for line in f:
                L = line.strip()
                if len(L.split(',')) == 2:  # Assuming CSV format
                    V1, V2 = L.split(',')
                    indices_1.append(int(V1))
                    samples_1.append(float(V2))
        
        # Read generated file
        with open(file_generated, 'r') as f:
            # Skip the first four lines (assuming headers)
            for _ in range(4):
                next(f)

            for line in f:
                L = line.strip()
                if len(L.split(',')) == 2:  # Assuming CSV format
                    V1, V2 = L.split(',')
                    indices_2.append(int(V1))
                    samples_2.append(float(V2))
        
        # Compare samples
        if len(samples_1) != len(samples_2):
            print("Test case failed: your signal has different lengths from the expected one.")
            return

        for i in range(len(samples_1)):
            if abs(samples_2[i] - samples_1[i]) < 0.01:
                continue
            else:
                print("Test case failed: your signal has different values from the expected one.") 
                return

        print("Test case passed successfully.")

    # Wrapper function for the Compare button to pass dynamic values
    def compare_results():
        SignalSamplesAreEqual("SinOutput.txt")

    # Create the GUI window
    window = tk.Tk()
    window.title("Signal Generator")

    tk.Label(window, text="Enter amplitude", fg='black', font=('Arial', 15)).grid(row=0, column=0, padx=5, pady=10)
    amplitudetextbox = tk.Entry(window, fg='black', font=('Arial', 15))
    amplitudetextbox.grid(row=0, column=1)

    tk.Label(window, text="Enter analog frequency", fg='black', font=('Arial', 15)).grid(row=1, column=0, padx=5, pady=10)
    analogfrequencetextbox = tk.Entry(window, fg='black', font=('Arial', 15))
    analogfrequencetextbox.grid(row=1, column=1)

    tk.Label(window, text="Enter sampling frequency", fg='black', font=('Arial', 15)).grid(row=2, column=0, padx=5, pady=10)
    samplingfrequencetextbox = tk.Entry(window, fg='black', font=('Arial', 15))
    samplingfrequencetextbox.grid(row=2, column=1)

    tk.Label(window, text="Enter phase shift", fg='black', font=('Arial', 15)).grid(row=3, column=0, padx=5, pady=10)
    phaseshifttextbox = tk.Entry(window, fg='black', font=('Arial', 15))
    phaseshifttextbox.grid(row=3, column=1)

    sinbutton = tk.Button(window, command=sine_wave, text="Sine Wave", fg='black', font=('Arial', 15))
    sinbutton.grid(row=5, column=0, sticky=tk.W, padx=5, pady=10)

    cosbutton = tk.Button(window, command=cosine_wave, text="Cosine Wave", fg='black', font=('Arial', 15))
    cosbutton.grid(row=5, column=1, sticky=tk.W, padx=5, pady=10)

    loabutton = tk.Button(window, command=load_data, text="Read Data", fg='black', font=('Arial', 15))
    loabutton.grid(row=5, column=2, sticky=tk.W, padx=5, pady=10)

    compare_button = tk.Button(window, command=compare_results, text="Compare Results", fg='black', font=('Arial', 15))
    compare_button.grid(row=5, column=3, sticky=tk.W, padx=5, pady=10)

    signal_output_box = tk.Text(window, height=10, width=50, font=('Arial', 12))
    signal_output_box.grid(row=6, column=0, columnspan=2, padx=5, pady=10)

    window.mainloop()

def task2():
 child = Toplevel(root)
 child.title("Airthmatic Window")
 child.geometry("300x200")
 def browse_and_sum_signals():
        file_paths = filedialog.askopenfilenames(title="Select Signal Files", filetypes=[("Text Files", "*.txt")])
    
        signals = []
        for path in file_paths:
        # Load each signal from the file
         with open(path, 'r') as f:
            signal = np.array([float(line.split()[1]) for line in f.readlines()[3:]])  # Skip first 3 lines (header)
            signals.append(signal)

    # Ensure all signals are of the same length before summing
        min_length = min(len(s) for s in signals)
        truncated_signals = [s[:min_length] for s in signals]

    # Sum the signals element-wise
        result_signal = sum(truncated_signals)

    # Plot the result
        plt.plot(result_signal, label='Resulting Signal')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.show()
        # Call the function to run the program
 def browse_and_subtract_signals():
    file_paths = filedialog.askopenfilenames(title="Select Signal Files", filetypes=[("Text Files", "*.txt")])
    
    signals = []
    for path in file_paths:
        # Load each signal from the file
        with open(path, 'r') as f:
            signal = np.array([float(line.split()[1]) for line in f.readlines()[3:]])  # Skip first 3 lines (header)
            signals.append(signal)

    # Ensure all signals are of the same length before subtracting
    min_length = min(len(s) for s in signals)
    truncated_signals = [s[:min_length] for s in signals]

    # Subtract signals element-wise (subtracting all signals from the first signal)
    result_signal = truncated_signals[0] - sum(truncated_signals[1:], np.zeros(min_length))

    # Plot the result
    plt.plot(result_signal, label='Resulting Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()
 def browse_and_multiply_signal():

    
    file_paths = filedialog.askopenfilenames(title="Select Signal Files", filetypes=[("Text Files", "*.txt")])
    
    # Prompt user for a constant value to multiply the signal
    constant = simpledialog.askfloat("Input", "Enter a constant value to multiply the signal (e.g., 2 for amplification, -1 for inversion):")
    
    if constant is None:  # Check if the user canceled the input dialog
        return
    
    signals = []
    for path in file_paths:
        # Load each signal from the file
        with open(path, 'r') as f:
            signal = np.array([float(line.split()[1]) for line in f.readlines()[3:]])  # Skip first 3 lines (header)
            signals.append(signal)

    # Ensure all signals are of the same length before multiplication
    min_length = min(len(s) for s in signals)
    truncated_signals = [s[:min_length] for s in signals]

    # Multiply each signal by the constant value
    amplified_signals = [s * constant for s in truncated_signals]

    # If you want to visualize the first amplified signal
    plt.plot(amplified_signals[0], label='Amplified Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title(f'Signal Multiplied by {constant}')
    plt.show()
 def browse_and_square_signal():
    # Create a hidden Tkinter root window


    file_paths = filedialog.askopenfilenames(title="Select Signal Files", filetypes=[("Text Files", "*.txt")])
    
    signals = []
    for path in file_paths:
        # Load each signal from the file
        with open(path, 'r') as f:
            signal = np.array([float(line.split()[1]) for line in f.readlines()[3:]])  # Skip first 3 lines (header)
            signals.append(signal)

    # Ensure all signals are of the same length before squaring
    min_length = min(len(s) for s in signals)
    truncated_signals = [s[:min_length] for s in signals]

    # Square each signal
    squared_signals = [np.square(s) for s in truncated_signals]

    # If you want to visualize the first squared signal
    plt.plot(squared_signals[0], label='Squared Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title('Signal Squared')
    plt.show()
 def browse_and_normalize_signal():
    # Create a hidden Tkinter root window
  
    # Ask the user for the normalization range choice
    choice = simpledialog.askstring("Input", "Choose normalization range:\n'1' for [-1, 1]\n'2' for [0, 1]")

    # Validate user input
    if choice not in ['1', '2']:
        print("Invalid choice. Please select '1' or '2'.")
        return

    file_paths = filedialog.askopenfilenames(title="Select Signal Files", filetypes=[("Text Files", "*.txt")])
    
    signals = []
    for path in file_paths:
        # Load each signal from the file
        with open(path, 'r') as f:
            signal = np.array([float(line.split()[1]) for line in f.readlines()[3:]])  # Skip first 3 lines (header)
            signals.append(signal)

    # Ensure all signals are of the same length
    min_length = min(len(s) for s in signals)
    truncated_signals = [s[:min_length] for s in signals]

    # Normalize each signal
    normalized_signals = []
    for s in truncated_signals:
        if choice == '1':
            # Normalize to [-1, 1]
            norm_signal = 2 * (s - np.min(s)) / (np.max(s) - np.min(s)) - 1
        else:
            # Normalize to [0, 1]
            norm_signal = (s - np.min(s)) / (np.max(s) - np.min(s))
        
        normalized_signals.append(norm_signal)

    # If you want to visualize the first normalized signal
    plt.plot(normalized_signals[0], label='Normalized Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title(f'Signal Normalized to { "[-1, 1]" if choice == "1" else "[0, 1]"}')
    plt.show()
 def browse_and_accumulate_signal():
 
    file_paths = filedialog.askopenfilenames(title="Select Signal Files", filetypes=[("Text Files", "*.txt")])
    
    signals = []
    for path in file_paths:
        # Load each signal from the file
        with open(path, 'r') as f:
            signal = np.array([float(line.split()[1]) for line in f.readlines()[3:]])  # Skip first 3 lines (header)
            signals.append(signal)

    # Ensure all signals are of the same length
    min_length = min(len(s) for s in signals)
    truncated_signals = [s[:min_length] for s in signals]

    # Accumulate each signal
    accumulated_signals = [np.cumsum(s) for s in truncated_signals]

    # If you want to visualize the first accumulated signal
    plt.plot(accumulated_signals[0], label='Accumulated Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title('Accumulated Signal (Cumulative Sum)')
    plt.show()

 Button(child, text="Add waves", command=browse_and_sum_signals, font=('Arial', 12)).pack(pady=20)
 Button(child, text="Sub waves", command=browse_and_subtract_signals, font=('Arial', 12)).pack(pady=20)
 Button(child, text="Scale Wave", command=browse_and_multiply_signal, font=('Arial', 12)).pack(pady=20)
 Button(child, text="Square Wave", command=browse_and_square_signal, font=('Arial', 12)).pack(pady=20)
 Button(child, text="Normalize Wave", command=browse_and_normalize_signal, font=('Arial', 12)).pack(pady=20)
 Button(child, text="Accumulate Wave", command=browse_and_accumulate_signal, font=('Arial', 12)).pack(pady=20)



root = tk.Tk()

sinbutton = tk.Button(root, command=task1, text="Signal Generator", fg='black', font=('Arial', 15))
sinbutton.grid(row=5, column=1, sticky=tk.W, padx=5, pady=10)

AIrthmaticbutton = tk.Button(root, command=task2, text="Airathmatic Operations", fg='black', font=('Arial', 15))
AIrthmaticbutton.grid(row=5, column=3, sticky=tk.W, padx=5, pady=10)

root.mainloop()

