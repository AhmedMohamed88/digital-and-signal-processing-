import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk 
from tkinter import filedialog
import copy
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

save_indeces=[]
save_samples=[]
save_samples_mult=[]
save_indeces_mult=[]


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
        global save_indeces , save_samples
        amplitude = float(amplitudetextbox.get())
        analog_frequence = float(analogfrequencetextbox.get())
        sampling_frequence = float(samplingfrequencetextbox.get())
        phaseshift = float(phaseshifttextbox.get())

        # Generate the signal
        indices, signal = signals('sin', amplitude, analog_frequence, sampling_frequence, phaseshift)
       
        save_indeces=indices
        save_samples=signal

        if indices is not None and signal is not None:
            # Display the signal
            signal_continuous(indices, signal)
            signal_discrete(indices, signal)      

       
    def cosine_wave():
        global  save_indeces , save_samples
        amplitude = float(amplitudetextbox.get())
        analog_frequence = float(analogfrequencetextbox.get())
        sampling_frequence = float(samplingfrequencetextbox.get())
        phaseshift = float(phaseshifttextbox.get())
        indices, signal = signals('cos', amplitude, analog_frequence, sampling_frequence, phaseshift)
        save_indeces=indices
        save_samples=signal

        if indices is not None and signal is not None:
            signal_continuous(indices, signal)
            signal_discrete(indices, signal)

    def SignalSamplesAreEqual(file_name,indices,samples):
        expected_indices=[]
        expected_samples=[]
        with open(file_name, 'r') as f:
            line = f.readline()
            line = f.readline()
            line = f.readline()
            line = f.readline()
            while line:
                # process line
                L=line.strip()
                if len(L.split(' '))==2:
                    L=line.split(' ')
                    V1=int(L[0])
                    V2=float(L[1])
                    expected_indices.append(V1)
                    expected_samples.append(V2)
                    line = f.readline()
                else:
                    break
                    
        if len(expected_samples)!=len(samples):
            print("Test case failed, your signal have different length from the expected one")
            return
        for i in range(len(expected_samples)):
            if abs(samples[i] - expected_samples[i]) < 0.01:
                continue
            else:
                print("Test case failed, your signal have different values from the expected one") 
                return
        print("Test case passed successfully")


    def compare_results():
      file=filedialog.askopenfilename()
      SignalSamplesAreEqual(file,save_indeces,save_samples)
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


 
 def SignalSamplesAreEqual(file_name, indices, samples):
    expected_indices = []
    expected_samples = []

    with open(file_name, 'r') as f:
        # Skip the first four lines
        for _ in range(3):
            f.readline()

        line = f.readline()
        while line:
            # Process line
            L = line.strip()
            if len(L.split(' ')) == 2:
                L = line.split(' ')
                V1 = int(L[0])
                V2 = float(L[1])
                expected_indices.append(V1)
                expected_samples.append(V2)
                line = f.readline()
            else:
                break

    # Debugging prints
    print("Expected Samples:", expected_samples)
    print("Number of expected samples:", len(expected_samples))
    print("Samples:", samples)
    print("Number of actual samples:", len(samples))

    if len(expected_samples) != len(samples):
        print("Test case failed, your signal have different length from the expected one")
        return

    for i in range(len(expected_samples)):
        if abs(samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Test case failed, your signal have different values from the expected one") 
            return

    print("Test case passed successfully")




 def browse_and_sum_signals():
    global save_samples
    file_path1 = filedialog.askopenfilename(title="Select First Signal File", filetypes=[("Text Files", "*.txt")])
    file_path2 = filedialog.askopenfilename(title="Select Second Signal File", filetypes=[("Text Files", "*.txt")])
    
    # Initialize lists to store samples
    sampels1 = []
    sampels2 = []
    
    # Read from the first file
    with open(file_path1, 'r') as f:
        # Skip the first three lines
        for _ in range(3):
            f.readline()
        
        line = f.readline()
        while line:
            L = line.strip()
            if len(L.split(' ')) == 2:
                L = line.split(' ')
                V2 = float(L[1])
                sampels1.append(V2)
                line = f.readline()
            else:
                break

    # Read from the second file
    with open(file_path2, 'r') as f:
        # Skip the first three lines
        for _ in range(3):
            f.readline()
        
        line = f.readline()
        while line:
            L = line.strip()
            if len(L.split(' ')) == 2:
                L = line.split(' ')
                V2 = float(L[1])
                sampels2.append(V2)
                line = f.readline()
            else:
                break

    # Print the samples for debugging
    print("Samples from file 1:", sampels1)
    print("Samples from file 2:", sampels2)

    # Check if both lists have the same length and are not empty
    if len(sampels1) == len(sampels2) and len(sampels1) > 0:
        # Create a new list to store the sum
        summed_samples = [a + b for a, b in zip(sampels1, sampels2)]
        save_samples = summed_samples
        print("Summed Samples:", save_samples)  # Print the summed results
    else:
        print("Error: The two sample lists are of different lengths or empty.")

        print(len(save_samples),"in the function")

        plt.plot(summed_samples, label='Resulting Signal')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.show()


 def browse_and_subtract_signals():
    global save_samples
    # Select the two signal files
    file_path1 = filedialog.askopenfilename(title="Select First Signal File", filetypes=[("Text Files", "*.txt")])
    file_path2 = filedialog.askopenfilename(title="Select Second Signal File", filetypes=[("Text Files", "*.txt")])

    # Initialize lists to store samples
    samples1 = []
    samples2 = []

    # Read from the first file
    with open(file_path1, 'r') as f:
        for _ in range(3):  # Skip the first three lines
            f.readline()
        
        line = f.readline()
        while line:
            L = line.strip()
            if len(L.split(' ')) == 2:
                L = line.split(' ')
                V2 = float(L[1])
                samples1.append(V2)
                line = f.readline()
            else:
                break

    # Read from the second file
    with open(file_path2, 'r') as f:
        for _ in range(3):  # Skip the first three lines
            f.readline()
        
        line = f.readline()
        while line:
            L = line.strip()
            if len(L.split(' ')) == 2:
                L = line.split(' ')
                V2 = float(L[1])
                samples2.append(V2)
                line = f.readline()
            else:
                break

    # Print the samples for debugging
    print("Samples from file 1:", samples1)
    print("Samples from file 2:", samples2)

    # Check if both lists have the same length and are not empty
    if len(samples1) == len(samples2) and len(samples1) > 0:
        # Subtract the samples from the two files element-wise
        subtracted_samples = [a - b for a, b in zip(samples1, samples2)]
        save_samples = subtracted_samples
        print("Subtracted Samples:", save_samples)  # Print the subtracted results

        # Plot the resulting signal
        plt.plot(subtracted_samples, label='Resulting Signal (File1 - File2)')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.show()
    else:
        print("Error: The two sample lists are of different lengths or empty.")

# Example usage
# browse_and_subtract_signals()

 def browse_and_multiply_signal():
    global   save_samples
    
    file_paths = filedialog.askopenfilenames(title="Select Signal Files", filetypes=[("Text Files", "*.txt")])
    
    constant = simpledialog.askinteger("Input", "Enter a constant value to multiply the signal (e.g., 2 for amplification, -1 for inversion):")
    
    if constant is None:  
        return
    
    signals = []
    for path in file_paths:
        with open(path, 'r') as f:
            signal = np.array([float(line.split()[1]) for line in f.readlines()[3:]])  # Skip first 3 lines (header)
            signals.append(signal)

    amplified_signals = np.array(signals[0]) * constant


    save_samples=amplified_signals

    print(len(save_samples))
    print(save_samples.shape)
    print(save_samples)

    plt.plot(amplified_signals[0], label='Amplified Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title(f'Signal Multiplied by {constant}')
    plt.show()

 def browse_and_square_signal():
    global save_samples

    # Open the dialog to select a single signal file
    file_path = filedialog.askopenfilename(title="Select Signal File", filetypes=[("Text Files", "*.txt")])

    # Initialize an empty list to store the signal
    signal = []

    # Read the signal from the file, skipping the first 3 lines (header)
    with open(file_path, 'r') as f:
        for _ in range(3):  # Skip header lines
            f.readline()

        for line in f:  # Read the remaining lines
            parts = line.strip().split()
            if len(parts) == 2:
                value = float(parts[1])  # Get the signal value
                signal.append(value)

    # Convert the list to a numpy array
    signal = np.array(signal)

    # Square the signal element-wise
    squared_signal = np.square(signal)

    # Save the squared signal globally
    save_samples = squared_signal

    # Print the squared signal for debugging
    print("Squared Signal:", squared_signal)

# Example usage (uncomment to test)
# browse_and_square_signal()

    
    plt.plot(browse_and_square_signal[0], label='Squared Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title('Signal Squared')
    plt.show()
 def browse_and_normalize_signal():
    global save_samples  # Store the normalized samples globally if needed

    # Ask the user to choose the normalization range
    choice = simpledialog.askstring("Input", 
                                    "Choose normalization range:\n'1' for [-1, 1]\n'2' for [0, 1]")

    if choice not in ['1', '2']:
        print("Invalid choice. Please select '1' or '2'.")
        return

    # Open the signal file (assuming one signal per file)
    file_paths = filedialog.askopenfilenames(
        title="Select Signal Files", 
        filetypes=[("Text Files", "*.txt")]
    )
    
    signals = []  # List to store signals

    # Read the signal(s) from the file(s)
    for path in file_paths:
        with open(path, 'r') as f:

            signal = np.array([float(line.split()[1]) for line in f.readlines()[3:]])
            signals.append(signal)


    signal = signals[0]
    if choice == '1':
        norm_signal = 2 * (signal - np.min(signal)) / (np.max(signal) - np.min(signal)) - 1
    else:
        norm_signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

    save_samples = norm_signal

    # Plot the normalized signal
    plt.plot(norm_signal, label='Normalized Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title(f'Signal Normalized to { "[-1, 1]" if choice == "1" else "[0, 1]"}')
    plt.show()
 
 def accumulate_signal():
    global save_samples 

    file_path = filedialog.askopenfilename(title="Select Signal File", filetypes=[("Text Files", "*.txt")])

    accumulated_signal = [0]  # Start with an initial value of 0

    # Read signal values from the input file
    with open(file_path, 'r') as f:
        # Skip the first three lines (headers)
        for _ in range(3):
            f.readline()

        # Process each signal value and accumulate
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                value = int(parts[1])  # Extract the signal value as integer
                accumulated_signal.append(accumulated_signal[-1] + value)
        accumulated_signal = accumulated_signal[1:]
        save_samples=accumulated_signal


        plt.plot(accumulated_signal, label=f'Accumulated Signal ')

        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.title('Accumulated Signal(s) (Cumulative Sum)')
        plt.show()
 def compare_results():
      file_paths = filedialog.askopenfilename()

      SignalSamplesAreEqual(file_paths,save_indeces_mult,save_samples)

 Button(child, text="Add waves", command=browse_and_sum_signals, font=('Arial', 12)).pack(pady=20)
 Button(child, text="Sub waves", command=browse_and_subtract_signals, font=('Arial', 12)).pack(pady=20)
 Button(child, text="Scale Wave", command=browse_and_multiply_signal, font=('Arial', 12)).pack(pady=20)
 Button(child, text="Square Wave", command=browse_and_square_signal, font=('Arial', 12)).pack(pady=20)
 Button(child, text="Normalize Wave", command=browse_and_normalize_signal, font=('Arial', 12)).pack(pady=20)
 Button(child, text="Accumulate Wave", command=accumulate_signal, font=('Arial', 12)).pack(pady=20)
 Button(child, text="Compare", command=compare_results, font=('Arial', 12)).pack(pady=20)



root = tk.Tk()

sinbutton = tk.Button(root, command=task1, text="Signal Generator", fg='black', font=('Arial', 15))
sinbutton.grid(row=5, column=1, sticky=tk.W, padx=5, pady=10)

AIrthmaticbutton = tk.Button(root, command=task2, text="Airathmatic Operations", fg='black', font=('Arial', 15))
AIrthmaticbutton.grid(row=5, column=3, sticky=tk.W, padx=5, pady=10)

print(len(save_samples), "out of function")

root.mainloop()

