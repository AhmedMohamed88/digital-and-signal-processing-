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
from tkinter import messagebox , simpledialog,scrolledtext
from tkinter import filedialog
from scipy.signal import convolve

save_indeces=[]
save_samples=[]
save_samples_mult=[]
save_indeces_mult=[]
save_window_size=0


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
      SignalSamplesAreEqual("SinOutput.txt",save_indeces,save_samples)
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
            # Skip the first three lines (header) and read the signal values
            signal = np.array([float(line.split()[1]) for line in f.readlines()[3:]])
            signals.append(signal)

    # Normalize the first signal (or all if needed)
    signal = signals[0]  # Use the first signal
    if choice == '1':  # Normalize to [-1, 1]
        norm_signal = 2 * (signal - np.min(signal)) / (np.max(signal) - np.min(signal)) - 1
    else:  # Normalize to [0, 1]
        norm_signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

    save_samples = norm_signal  # Save normalized signal for further use

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

def task3():
 



# Global variables
    file_path = ""
    amplitude = None
    min_val = None
    max_val = None
    levels = None
    delta = None
    quantized_values = None
    quantization_labels = None  # Store encoded values (labels) for comparison
    sampled_error = None  # Store quantization error


    def load_data():
        """Load data from a file."""
        global file_path, amplitude, min_val, max_val
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if file_path:
            file_label.config(text=file_path)
            try:
                data = np.loadtxt(file_path, skiprows=3)
                amplitude = data[:, 1]  # Amplitude column
                min_val, max_val = np.min(amplitude), np.max(amplitude)
                messagebox.showinfo("Success", "Data loaded successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Error loading data: {e}")


    def update_input_label():
        """Update the input label based on the selected choice."""
        if choice_var.get() == "bits":
            input_label.config(text="Enter Number of Bits:")
        else:
            input_label.config(text="Enter Number of Levels:")


    def perform_quantization():
        """Perform the quantization based on user input."""
        global levels, delta, quantized_values, quantization_labels, sampled_error
        if amplitude is None:
            messagebox.showwarning("Warning", "Please load the data file first.")
            return

        try:
            user_input = int(input_entry.get())
            if choice_var.get() == "bits":
                levels = 2 ** user_input
                bits = user_input
            else:
                levels = user_input
                bits = int(np.log2(levels))  # Calculate bits based on levels

            delta = (max_val - min_val) / levels

            # Create quantization ranges and midpoints
            ranges = [(min_val + i * delta, min_val + (i + 1) * delta) for i in range(levels)]
            midpoints = [(start + end) / 2 for start, end in ranges]

            quantized_values = np.zeros_like(amplitude)
            quantization_labels = []  # Store the encoded levels (labels)
            sampled_error = []  # Store the quantization errors

            for i, value in enumerate(amplitude):
                for j, (start, end) in enumerate(ranges):
                    if start <= value < end:
                        quantized_values[i] = midpoints[j]
                        # Calculate the quantization error
                        error = quantized_values[i] - amplitude[i]
                        sampled_error.append(error)  # Store error
                        # Format the label to have leading zeros
                        binary_representation = format(j, f'0{bits}b')  # Use bits for formatting
                        quantization_labels.append(binary_representation)  # Store label as binary string
                        break
                else:
                    if value == max_val:
                        quantized_values[i] = midpoints[-1]
                        error = quantized_values[i] - amplitude[i]
                        sampled_error.append(error)  # Store error
                        binary_representation = format(levels - 1, f'0{bits}b')  # Format the label for max value
                        quantization_labels.append(binary_representation)

            # Display quantized values and errors
            result = ""
            for quant, label, error in zip(quantized_values, quantization_labels, sampled_error):
                result += f"Quantized Value: {quant:.4f}  Bit: {label}  Error: {error:.4f}\n"

            result_label.config(text=result)
            messagebox.showinfo("Quantization Complete", "Quantization completed successfully.")

        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for bits or levels.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during quantization: {e}")


    def read_test_output(file_name):
        """Read the expected output from the test file."""
        expected_indices = []
        expected_encoded_values = []
        expected_quantized_values = []
        expected_errors = []

        with open(file_name, 'r') as f:
            for line in f:
                if line.strip():  # Check for non-empty line
                    parts = line.split()
                    if len(parts) == 4:  # Expecting 4 values in the output
                        expected_indices.append(int(parts[0]))
                        expected_encoded_values.append(parts[1])
                        expected_quantized_values.append(float(parts[2]))
                        expected_errors.append(float(parts[3]))

        return expected_indices, expected_encoded_values, expected_quantized_values, expected_errors


    def compare_quantization():
        """Compare the quantized values using QuantizationTest1."""
        if not file_path:
            messagebox.showwarning("Warning", "Please load the data file first.")
            return
        if quantized_values is None or quantization_labels is None:
            messagebox.showwarning("Warning", "Please perform quantization first.")
            return

        test_file = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if test_file:
            expected_indices, expected_encoded_values, expected_quantized_values, expected_errors = read_test_output(
                test_file)
            if len(expected_encoded_values) != len(quantization_labels):
                print("QuantizationTest1 Test case failed, your signal has different lengths from the expected one")
                return

            for i in range(len(quantization_labels)):
                if quantization_labels[i] != expected_encoded_values[i]:
                    print(
                        f"QuantizationTest1 Test case failed at index {i}, your EncodedValue: {quantization_labels[i]}, Expected: {expected_encoded_values[i]}")
                    return

                if abs(quantized_values[i] - expected_quantized_values[i]) >= 0.01:
                    print(
                        f"QuantizationTest1 Test case failed at index {i}, your QuantizedValue: {quantized_values[i]}, Expected: {expected_quantized_values[i]}")
                    return

            print("QuantizationTest1 Test case passed successfully")


    def compare_quantization_levels():
        """Compare the quantized values using QuantizationTest2."""
        if not file_path:
            messagebox.showwarning("Warning", "Please load the data file first.")
            return
        if quantized_values is None or quantization_labels is None or sampled_error is None:
            messagebox.showwarning("Warning", "Please perform quantization first.")
            return

        test_file = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if test_file:
            expected_indices, expected_encoded_values, expected_quantized_values, expected_errors = read_test_output(
                test_file)

            # Generate expected interval indices (0-based for comparison)
            interval_indices = list(range(len(quantization_labels)))
            if len(expected_encoded_values) != len(quantization_labels) or len(expected_quantized_values) != len(
                    quantized_values) or len(expected_errors) != len(sampled_error):
                print("QuantizationTest2 Test case failed, your signal has different lengths from the expected one")
                return

            for i in range(len(quantization_labels)):
                if quantization_labels[i] != expected_encoded_values[i]:
                    print(
                        f"QuantizationTest2 Test case failed at index {i}, your EncodedValue: {quantization_labels[i]}, Expected: {expected_encoded_values[i]}")
                    return

                if abs(quantized_values[i] - expected_quantized_values[i]) >= 0.01:
                    print(
                        f"QuantizationTest2 Test case failed at index {i}, your QuantizedValue: {quantized_values[i]}, Expected: {expected_quantized_values[i]}")
                    return

                if abs(sampled_error[i] - expected_errors[i]) >= 0.01:
                    print(
                        f"QuantizationTest2 Test case failed at index {i}, your Error: {sampled_error[i]}, Expected: {expected_errors[i]}")
                    return

            print("QuantizationTest2 Test case passed successfully")


    # GUI Setup
    root = tk.Tk()
    root.title("Quantization Tool")

    frame = tk.Frame(root)
    frame.pack(pady=10)

    file_label = tk.Label(frame, text="No file loaded")
    file_label.pack()

    load_button = tk.Button(frame, text="Load Data", command=load_data)
    load_button.pack(pady=5)

    choice_var = tk.StringVar(value="levels")
    tk.Radiobutton(frame, text="Number of Levels", variable=choice_var, value="levels", command=update_input_label).pack()
    tk.Radiobutton(frame, text="Number of Bits", variable=choice_var, value="bits", command=update_input_label).pack()

    input_label = tk.Label(frame, text="Enter Number of Levels:")
    input_label.pack()

    input_entry = tk.Entry(frame)
    input_entry.pack()

    quantize_button = tk.Button(frame, text="Quantize", command=perform_quantization)
    quantize_button.pack(pady=5)

    compare_button = tk.Button(frame, text="Compare Quantization (Test 1)", command=compare_quantization)
    compare_button.pack(pady=5)

    compare_levels_button = tk.Button(frame, text="Compare Quantization (Test 2)", command=compare_quantization_levels)
    compare_levels_button.pack(pady=5)

    result_label = tk.Label(frame, text="", justify=tk.LEFT)
    result_label.pack()

    root.mainloop()

def task4():
 def read_signal_from_file2():
    filename = filedialog.askopenfilename(title="Select a Signal File")
    with open(filename, 'r') as file:
        lines = file.readlines()

    signal_type = int(lines[1].strip())

    samples = []

    if signal_type == 0:  # Time domain
        for line in lines[3:]:
            values = line.strip().split()
            if len(values) == 2:
                index, amplitude = values
                samples.append((float(index), float(amplitude)))
        return [sample[0] for sample in samples], [sample[1] for sample in samples]

    elif signal_type == 1:  # Frequency domain
        complex_numbers = []
        for line in lines[3:]:
            values = line.strip().split()
            if len(values) == 2:
                amplitude, phase = values
                amplitude = float(amplitude)
                phase = float(phase)
                # apply euler formoula
                real_part = amplitude * np.cos(float(phase))
                imaginary_part = amplitude * np.sin(float(phase))
                complex_number = real_part + 1j * imaginary_part
                complex_numbers.append(complex_number)
        return complex_numbers
 def dft(signal, sampling_frequency):
    N = len(signal)
    frequencies = np.zeros(N)
    amplitude = np.zeros(N)
    phase = np.zeros(N)
    for k in range(N):
        frequencies[k] = 2 * np.pi * k * sampling_frequency / N
        dft_sum = 0
        for n in range(N):
            complex_exp = np.exp(-2j * np.pi * k * n / N)
            dft_sum += signal[n] * complex_exp

        amplitude[k] = np.sqrt(dft_sum.real ** 2 + dft_sum.imag ** 2)
        phase[k] = np.arctan2(dft_sum.imag, dft_sum.real)

   
    
        plt.figure(figsize=(8, 4))
    plt.stem(frequencies, phase, basefmt="", label='samples')
    plt.xlabel('freq')
    plt.ylabel('phase')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.stem(frequencies, amplitude, basefmt="", label='samples')
    plt.xlabel('freq')
    plt.ylabel('amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()

    return frequencies, amplitude, phase
 def idft(complex_nums):
    N = len(complex_nums)

    signal = np.zeros(N, dtype=complex)

    for n in range(N):
        idft_sum = 0
        for k in range(N):
            complex_exp = np.exp(2j * np.pi * k * n / N)
            idft_sum += complex_nums[k] * complex_exp

        signal[n] = idft_sum / N
    signal = np.round(np.real(signal)).astype(int)

    plt.figure(figsize=(8, 4))
    plt.plot(signal)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('IDFT: Reconstructed Signal')
    plt.show()
    return signal
 def dft_test(freq):
    co, signal = read_signal_from_file2()
    frequencies, amp, phase = dft(signal, float(freq))  # Convert freq to float
    print(amp)
    print(phase)
 def idft_test():
    co = read_signal_from_file2()
    signal = idft(co)
    print(signal)


# Create a style
 child = Toplevel(root)
 child.title("Airthmatic Window")
 child.geometry("300x200")

 ttk.Label(child, text="Enter the sampling frequency:",font=("Helvetica", 10)).grid(row=1, column=0)
 sampling_freq_entry = ttk.Entry(child)
 sampling_freq_entry.grid(row=1, column=1)


 ttk.Button(child, text="DFT", command=lambda: dft_test(sampling_freq_entry.get()),width=30,padding=2).grid(row=2,column=0,columnspan=2)

 ttk.Button(child, text="IDFT ", command=idft_test,width=30,padding=2).grid(row=3,column=0,columnspan=2)




# Start the GUI main loop
 root.mainloop()

def task5():
    # Function to load the signal, skipping the first three rows
    def load_signal():
        global signal_indices, signal_values
        file_path = filedialog.askopenfilename(title="Select a signal file")
        if file_path:
            try:
                data = np.loadtxt(file_path, skiprows=4)  # Assuming 2 columns of data (index and value)
                signal_indices = data[:, 0]  # First column as indices
                signal_values = data[:, 1]  # Second column as values
                print(f"Signal Loaded: {len(signal_values)} samples")
                messagebox.showinfo("Success", "Signal Loaded Successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Error loading or processing the file: {e}")


    # DerivativeSignal: Function to compute first and second derivatives and test
    def DerivativeSignal():
        if signal_values is None:
            messagebox.showerror("Error", "Please load a signal first!")
            return

        # Compute first and second derivatives
        first_derivative = signal_values[1:] - signal_values[:-1]
        second_derivative = signal_values[2:] - 2 * signal_values[1:-1] + signal_values[:-2]

        # Adjust expected output lengths
        expectedOutput_first = [1] * len(first_derivative)  # Match length with first_derivative
        expectedOutput_second = [0] * len(second_derivative)  # Match length with second_derivative

        if len(first_derivative) != len(expectedOutput_first) or len(second_derivative) != len(expectedOutput_second):
            print("Mismatch in length")
            return

        first = second = True

        # Testing first derivative
        for i in range(len(expectedOutput_first)):
            if abs(first_derivative[i] - expectedOutput_first[i]) > 0.01:
                first = False
                print("1st derivative wrong")
                return

        # Testing second derivative
        for i in range(len(expectedOutput_second)):
            if abs(second_derivative[i] - expectedOutput_second[i]) > 0.01:
                second = False
                print("2nd derivative wrong")
                return

        if first and second:
            print("Derivative Test case passed successfully")
        else:
            print("Derivative Test case failed")


    # Sharpening: Compute first and second derivatives
    def compute_sharpening():
        if signal_values is None:
            messagebox.showerror("Error", "Please load a signal first!")
            return
        first_derivative = signal_values[1:] - signal_values[:-1]
        second_derivative = signal_values[2:] - 2 * signal_values[1:-1] + signal_values[:-2]

        # Display results in a plot
        plt.figure(figsize=(8, 6))
        plt.subplot(2, 1, 1)
        plt.plot(first_derivative, label="First Derivative", color='blue')
        plt.title("First Derivative")
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(second_derivative, label="Second Derivative", color='orange')
        plt.title("Second Derivative")
        plt.legend()
        plt.tight_layout()
        plt.show()


    # Delaying or advancing a signal
    def shift_signal():
        if signal_values is None:
            messagebox.showerror("Error", "Please load a signal first!")
            return
        try:
            k = int(shift_k_entry.get())
            action = shift_option.get()
            shifted_signal = np.roll(signal_values, k if action == "Delay" else -k)

            # Display results in a plot
            plt.figure(figsize=(6, 4))
            plt.plot(signal_indices, signal_values, label="Original Signal", color='black', linestyle='dotted')
            plt.plot(signal_indices, shifted_signal, label=f"{action}ed Signal by {k} steps", color='green')
            plt.title(f"{action}ed Signal")
            plt.legend()
            plt.show()
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid integer for k!")


    # Folding the signal and displaying values (only fold values, not indices)
    def fold_signal():
        if signal_values is None:
            messagebox.showerror("Error", "Please load a signal first!")
            return
        length = len(signal_values)
        folded_values = [0] * length

        for i in range(length):
            folded_values[i] = signal_values[length - 1 - i]

        # Display results in a plot
        plt.figure(figsize=(6, 4))
        plt.plot(signal_indices, signal_values, label="Original Signal", color='black', linestyle='dotted')
        plt.plot(signal_indices, folded_values, label="Folded Signal", color='purple')
        plt.title("Folded Signal")
        plt.legend()
        plt.show()

        # Output values of folded signal (including indices)
        # output_text.delete(1.0, tk.END)
        output_text.insert(tk.END, "Folded Signal Values (Index n and x(n) values are folded):\n")
        for idx, value in zip(signal_indices, folded_values):
            output_text.insert(tk.END, f"Index: {int(idx)}, Folded Value: {value:.4f}\n")


    # Delaying or advancing the folded signal and displaying values
    def shift_folded_signal():
        if signal_values is None:
            messagebox.showerror("Error", "Please load a signal first!")
            return
        try:
            k = int(shift_folded_k_entry.get())
            action = shift_folded_option.get()
            folded_values = np.flip(signal_values)
            shifted_folded_signal = np.roll(folded_values, k if action == "Delay" else -k)

            # Display results in a plot
            plt.figure(figsize=(8, 6))
            plt.subplot(2, 1, 1)
            plt.plot(signal_indices, folded_values, label="Folded Signal", color='purple')
            plt.title("Folded Signal")
            plt.legend()
            plt.subplot(2, 1, 2)
            plt.plot(signal_indices, shifted_folded_signal, label=f"{action}ed Folded Signal by {k} steps", color='brown')
            plt.title(f"{action}ed Folded Signal")
            plt.legend()
            plt.tight_layout()
            plt.show()

            # Output values of shifted folded signal (including indices)
            output_text.insert(tk.END, f"{action}ed Folded Signal Values (k={k}):\n")
            for idx, value in zip(signal_indices, shifted_folded_signal):
                output_text.insert(tk.END, f"Index: {int(idx)}, Shifted Folded Value: {value:.4f}\n")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid integer for k!")


    # Function to calculate Mean Absolute Error (MAE)
    def mean_absolute_error(actual, expected):
        return np.mean(np.abs(actual - expected))


    # Function to calculate Root Mean Squared Error (RMSE)
    def root_mean_squared_error(actual, expected):
        return np.sqrt(np.mean((actual - expected) ** 2))


    # Updated Shift_Fold_Signal function to calculate error
    def Shift_Fold_Signal(file_name, Your_indices, Your_samples):
        expected_indices = []
        expected_samples = []
        try:
            # Read expected samples from the file
            with open(file_name, 'r') as f:
                f.readline()  # Skip header lines
                f.readline()
                f.readline()
                f.readline()
                line = f.readline()
                while line:
                    L = line.strip().split(' ')
                    if len(L) == 2:
                        expected_indices.append(int(L[0]))
                        expected_samples.append(float(L[1]))
                        line = f.readline()
                    else:
                        break
        except FileNotFoundError:
            messagebox.showerror("Error", f"File not found: {file_name}")
            return

        # Check if the length matches
        if len(expected_samples) != len(Your_samples) or len(expected_indices) != len(Your_indices):
            print("Shift_Fold_Signal Test case failed, your signal has different length from the expected one")
            return

        # Print loaded data for debugging
        print("Loaded Signal Samples (Your samples):", Your_samples[:5])  # Print the first 5 values
        print("Expected Samples from Test File:", expected_samples[:5])  # Print the first 5 expected values

        # Calculate the error between expected and actual signals
        mae = mean_absolute_error(np.array(Your_samples), np.array(expected_samples))
        rmse = root_mean_squared_error(np.array(Your_samples), np.array(expected_samples))

        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")

        # Check if the error is within an acceptable threshold
        if mae < 0.01 and rmse < 0.01:
            print("Shift_Fold_Signal Test case passed successfully")
        else:
            print(f"Shift_Fold_Signal Test case failed, MAE: {mae}, RMSE: {rmse}")


    # Test Shift Fold Signal Button
    def test_shift_fold_signal():
        # Prompt the user to select the test file
        file_path = filedialog.askopenfilename(title="Select Test File", filetypes=[("Text files", "*.txt")])

        if not file_path:
            messagebox.showerror("Error", "No test file selected!")
            return

        Shift_Fold_Signal(file_path, signal_indices, signal_values)


    # GUI Setup
    root = tk.Tk()
    root.title("Signal Processing")

    signal_indices, signal_values = None, None

    # Load Signal Button
    load_button = tk.Button(root, text="Load Signal", command=load_signal)
    load_button.pack(pady=10)

    # Derivative Button
    derivative_button = tk.Button(root, text="Compute Derivative", command=DerivativeSignal)
    derivative_button.pack(pady=10)

    # Sharpening Button
    sharpening_button = tk.Button(root, text="Sharpening", command=compute_sharpening)
    sharpening_button.pack(pady=10)

    # Fold Signal Button
    fold_button = tk.Button(root, text="Fold Signal", command=fold_signal)
    fold_button.pack(pady=10)

    # Shift Signal Button
    shift_k_label = tk.Label(root, text="Enter number of steps (k):")
    shift_k_label.pack()
    shift_k_entry = tk.Entry(root)
    shift_k_entry.pack()

    shift_option = tk.StringVar(value="Delay")
    shift_delay_button = tk.Radiobutton(root, text="Delay", variable=shift_option, value="Delay")
    shift_delay_button.pack()
    shift_advance_button = tk.Radiobutton(root, text="Advance", variable=shift_option, value="Advance")
    shift_advance_button.pack()

    shift_button = tk.Button(root, text="Shift Signal", command=shift_signal)
    shift_button.pack(pady=10)

    # Fold Shift Signal Button
    shift_folded_k_label = tk.Label(root, text="Enter number of steps (k) for Folded Signal:")
    shift_folded_k_label.pack()
    shift_folded_k_entry = tk.Entry(root)
    shift_folded_k_entry.pack()

    shift_folded_option = tk.StringVar(value="Delay")
    shift_folded_delay_button = tk.Radiobutton(root, text="Delay", variable=shift_folded_option, value="Delay")
    shift_folded_delay_button.pack()
    shift_folded_advance_button = tk.Radiobutton(root, text="Advance", variable=shift_folded_option, value="Advance")
    shift_folded_advance_button.pack()

    shift_folded_button = tk.Button(root, text="Shift Folded Signal", command=shift_folded_signal)
    shift_folded_button.pack(pady=10)

    # Test Button
    test_button = tk.Button(root, text="Test Shift Fold Signal", command=test_shift_fold_signal)
    test_button.pack(pady=10)

    # Output Text Box
    output_text = scrolledtext.ScrolledText(root, width=50, height=20)
    output_text.pack(pady=10)

    root.mainloop()
    
def task51():

    def read_signal_from_file_task5():
        filename = filedialog.askopenfilename(title="Select a Signal File")
        with open(filename, 'r') as file:
            lines = file.readlines()

        signal_type = int(lines[1].strip())
        samples = []

        if signal_type == 0:
            for line in lines[3:]:
                values = line.strip().split()
                if len(values) == 2:
                    index, amplitude = values
                    samples.append((float(index), float(amplitude)))
            return [sample[1] for sample in samples]  # Returning only amplitudes

        elif signal_type == 1:
            complex_numbers = []
            for line in lines[3:]:
                values = line.strip().split()
                if len(values) == 2:
                    amplitude, phase = values
                    amplitude, phase = float(amplitude), float(phase)
                    real_part = amplitude * np.cos(phase)
                    imaginary_part = amplitude * np.sin(phase)
                    complex_number = real_part + 1j * imaginary_part
                    complex_numbers.append(complex_number)
            return complex_numbers
        else:
            raise ValueError("Unsupported signal type")


    # Function to perform Discrete Cosine Transform (DCT)
    def dct(signal):
        N = len(signal)
        dct_result = np.zeros(N, dtype=np.float64)  # Ensure the result array has a specific data type

        for k in range(N):
            dct_sum = 0
            for n in range(N):
                # Check if the element in the signal array is numerical
                if (signal[n], (int, float, np.integer, np.floating)):
                    dct_sum += signal[n] * np.cos(np.pi / (4 * N) * (2 * n - 1) * (2 * k - 1))
                else:
                    raise TypeError("Input signal must contain numerical elements.")

            dct_result[k] = np.sqrt(2 / N) * dct_sum

        print(dct_result)

        return dct_result

    # Function to compute DCT, display the result, and save selected coefficients
    def compute_dct_and_save_coefficients(signal_samples):
        global dct_result  # Declare as global to use it later for comparison
        dct_result = dct(signal_samples)

        # Display DCT result
        plt.figure(figsize=(8, 4))
        plt.stem(dct_result, basefmt="", label='samples')
        plt.xlabel('DCT Coefficients')
        plt.ylabel('Amplitude')
        plt.title('DCT Result')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Ask the user to choose the number of coefficients to save
        num_coefficients = simpledialog.askinteger("Input", "Enter the number of coefficients to save:", parent=root)

        if num_coefficients is not None:
            # Save selected coefficients to a text file
            with open('selected_dct_coefficients.txt', 'w') as file:
                for i in range(num_coefficients):
                    file.write(f"{dct_result[i]:.4f}\n")
            messagebox.showinfo("Success", f"{num_coefficients} DCT coefficients saved to file.")

    # Function to compare the DCT result with the expected signal
    def compare_dct_with_expected():
        # Read the signal that was computed earlier and the expected signal
        file_name = filedialog.askopenfilename(title="Select the Expected Signal File")
        if file_name:
            SignalSamplesAreEqual(file_name, range(len(dct_result)), dct_result)

    # Function to compare signal samples
    def SignalSamplesAreEqual(file_name, indices, samples):
        expected_indices = []
        expected_samples = []
        with open(file_name, 'r') as f:
            line = f.readline()
            line = f.readline()
            line = f.readline()
            line = f.readline()
            while line:
                # process line
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

        if len(expected_samples) != len(samples):
            print("Test case failed, your signal has different length from the expected one")
            return
        for i in range(len(expected_samples)):
            if abs(samples[i] - expected_samples[i]) < 0.01:
                continue
            else:
                print("Test case failed, your signal has different values from the expected one")
                return
        print("Test case passed successfully")

    # end of task 5
    # ------------------------------------------
    # ---------------------------------------

    root = tk.Tk()
    root.title("DSP Tasks")
    root.configure(bg='#2C3E50')  # Dark background color

    # Create a style
    style = ttk.Style()
    style.configure("TButton",
                    font=('Arial', 16, 'bold'),
                    background='black',  # Button background color
                    foreground='white',  # Button text color (text inside the button)
                    padding=10)

    style.configure("TLabel",
                    font=('Helvetica', 20, 'bold'),
                    foreground='#ECF0F1')  # Label text color

    # ---------------------------------------------------------------------------------------
    # gui task 5

    task5_frame = tk.Frame(root, bg='#34495E')  # Change to tk.Frame for background color support
    task5_frame.grid(row=0, column=2, padx=5, pady=5)
    ttk.Label(task5_frame, text="Task 5", style="TLabel").grid(row=0, column=0, columnspan=2, pady=10)

    ttk.Button(task5_frame, text="Compute DCT", command=lambda: compute_dct_and_save_coefficients(read_signal_from_file_task5()), width=30, style="TButton").grid(row=2, column=0, columnspan=2, pady=10)

    # Compare button: Trigger the comparison after DCT coefficients are computed and saved
    ttk.Button(task5_frame, text="Compare with Expected", command=compare_dct_with_expected, width=30, style="TButton").grid(row=3, column=0, columnspan=2, pady=10)

# ----------------------------------------------------------------------------------------

# Start the GUI main loop
    root.mainloop()
def task6():
 
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
 def read_signal_from_file(file_path):
    indices, samples = [], []
    with open(file_path, 'r') as f:
        for _ in range(3):  # Skip the first three lines
            f.readline()
        for line in f:
            values = line.strip().split()
            if len(values) == 2:
                indices.append(int(values[0]))
                samples.append(float(values[1]))
    return indices, samples


 def open_signal_file():
    file_path = filedialog.askopenfilename(title="Select a Signal File")
    if file_path:
        indices, samples = read_signal_from_file(file_path)

        return file_path, indices, samples
    return None, None, None

 def moving_average():
    window_size = simpledialog.askinteger("Window Size", "Enter window size:")
    if window_size is not None:
            
     filepath , indeses , samples = open_signal_file()
     window_size = int(window_size)
     result = np.zeros(len(samples) - window_size + 1)
     for i in range(len(result)):
        result[i] = np.sum(samples[i:i+window_size]) / window_size

     file_path = filedialog.askopenfilename(title="Select a Signal File")
    SignalSamplesAreEqual(file_path,indeses,result)


    return result
 
 def DFT(signal):
    signal_length = len(signal)
    amplitude_spectrum = np.zeros(signal_length)
    phase_spectrum = np.zeros(signal_length)
    for k in range(signal_length):
        exponential = sum(signal * np.exp(-1j * 2 * np.pi * k * np.arange(signal_length) / signal_length))
        amplitude_spectrum[k] = np.sqrt(np.real(exponential) ** 2 + np.imag(exponential) ** 2)
        phase_spectrum[k] = np.arctan2(exponential.imag, exponential.real)
    return amplitude_spectrum, phase_spectrum
 def Idft(amplitude, phase):
    N = len(amplitude)
    t = np.arange(N)
    real_signal = np.zeros(N)
    imag_signal = np.zeros(N)
    for k in range(N):
        complex_exp = amplitude[k] * np.exp(1j * (2 * np.pi * k * t / N + phase[k]))
        real_signal += complex_exp.real
        imag_signal += complex_exp.imag
    complex_signal = real_signal + 1j * imag_signal
    complex_signal /= N
    return complex_signal.real
 def remove_DC_freq():

    filepath, indices, samples = open_signal_file()

    amplitude, phase = DFT(samples)

    amplitude[0] = 0

    dc_removed_signal = Idft(amplitude, phase)

    dc_removed_signal_rounded = np.round(dc_removed_signal, 3)

    file_path = filedialog.askopenfilename(title="Select a Signal File")
    SignalSamplesAreEqual(file_path, indices, dc_removed_signal_rounded)
    return dc_removed_signal_rounded
 def remove_DC_time():
    filepath, indices, samples = open_signal_file()

    total = 0
    for sample in samples:
        total += sample
    dc_component = total / len(samples) 

    dc_removed_signal = [sample - dc_component for sample in samples]

    dc_removed_signal_rounded = np.round(dc_removed_signal, 3)

    file_path = filedialog.askopenfilename(title="Select a Signal File")
    SignalSamplesAreEqual(file_path, indices, dc_removed_signal_rounded)
    return dc_removed_signal_rounded
 def ConvTest(Your_indices,Your_samples): 
    """
    Test inputs
    InputIndicesSignal1 =[-2, -1, 0, 1]
    InputSamplesSignal1 = [1, 2, 1, 1 ]
    
    InputIndicesSignal2=[0, 1, 2, 3, 4, 5 ]
    InputSamplesSignal2 = [ 1, -1, 0, 0, 1, 1 ]
    """
    
    expected_indices=[-2, -1, 0, 1, 2, 3, 4, 5, 6]
    expected_samples = [1, 1, -1, 0, 0, 3, 3, 2, 1 ]

    
    if (len(expected_samples)!=len(Your_samples)) and (len(expected_indices)!=len(Your_indices)):
        print("Conv Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_indices)):
        if(Your_indices[i]!=expected_indices[i]):
            print("Conv Test case failed, your signal have different indicies from the expected one") 
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Conv Test case failed, your signal have different values from the expected one") 
            return
    print("Conv Test case passed successfully")
 def convelution():
  filepath1 , indeses1 , samples1 = open_signal_file()
  filepath2 , indeses2 , samples2 = open_signal_file()
  newsamples = np.zeros(len(samples1) + len(samples2) - 1)
  for i in range(len(samples1)):
        for j in range(len(samples2)):
            newsamples[i+j] += samples1[i] * samples2[j]


  newindeses = []

  current_index = indeses1[0]

  while len(newindeses) < len(newsamples):
        newindeses.append(current_index)
        current_index += 1

  ConvTest(newindeses,newsamples)


  return newsamples


  ConvTest(Your_indices,Your_samples)
 def Compare_Signals(file_name,Your_indices,Your_samples):      
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
    print("Current Output Test file is: ")
    print(file_name)
    print("\n")
    if (len(expected_samples)!=len(Your_samples)) :
        print("Shift_Fold_Signal Test case failed, your signal have different length from the expected one")
        return
    # for i in range(len(Your_indices)):
    #     if(Your_indices[i]!=expected_indices[i]):
    #         print("Shift_Fold_Signal Test case failed, your signal have different indicies from the expected one") 
    #         return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Correlation Test case failed, your signal have different values from the expected one") 
            return
    print("Correlation Test case passed successfully") 
 def calc_correlation(signal1, signal2, normalized):
    sqr_signal1_sum = 0
    sqr_signal2_sum = 0

    correlation = []
    r = 0

    for i in range(len(signal1)):
        sqr_signal1_sum += signal1[i] ** 2
        sqr_signal2_sum += signal2[i] ** 2
    for j in range(len(signal1)):
        for n in range(len(signal1)):
            r += (1 / len(signal1)) * (signal1[n] * signal2[(n + j) % len(signal1)])

        if normalized:
            result = r / ((1 / len(signal1)) * (math.sqrt(sqr_signal1_sum * sqr_signal2_sum)))
            correlation.append(round(result, 8))
        else:
            correlation.append(round(r, 8))

        r = 0

    file_name=filedialog.askopenfilename(title="Select a Signal File")
    print(correlation)
    Compare_Signals(file_name,save_indeces,correlation)
    return correlation
 

 def do_corr():
    file_path, indices, samples1=open_signal_file()
    file_path, indices, samples2=open_signal_file()
    result = calc_correlation(samples1, samples2,True)
    print(result)
 root = tk.Tk()
 root.title("Task 6")

 frame = tk.Frame(root)
 frame.pack(pady=10)


 moving_avaerage = tk.Button(frame, text="Moving average", command=moving_average)
 moving_avaerage.pack(pady=5)

 compare_button = tk.Button(frame, text="remove DC Time domain ", command=remove_DC_time)
 compare_button.pack(pady=5)

 compare_button = tk.Button(frame, text="remove DC Freq domain", command=remove_DC_freq)
 compare_button.pack(pady=5)

 compare_button = tk.Button(frame, text="Convolve", command=convelution)
 compare_button.pack(pady=5)



 compare_levels_button = tk.Button(frame, text="Correlation", command=do_corr)
 compare_levels_button.pack(pady=5)

 result_label = tk.Label(frame, text="", justify=tk.LEFT)
 result_label.pack()

 root.mainloop()


root = tk.Tk()

sinbutton = tk.Button(root, command=task1, text="Signal Generator", fg='black', font=('Arial', 15))
sinbutton.grid(row=5, column=1, sticky=tk.W, padx=5, pady=10)

AIrthmaticbutton = tk.Button(root, command=task2, text="Airathmatic Operations", fg='black', font=('Arial', 15))
AIrthmaticbutton.grid(row=5, column=3, sticky=tk.W, padx=5, pady=10)

AIrthmaticbutton = tk.Button(root, command=task3, text="Quantization", fg='black', font=('Arial', 15))
AIrthmaticbutton.grid(row=5, column=5, sticky=tk.W, padx=5, pady=10)

AIrthmaticbutton = tk.Button(root, command=task4, text="Fourier transform", fg='black', font=('Arial', 15))
AIrthmaticbutton.grid(row=5, column=7, sticky=tk.W, padx=5, pady=10)

AIrthmaticbutton = tk.Button(root, command=task51, text="DCT", fg='black', font=('Arial', 15))
AIrthmaticbutton.grid(row=5, column=9, sticky=tk.W, padx=5, pady=10)

AIrthmaticbutton = tk.Button(root, command=task5, text="Time domain", fg='black', font=('Arial', 15))
AIrthmaticbutton.grid(row=5, column=11, sticky=tk.W, padx=5, pady=10)

AIrthmaticbutton = tk.Button(root, command=task6, text="Task 6", fg='black', font=('Arial', 15))
AIrthmaticbutton.grid(row=5, column=13, sticky=tk.W, padx=5, pady=10)

root.mainloop()

