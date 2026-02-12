


import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox

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

