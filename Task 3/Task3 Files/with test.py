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
            bits = int(np.log2(levels))

        delta = (max_val - min_val) / levels


        ranges = [(min_val + i * delta, min_val + (i + 1) * delta) for i in range(levels)]
        midpoints = [(start + end) / 2 for start, end in ranges]

        quantized_values = np.zeros_like(amplitude)
        quantization_labels = []  # Store the encoded levels (labels)
        sampled_error = []  # Store the quantization errors

        for i, value in enumerate(amplitude):
            for j, (start, end) in enumerate(ranges):
                if start <= value < end:
                    quantized_values[i] = midpoints[j]
                    error = quantized_values[i] - amplitude[i]
                    sampled_error.append(error)  # Store error
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


def QuantizationTest1(file_name, Your_EncodedValues, Your_QuantizedValues):
    expectedEncodedValues = []
    expectedQuantizedValues = []

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
                V2 = str(L[0])
                V3 = float(L[1])
                expectedEncodedValues.append(V2)
                expectedQuantizedValues.append(V3)
                line = f.readline()
            else:
                break

    # Check lengths
    if (len(Your_EncodedValues) != len(expectedEncodedValues)) or (
            len(Your_QuantizedValues) != len(expectedQuantizedValues)):
        print("QuantizationTest1 Test case failed, your signal has different lengths from the expected one")
        print(f"Your Encoded Length: {len(Your_EncodedValues)}, Expected Length: {len(expectedEncodedValues)}")
        print(f"Your Quantized Length: {len(Your_QuantizedValues)}, Expected Length: {len(expectedQuantizedValues)}")
        return

    # Check encoded values
    for i in range(len(Your_EncodedValues)):
        if Your_EncodedValues[i] != expectedEncodedValues[i]:
            print(
                "QuantizationTest1 Test case failed, your EncodedValues have different EncodedValues from the expected one")
            print(f"Your Encoded Value: {Your_EncodedValues[i]}, Expected Encoded Value: {expectedEncodedValues[i]}")
            return

    # Check quantized values
    for i in range(len(expectedQuantizedValues)):
        if abs(Your_QuantizedValues[i] - expectedQuantizedValues[i]) < 0.01:
            continue
        else:
            print(
                "QuantizationTest1 Test case failed, your QuantizedValues have different values from the expected one")
            print(
                f"Your Quantized Value: {Your_QuantizedValues[i]}, Expected Quantized Value: {expectedQuantizedValues[i]}")
            return

    print("QuantizationTest1 Test case passed successfully")


def QuantizationTest2(file_name, Your_IntervalIndices, Your_EncodedValues, Your_QuantizedValues, Your_SampledError):
    expectedIntervalIndices = []
    expectedEncodedValues = []
    expectedQuantizedValues = []
    expectedSampledError = []

    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L = line.strip()
            if len(L.split(' ')) == 4:
                L = line.split(' ')
                V1 = int(L[0])
                V2 = str(L[1])
                V3 = float(L[2])
                V4 = float(L[3])
                expectedIntervalIndices.append(V1)
                expectedEncodedValues.append(V2)
                expectedQuantizedValues.append(V3)
                expectedSampledError.append(V4)
                line = f.readline()
            else:
                break

    if (len(Your_IntervalIndices) != len(expectedIntervalIndices) or
            len(Your_EncodedValues) != len(expectedEncodedValues) or
            len(Your_QuantizedValues) != len(expectedQuantizedValues) or
            len(Your_SampledError) != len(expectedSampledError)):
        print("QuantizationTest2 Test case failed, your signal has different lengths from the expected one")
        return

    for i in range(len(Your_IntervalIndices)):
        if Your_IntervalIndices[i] != expectedIntervalIndices[i]:
            print("QuantizationTest2 Test case failed, your signal has different indices from the expected one")
            return

    for i in range(len(Your_EncodedValues)):
        if Your_EncodedValues[i] != expectedEncodedValues[i]:
            print("QuantizationTest2 Test case failed, your EncodedValues have different EncodedValues from the expected one")
            return

    for i in range(len(expectedQuantizedValues)):
        if abs(Your_QuantizedValues[i] - expectedQuantizedValues[i]) < 0.01:
            continue
        else:
            print("QuantizationTest2 Test case failed, your QuantizedValues have different values from the expected one")
            return

    for i in range(len(expectedSampledError)):
        if abs(Your_SampledError[i] - expectedSampledError[i]) < 0.01:
            continue
        else:
            print("QuantizationTest2 Test case failed, your SampledError have different values from the expected one")
            return

    print("QuantizationTest2 Test case passed successfully")


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
        QuantizationTest1(test_file, quantization_labels, quantized_values)


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
        # Generate expected interval indices (1-based or adjust as needed)
        interval_indices = list(range(1, len(quantization_labels) + 1))  # Assuming 1-based index
        QuantizationTest2(test_file, interval_indices, quantization_labels, quantized_values, sampled_error)


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

compare_levels_button = tk.Button(frame, text="Compare Quantization Levels (Test 2)", command=compare_quantization_levels)
compare_levels_button.pack(pady=5)

result_label = tk.Label(frame, text="", justify=tk.LEFT)
result_label.pack(pady=5)

root.mainloop()


