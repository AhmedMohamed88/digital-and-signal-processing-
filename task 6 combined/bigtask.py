import numpy as np
import tkinter as tk
from tkinter import messagebox , simpledialog,scrolledtext
from tkinter import filedialog

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

def conv(x, y, x1, y1):
    conv_length = len(y) + len(y1) - 1
    y = np.pad(y, (0, conv_length - len(y)), 'constant')
    y1 = np.pad(y1, (0, conv_length - len(y1)), 'constant')
    complexreal, compleximag, complexreal1, compleximag1 = [], [], [], []
    for n in range(conv_length):
        real = 0.0
        imag = 0.0
        for k in range(conv_length):
            amplitude = y[k]
            phase = np.pi * 2 * n * k / conv_length
            real += amplitude * np.cos(phase)
            imag += (-amplitude * np.sin(phase))
        complexreal.append(real)
        compleximag.append(imag)

    for n in range(conv_length):
        real = 0.0
        imag = 0.0
        for k in range(conv_length):
            amplitude = y1[k]
            phase = np.pi * 2 * n * k / conv_length
            real += amplitude * np.cos(phase)
            imag += (-amplitude * np.sin(phase))
        complexreal1.append(real)
        compleximag1.append(imag)

    indices = []
    min_n = int(np.min(x) + np.min(x1))
    max_n = int(np.max(x) + np.max(x1))
    con = min_n
    for i in range(conv_length):
        indices.append(con)
        con += 1

    arr, arrr, res = [], [], []
    for i in range(conv_length):
        arr.append(complex(complexreal[i], compleximag[i]))
        arrr.append(complex(complexreal1[i], compleximag1[i]))

    for i in range(conv_length):
        res.append(arr[i] * arrr[i])

    xn = np.fft.ifft(res)
    xn_real = np.real(xn)
    return indices, xn_real
def ReadingFile(file_path):
    file = open(file_path, "r")
    signal_data = file.readlines()
    ignored_lines = signal_data[3:]
    x0, y0 = [], []
    for l in ignored_lines:
        row = l.split()
        x0.append(float(row[0]))
        y0.append(float(row[1]))
    return x0, y0

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
    if (len(expected_samples)!=len(Your_samples)) and (len(expected_indices)!=len(Your_indices)):
        print("Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_indices)):
        if(Your_indices[i]!=expected_indices[i]):
            print("Test case failed, your signal have different indicies from the expected one") 
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Test case failed, your signal have different values from the expected one") 
            return
    print("Test case passed successfully")

def SetSpecificationsFilters(FileName):
    global FilterType, FS, StopBandAttenuation, FC, F1, F2, TransitionBand
    FilterType = 0
    FS = 0
    StopBandAttenuation = 0
    FC = 0
    F1 = 0
    F2 = 0
    TransitionBand = 0

    with open(FileName, 'r') as file:
        lines1 = file.readlines()

    x = []
    y = ""

    ignored_lines = lines1[1:]
    for l in ignored_lines:
        row = l.split()
        x.append(float(row[2]))  # Assuming the 3rd column contains numeric data

    for l in lines1:
        row = l.split()
        y = row[2] + row[3]  # Assuming the 3rd and 4th columns contain filter type
        break  # Only read the first line for `y`

    # Determine filter type
    if y == "Lowpass":
        FilterType = 1
    elif y == "Highpass":
        FilterType = 2
    elif y == "Bandpass":
        FilterType = 3
    elif y == "Bandstop":
        FilterType = 4

    # Assign filter specifications based on the number of parameters
    if len(x) == 4:
        FS, StopBandAttenuation, FC, TransitionBand = x
    elif len(x) == 5:
        FS, StopBandAttenuation, F1, F2, TransitionBand = x

    return FilterType, FS, StopBandAttenuation, FC, F1, F2, TransitionBand


def CheckNOddOrEven(N):
    if (np.fmod(N, 2) == 1):
        return int(N)
    elif (np.fmod(N, 2) == 0 or (np.fmod(N, 2)) < 1):
        return int(N) + 1
    elif ((np.fmod(N, 2)) > 1):
        return int(N) + 2


def CalculateFC_new(Type, TransitionBand, FS, Fc=None, FC1=None, FC2=None):
    if (Type == 1):

        FC_Low_New = (Fc + (TransitionBand / 2)) / FS
        return FC_Low_New
    elif (Type == 2):

        FC_High_New = (Fc - (TransitionBand / 2)) / FS
        return FC_High_New
    elif (Type == 3):

        FC1_New = (FC1 - (TransitionBand / 2)) / FS
        FC2_New = (FC2 + (TransitionBand / 2)) / FS
        return FC1_New, FC2_New
    elif (Type == 4):

        FC1_New = (FC1 + (TransitionBand / 2)) / FS
        FC2_New = (FC2 - (TransitionBand / 2)) / FS
        return FC1_New, FC2_New


def No(StopBandAttenuation, TransitionBand, FS):
    if (StopBandAttenuation <= 21):
        N = 0.9 * FS / TransitionBand
    elif (StopBandAttenuation > 21 and StopBandAttenuation <= 44):
        N = 3.1 * FS / TransitionBand
    elif (StopBandAttenuation > 44 and StopBandAttenuation <= 53):
        N = 3.3 * FS / TransitionBand
    elif (StopBandAttenuation > 53 and StopBandAttenuation <= 74):
        N = 5.5 * FS / TransitionBand
    N_new = CheckNOddOrEven(N)
    return N_new


def Wn(StopBandAttenuation, N, n):
    if (StopBandAttenuation <= 21):
        return 1
    elif (StopBandAttenuation > 21 and StopBandAttenuation <= 44):
        eq = 0.5 + 0.5 * np.cos((2 * np.pi * n) / N)
        return eq
    elif (StopBandAttenuation > 44 and StopBandAttenuation <= 53):
        eq = 0.54 + 0.46 * np.cos((2 * np.pi * n) / N)
        return eq
    elif (StopBandAttenuation > 53 and StopBandAttenuation <= 74):
        eq = 0.42 + 0.5 * np.cos((2 * np.pi * n) / (N - 1)) + 0.08 * np.cos((4 * np.pi * n) / (N - 1))
        return eq


def CalculateWindowFunction(StopBandAttenuation, TransitionBand, FS, Index):
    N = No(StopBandAttenuation, TransitionBand, FS)
    n = - int(N / 2)
    wn = []
    for i in range(N):
        wn.append(Wn(StopBandAttenuation, N, n))
        Index.append(n)
        n = n + 1
    return wn


def Hn(FilterType, n, TransitionBand, FS, FC, F1, F2):
    if (FilterType == 1):
        if (n == 0):
            eq = 2 * CalculateFC_new(FilterType, TransitionBand, FS, Fc=FC)
            return eq
        else:
            f = CalculateFC_new(FilterType, TransitionBand, FS, Fc=FC)
            eq = 2 * f * np.sin(n * 2 * np.pi * f) / (n * 2 * np.pi * f)
            return eq

    elif (FilterType == 2):
        if (n == 0):
            eq = 1 - 2 * CalculateFC_new(FilterType, TransitionBand, FS, Fc=FC)
            return eq
        else:
            f = CalculateFC_new(FilterType, TransitionBand, FS, Fc=FC)
            eq = - 2 * f * np.sin(n * 2 * np.pi * f) / (n * 2 * np.pi * f)
            return eq
    elif (FilterType == 3):
        if (n == 0):
            f1, f2 = CalculateFC_new(FilterType, TransitionBand, FS, FC1=F1, FC2=F2)
            eq = 2 * (f2 - f1)
            return eq
        else:
            f1, f2 = CalculateFC_new(FilterType, TransitionBand, FS, FC1=F1, FC2=F2)
            eq = (2 * f2 * np.sin(n * 2 * np.pi * f2) / (n * 2 * np.pi * f2)) - (
                        2 * f1 * np.sin(n * 2 * np.pi * f1) / (n * 2 * np.pi * f1))
            return eq
    elif (FilterType == 4):
        if (n == 0):
            f1, f2 = CalculateFC_new(FilterType, TransitionBand, FS, FC1=F1, FC2=F2)
            eq = 1 - 2 * (f2 - f1)
            return eq
        else:
            f1, f2 = CalculateFC_new(FilterType, TransitionBand, FS, FC1=F1, FC2=F2)
            eq = (2 * f1 * np.sin(n * 2 * np.pi * f1) / (n * 2 * np.pi * f1)) - (
                        2 * f2 * np.sin(n * 2 * np.pi * f2) / (n * 2 * np.pi * f2))
            return eq


def CalculateFilter(FilterType, StopBandAttenuation, TransitionBand, FS, FC, F1, F2):
    N = No(StopBandAttenuation, TransitionBand, FS)
    n = -int(N / 2)
    hn = []
    for i in range(N):
        hn.append(Hn(FilterType, n, TransitionBand, FS, FC, F1, F2))
        n = n + 1
    return hn


def FIR(FilterType, StopBandAttenuation, TransitionBand, FS, FC, F1, F2):
    Index = []
    wn = CalculateWindowFunction(StopBandAttenuation, TransitionBand, FS, Index)
    hn = CalculateFilter(FilterType, StopBandAttenuation, TransitionBand, FS, FC, F1, F2)
    h = []
    N = No(StopBandAttenuation, TransitionBand, FS)
    for i in range(N):
        h.append(wn[i] * hn[i])
    return h, Index


def Testcases(testcase_entry):
    TestCase_no = int(testcase_entry.get())

    if TestCase_no == 1:
        # TestCase1
        file_path=filedialog.askopenfilename()
        SetSpecificationsFilters(file_path)
        LowPass, Index = FIR(FilterType, StopBandAttenuation, TransitionBand, FS, FC, F1, F2)
        LPFCoefficients=filedialog.askopenfilename()
        Compare_Signals(LPFCoefficients, Index, LowPass)
    elif TestCase_no == 2:

        # TestCase2
        file_path=filedialog.askopenfilename()
        SetSpecificationsFilters(file_path)
        LowPass0, Index0 = FIR(FilterType, StopBandAttenuation, TransitionBand, FS, FC, F1, F2)
        file_path2=filedialog.askopenfilename()
        x0, y0 = ReadingFile(file_path2)
        output = conv(x0, y0, Index0, LowPass0)
        outputx, outputy = output
        ecg_low_pass_filtered=filedialog.askopenfilename()
        Compare_Signals(ecg_low_pass_filtered, outputx, outputy)
    elif TestCase_no == 3:
        # TestCase3
        file_path=filedialog.askopenfilename()
        SetSpecificationsFilters(file_path)
        HighPass, Index = FIR(FilterType, StopBandAttenuation, TransitionBand, FS, FC, F1, F2)
        HPFCoefficientshigh=filedialog.askopenfilename()
        Compare_Signals(HPFCoefficientshigh, Index, HighPass)
    elif TestCase_no == 4:
        # TestCase4
        file_path=filedialog.askopenfilename()
        SetSpecificationsFilters(file_path)
        HighPass, Index0 = FIR(FilterType, StopBandAttenuation, TransitionBand, FS, FC, F1, F2)
        file_path2=filedialog.askopenfilename()
        x0, y0 = ReadingFile(file_path2)
        output = conv(x0, y0, Index0, HighPass)
        outputx, outputy = output
        ecg_high_pass_filtered=filedialog.askopenfilename()
        Compare_Signals(ecg_high_pass_filtered, outputx, outputy)
    elif TestCase_no == 5:
        # TestCase5
        file_path=filedialog.askopenfilename()
        SetSpecificationsFilters(file_path)
        BandPass, Index = FIR(FilterType, StopBandAttenuation, TransitionBand, FS, FC, F1, F2)
        BPFCoefficients=filedialog.askopenfilename()
        Compare_Signals(BPFCoefficients, Index, BandPass)
    elif TestCase_no == 6:
        # TestCase6
        file_path=filedialog.askopenfilename()
        SetSpecificationsFilters(file_path)
        BandPass, Index0 = FIR(FilterType, StopBandAttenuation, TransitionBand, FS, FC, F1, F2)
        file_path2=filedialog.askopenfilename()
        x0, y0 = ReadingFile(file_path2)
        output = conv(x0, y0, Index0, BandPass)
        outputx, outputy = output
        ecg_band_pass_filtered=filedialog.askopenfilename()
        Compare_Signals(ecg_band_pass_filtered, outputx, outputy)
    elif TestCase_no == 7:
        # TestCase7
        file_path=filedialog.askopenfilename()
        SetSpecificationsFilters(file_path)
        BandStop, Index = FIR(FilterType, StopBandAttenuation, TransitionBand, FS, FC, F1, F2)
        BSFCoefficients=filedialog.askopenfilename()
        Compare_Signals(BSFCoefficients, Index, BandStop)
    elif TestCase_no == 8:
        # TestCase8
        file_path=filedialog.askopenfilename()
        SetSpecificationsFilters(file_path)
        BandStop, Index0 = FIR(FilterType, StopBandAttenuation, TransitionBand, FS, FC, F1, F2)
        file_path2=filedialog.askopenfilename()
        x0, y0 = ReadingFile(file_path2)
        output = conv(x0, y0, Index0, BandStop)
        outputx, outputy = output
        ecg_band_stop_filtered=filedialog.askopenfilename()
        Compare_Signals(ecg_band_stop_filtered, outputx, outputy)


def Resampling(m_entry, l_entry):
    file_path = filedialog.askopenfilename()
    file = open(file_path, "r")
    signal_data = file.readlines()
    ignored_lines = signal_data[3:]
    x, y = [], []
    for l in ignored_lines:
        row = l.split()
        x.append(float(row[0]))
        y.append(float(row[1]))
    Specifications=filedialog.askopenfilename()
    FilterType, FS, StopBandAttenuation, FC, F1, F2, TransitionBand = SetSpecificationsFilters(
        Specifications)

    LowPass, Index = FIR(FilterType, StopBandAttenuation, TransitionBand, FS, FC, F1, F2)

    M = int(m_entry.get())
    L = int(l_entry.get())

    if M == 0 and L != 0:
        output, outputx, outputy = [], [], []
        y_upsampled, x_upsampled = [], []
        for i in range(len(y) - 1):
            y_upsampled.append(y[i])
            for j in range(L - 1):
                y_upsampled.append(0)
            for j in range(L):
                x_upsampled.append(x[i] + ((x[i + 1] - x[i]) / L) * j)

        y_upsampled.append(y[-1])
        x_upsampled.append(x[-1])

        output = conv(x_upsampled, y_upsampled, Index, LowPass)
        outputx, outputy = output
        Sampling_Up=filedialog.askopenfilename()
        Compare_Signals(Sampling_Up, outputx, outputy)
    elif M != 0 and L == 0:
        output, outputx, outputy = [], [], []
        output = conv(x, y, Index, LowPass)
        outputx, outputy = output

        y_downsampled, x_downsampled = [], []
        y_downsampled = outputy[::M]
        for i in range(len(y_downsampled)):
            x_downsampled.append(outputx[i])
        Sampling_Down=filedialog.askopenfilename()
        Compare_Signals(Sampling_Down, x_downsampled, y_downsampled)
    elif M != 0 and L != 0:
        output, outputx, outputy = [], [], []
        y_upsampled, x_upsampled = [], []
        for i in range(len(y) - 1):
            y_upsampled.append(y[i])
            for j in range(L - 1):
                y_upsampled.append(0)
            for j in range(L):
                x_upsampled.append(x[i] + ((x[i + 1] - x[i]) / L) * j)

        y_upsampled.append(y[-1])
        x_upsampled.append(x[-1])

        output = conv(x_upsampled, y_upsampled, Index, LowPass)
        outputx, outputy = output

        # Adjust the length of outputx and outputy to match the desired downsampling ratio
        target_length = len(outputx) // M * M  # Find the largest length divisible by M
        outputx = outputx[::target_length]
        outputy = outputy[::target_length]

        y_downsampled, x_downsampled = [], []
        for i in range(0, len(outputy), M):
            y_downsampled.append(outputy[i])
            x_downsampled.append(outputx[i])
        Sampling_Up_Down=filedialog.askopenfilename()
      #  print(x_downsampled,y_downsampled)
        print(Sampling_Up_Down)
        Compare_Signals(Sampling_Up_Down, x_downsampled, y_downsampled)
    else:
        print("Error!!!!!!!")


def open_window1():
    window1 = tk.Tk()
   
 

    button1 = tk.Button(window1, text="Filters and Reassmebling", height=5, width=20, command=Task1_window)
    button1.pack()

    window1.mainloop()


def Task1_window():
    window2 = tk.Tk()

    button2 = tk.Button(window2, text="Filtaring", height=5, width=20, command=Task1_A_window)
    button2.pack()

    button3 = tk.Button(window2, text="Re assembling", height=5, width=20, command=Task1_B_window)
    button3.pack()

    button4 = tk.Button(window2, text="Back", command=open_window1, height=5, width=20)
    button4.pack()

    window2.mainloop()


def Task1_A_window():
    window2 = tk.Tk()
    window2.title("Filter")
    window2.geometry('500x400')

    label2 = tk.Label(window2, text="filter window", height=5, width=20)
    label2.pack()

    testcase_label = tk.Label(window2, text="Testcase Number:", height=5, width=20)
    testcase_label.pack()
    testcase_entry = tk.Entry(window2)
    testcase_entry.pack()
    button2 = tk.Button(window2, text="Test Cases", command=lambda: Testcases(testcase_entry))
    button2.pack()

    button4 = tk.Button(window2, text="Back", command=open_window1)
    button4.pack()

    window2.mainloop()



def Task1_B_window():
    window2 = tk.Tk()


    label2 = tk.Label(window2, text="Resampling window")
    label2.pack()

    m_label = tk.Label(window2, text="M Value:")
    m_label.pack()
    m_entry = tk.Entry(window2)
    m_entry.pack()

    l_label = tk.Label(window2, text="L Value:")
    l_label.pack()
    l_entry = tk.Entry(window2)
    l_entry.pack()

    button2 = tk.Button(window2, text="Resample", command=lambda: Resampling(m_entry, l_entry))
    button2.pack()

    button4 = tk.Button(window2, text="Back", command=open_window1)
    button4.pack()

    window2.mainloop()


open_window1()    