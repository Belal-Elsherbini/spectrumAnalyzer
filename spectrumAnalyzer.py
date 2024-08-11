from rtlsdr import RtlSdr
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import time
import tkinter as tk
from scipy.signal import butter, lfilter

#configurations
FREQ_LOW = 93.1e6 #93.1 mhz
FREQ_HIGH = 103.5e6 #103.5 mhz
criticalFrequency = 30e3
filterChoice = "NONE"


#initialize SDR object
sdr = RtlSdr()
#Set up device configurations
sdr.sample_rate = 2.048e6
sdr.gain = 50
CHUNK_SIZE = 256*256


#set up Tkinter GUI
root = tk.Tk()
root.configure(background='lightblue')
root.geometry('1600x1000')
frame = tk.Frame(root)
label = tk.Label(text = "Center Frequency")
button = tk.Button(text="Quit",command=root.destroy)

#make slider and function to adust center frequency
def setCenterFrequency(freq):
    sdr.center_freq = freq
slider = tk.Scale(root, from_ = FREQ_LOW, to = FREQ_HIGH, orient = 'vertical', command = setCenterFrequency, length=475, resolution = 0.2e6)


#make slider and label to adust critical frequency (For filters)
fcLabel = tk.Label(text="Filter Critical Frequency")

def setCriticalFrequency(value):
    global criticalFrequency
    criticalFrequency = value

fcSlider = tk.Scale(root, from_ = 30e3, to = 400e3, orient='vertical', command = setCriticalFrequency, length = 475, resolution = 1e3)



#Set up initial plot configurations

fig, frequencyAX = plt.subplots()
#set parameters for frequency axis
frequencyAX.set_xlim(-sdr.sample_rate/4, sdr.sample_rate/4) #Maaximum frequency will ever be half the nyquist rate (sdr sample rate / 2)
frequencyAX.set_ylim(-25,75)
frequencyAX.set_title("IQ Data: Frequency Domain")
frequencyAX.set_xlabel("Frequency (Hz)")
frequencyAX.set_ylabel("Amplitude (dB)")
#set parameters for complex axis
freqLine, = frequencyAX.plot([], []) #set up an initial plot of no x or y coordinates, unpack the tuple by assigning the first and only line element to line variable
#ax.plot returns a tuple of 2D lines given a set of x and y coordinates. We only have one line so we unpack the tuple and take the first element
#embed matplotlib graph into tkinter gui
canvas = FigureCanvasTkAgg(fig, master = root)



#helper and application functions for several filter types
def butterLowpass(cutoff, fs, order = 5):
    nyquist = fs / 2
    normal_cutoff = float(cutoff)/nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a

def applyLowPass(data, cutoff, fs, order = 5):
    b, a = butterLowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butterHighPass(lowcut, fs, order = 5):
    nyquistFreq = fs/2
    normalizedCutoff = float(lowcut)/nyquistFreq
    b, a = butter(order, normalizedCutoff, btype = "highpass", analog=False)
    return b,a
def applyHighPass(data, lowCutoff, fs, order = 5):
    b, a = butterHighPass(lowCutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butterBandPass(lowcut, highcut, fs, order=5):
    nyquistFreq = fs/2
    lowNorm = float(lowcut)/nyquistFreq
    highNorm = float(highcut)/nyquistFreq
    b,a = butter(order, [lowNorm, highNorm], btype='band')
    return b,a

def applyBandPass(data, lowCut, highCut, fs, order=5):
    b, a = butterBandPass(lowCut, highCut, fs, order=order)
    y = lfilter(b, a, data)
    return y
#enables particular filters
def enableFilter(filterSelect):
    global filterChoice
    match filterSelect:
        case "LPF":
            filterChoice = "LPF"
        case "HPF":
            filterChoice = "HPF"
        case "BPF":
            filterChoice = "BPF"
        case _:
            filterChoice = "NONE"


#buttons to turn enable certain filter visualizations
lowPassButton = tk.Button(root, text="Apply Low Pass Filter", command = lambda: enableFilter("LPF")) #need to use a lambda function because it executes inline otherwise
highPassButton = tk.Button(root, text = "Apply High Pass Filter", command = lambda: enableFilter("HPF"))
bandPassButton = tk.Button(root, text = "Apply Band Pass Filter", command = lambda: enableFilter("BPF"))
noneButton = tk.Button(root, text = "Original Spectrum", command = lambda: enableFilter("NONE"))

#pack all the widgets, rearrange them
canvas.get_tk_widget().pack(side='left', padx=20)
label.place(x=670, y=225)
slider.pack(side='left')
button.pack(side="right", padx=300)
fcLabel.place(x=870, y=225)
fcSlider.place(x=900, y=270)
lowPassButton.place(x=1000, y=1080/3)
highPassButton.place(x=1000, y=1080*0.56)
bandPassButton.place(x=1000, y=1080*0.445)
noneButton.place(x=1300, y=1080*0.445)

#turn on real time updating plot
plt.ion()
try:

    while True:
        #get some samples
        samples = sdr.read_samples(CHUNK_SIZE)
        
        #User picks between these possible filter visualizations
        if filterChoice == "LPF":
            LPFSamples = applyLowPass(samples, criticalFrequency, sdr.sample_rate)
            LPFfftVals = 20*np.log10(np.abs(np.fft.fftshift(np.fft.fft(LPFSamples))))
            fftFreqs = np.fft.fftshift(np.fft.fftfreq(len(LPFSamples), 1/sdr.sample_rate))
            freqLine.set_xdata(fftFreqs)
            freqLine.set_ydata(LPFfftVals)
        elif filterChoice == "HPF":
            HPFfilteredSamples = applyHighPass(samples, criticalFrequency, sdr.sample_rate)
            HPFfftVals = 20*np.log10(np.abs(np.fft.fftshift(np.fft.fft(HPFfilteredSamples))))
            fftFreqs = np.fft.fftshift(np.fft.fftfreq(len(HPFfilteredSamples), 1/sdr.sample_rate))
            freqLine.set_xdata(fftFreqs)
            freqLine.set_ydata(HPFfftVals)
        elif filterChoice == "BPF":
            BPFfilteredSamples = applyBandPass(samples, float(criticalFrequency) - 20e3, criticalFrequency, sdr.sample_rate)
            BPFfftVals = 20*np.log10(np.abs(np.fft.fftshift(np.fft.fft(BPFfilteredSamples))))
            fftFreqs = np.fft.fftshift(np.fft.fftfreq(len(BPFfilteredSamples), 1/sdr.sample_rate))
            freqLine.set_xdata(fftFreqs)
            freqLine.set_ydata(BPFfftVals)
        elif filterChoice == "NONE":
            #convert iq data to frequency domain with decibel scale
            fftVals = np.fft.fftshift(np.fft.fft(samples))
            fftValsdB = 20*np.log10(np.abs(fftVals))
            #get frequency bins
            fftFreqs = np.fft.fftshift(np.fft.fftfreq(len(samples), 1/sdr.sample_rate))
            #apply Low pass filter if enabled
            freqLine.set_xdata(fftFreqs)
            freqLine.set_ydata(fftValsdB)

        root.update_idletasks()
        root.update()
        time.sleep(0.01)


except KeyboardInterrupt:
    pass
finally:
    sdr.close()
    plt.ioff()
