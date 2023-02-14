"""
This code contains utility routines for BCI SSVEP game - Including - xdf_reading, channels-columns, preprocessing filter etc.
This code is used to convert the samples-channels data from xdf_reading code and splits into
columns. It will then divide each column into chunks of 512 data.
The divided column will be sent to the routine "preprocess
"""

import numpy as np
import pyxdf
from scipy.signal import butter, lfilter, filtfilt
from scipy import signal
from sklearn.cross_decomposition import CCA

# Function 1 - read xdf file and give samples to channels
def read_xdf(file_name = "AB_SSVEP_12Hz.xdf"):
    """
    This function reads an xdf file and returns channels data
    The data will have channels as columns
    Input: File name - currently coded to an xdf file
    Output: Would be a np array containing channels as columns and sampling frequency (per second)
    """

    # Read the data
    data, header = pyxdf.load_xdf('AB_SSVEP_12Hz.xdf')
    data_stream = 0 #Default- choose the first data stream column
    count = 0
    # Find the correct stream index by looking at the shape of time_series
    for stream in data:
        if(count == 0):
            data_stream = 0
            data_size = np.shape(stream['time_series'])[0]
        if(count > 0):
            #print(stream.keys())
            #print(np.shape(stream['time_series'])[0])
            data_size_new = np.shape(stream['time_series'])[0]
            #print("checking data", data_size_new, data_size)
            if(data_size_new > data_size):
                data_stream = count
                data_size = data_size_new
 #           print("checking data", data_size_new, data_size)
        count+=1
    #print(data_stream)

    #print("Per second sampling rate is about " + str( 1/(data[data_stream]['time_stamps'][5000] - data[data_stream]['time_stamps'][4990]) ))
    #Sampling frequency
    percent_cutoff = 0.1
    #Finding sampling frequency based on last 10 % data
    fs_start = data_size - int(percent_cutoff * data_size)
    fs_end = data_size - int(2*percent_cutoff * data_size)
    fs_span = fs_start-fs_end

    #print("Per second sampling rate is about " + str( 1/(data[data_stream]['time_stamps'][fs_start] - data[data_stream]['time_stamps'][fs_end]) ))
    fs = int( ( 1/(data[data_stream]['time_stamps'][fs_start] - data[data_stream]['time_stamps'][fs_end]) )*fs_span) + 1
    #print(fs, fs_span, fs_start, fs_end)

    # Actual data
    dataout = data[data_stream]['time_series']
#    print("Data has " + str(np.shape(data[data_stream]['time_series'])) + " channels (columns)." )
    return (dataout, fs)

##########################################################################
#Function 2 - column data to chunks of data

def columns_to_chunks(data, fs = 256):
    """
    This function takes the array created from function 1 (read_xdf) and sampling frequency
    and feeds into preprocess filter ...
    Input: np array of column data and sampling frequency
    """
    print(np.shape(data), fs)
    i = 0
    #Filter constants - currently hard coded, could convert as input to function if needed
    lowcut = 3
    highcut = 30
    span = 3*fs
    count_freq1 = 0
    count_freq2 = 0
    count_freq3 = 0
    while(i<np.shape(data)[0]-span):
    #while(i< (4*fs)):
        chunk_array = data[i:i+span]
        #print(np.shape(chunk_array))
        b,a = preprocess_filter(lowcut, highcut, fs=fs, order=5)
        y = filtfilt(b, a, chunk_array.T)
        #print(np.shape(y))
        #print("checking CCA")
        cor1 = CCA_RAS(genRef(9, fs), y) #idx = 0 
        cor2 = CCA_RAS(genRef(12, fs), y) #idx = 1 
        cor3 = CCA_RAS(genRef(15,fs ), y) #idx = 2
        idx = np.argmax(([cor1, cor2, cor3]))
        if(idx==0):
            count_freq1+=1
        if(idx==1):
            count_freq2+=1
        if(idx==2):
            count_freq3+=1
        i+=span
    total = count_freq1+count_freq2+count_freq3
    print("I am feeling lucky", count_freq1/float(total), count_freq2/float(total), count_freq3/float(total))

##########################################################################
#Function 3- preprocess filter

def preprocess_filter(lowcut, highcut, fs, order=5):
    """

    """
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

##########################################################################

#Function 4 - Generate sine waves and their harmonics at targeted frequencies

def genRef(f1=12, fs=256):
    """
    This function generates sine waves and returns their harmonics from input frequency
    Input: Frequency in hz and sampling frequency
    Output: Sine curve harmonics 
    """
    t = np.arange(0,3, 1/fs)
    ref1 = np.sin(2*np.pi*f1*t)
    ref2 = np.sin(2*np.pi*2*f1*t)
    ref3 = np.sin(2*np.pi*3*f1*t)
    ref = [ref1, ref2, ref3]
    return ref

##########################################################################

#Function 5 - CCA
def CCA_RAS(ref, processed_chunk_data):
    """
    Performs CCA on chunk data and returns correlation coefficients
    Input: Reference data 0 from function 4- genref, and preprocessed chunk data
    Output: Correlation coefficient
    """
    ncomp =1 #CCA tuning parameter- can be changed if needed
    #print("shape",np.shape(processed_chunk_data), np.shape(ref))
    cca = CCA(n_components=ncomp)
    #print("check in CCA")
    #print(np.shape(processed_chunk_data.T), np.shape(ref))
    #cca.fit(processed_chunk_data, np.transpose(ref))
    cca.fit(processed_chunk_data.T, np.transpose(ref))
    #np.shape(data)
    #X_c, Y_c = cca.transform(data, np.transpose(ref))
    X_c, Y_c = cca.transform(processed_chunk_data.T, np.transpose(ref))
    corrs = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(ncomp)] 
    return corrs