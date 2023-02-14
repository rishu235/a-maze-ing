"""Example program to demonstrate how to send a multi-channel time series to
LSL."""

import random
import time
import numpy as np
from pylsl import StreamInfo, StreamOutlet

file_data = np.genfromtxt("testdata.txt", dtype=float)
print(file_data[:,11:19])
#print(np.shape(file_data))
#data = file_data[12,:]
#print(np.shape(data))
#print(len(data))

#data = File_data.astype(np.float32)
#print('Float value =', data)
#print(File_data)
#print(File_data[0])
#print(len(File_data[0]))

# first create a new stream info (here we set the name to BioSemi,
# the content-type to EEG, 8 channels, 100 Hz, and float-valued data) The
# last value would be the serial number of the device or some other more or
# less locally unique identifier for the stream as far as available (you
# could also omit it but interrupted connections wouldn't auto-recover).
info = StreamInfo('BioSemi', 'EEG', 8, 100, 'float32', 'myuid34234')

# next make an outlet
outlet = StreamOutlet(info)
k = 0;
print("now sending data...")
#print(np.shape(File_data[1][:]))
while True:
    # make a new random 8-channel sample; this is converted into a
    # pylsl.vectorf (the data type that is expected by push_sample)
    #mysample = [random.random(), random.random(), random.random(),
               # random.random(), random.random(), random.random(),
               # random.random(), random.random()]
    # now send it and wait for a bit
    #outlet.push_sample(mysample)
    outlet.push_sample(file_data[k,11:19])
    #print(file_data[k,:])
    k = k+1
    #np.shape(File_data[k][:])
    time.sleep(0.0033) 
    if(k == 16000):
        k = 0
#        break

