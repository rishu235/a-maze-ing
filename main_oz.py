import numpy as np
import pyxdf
from scipy.signal import butter, lfilter, filtfilt
from scipy import signal
from sklearn.cross_decomposition import CCA

import pygame
import sys
pygame.init()
import numpy as np
#This script has routines to test streaming data
from pylsl import StreamInlet, resolve_stream
from utils_ssvep_bci import *

# first resolve an EEG stream on the lab network
print("looking for data stream...")
streams = resolve_stream('type', 'EEG')
# create a new inlet to read from the stream
inlet = StreamInlet(streams[0])
#init a matrix to append data
read =  np.zeros((1,8))
#----------------- EEG/data parameters--------------------------
fs = 300 #sampling freqeuncy 
time_window = 3 #window time - 3 seconds

#seconds
no_of_sec = 0

#data points aquired
data_pts_aq = 0

#flag vairable
flag = 0

#total_chunk_length
total_chunk_length = fs*time_window

#-------------------------------------------------------------
white = (255, 255, 255)
black = (0,0,0)
red = (255,0, 0)
gameDisplay = pygame.display.set_mode((800,600))
pygame.display.set_caption('Ficker_Test')
clock = pygame.time.Clock()
pygame.display.update()

gameExit = False

i = 1
c = 3

lowcut = 3
highcut = 30
b,a = preprocess_filter(lowcut, highcut, fs=fs, order=5)
gameDisplay.fill(white)
pygame.display.update()


while not gameExit:

#-----------pylsl--------------
# read sample from stream
    #print(pygame.time.get_ticks())
    sample, _ = inlet.pull_sample()
    #print(pygame.time.get_ticks())
# store the sample and stack in array read
    read = np.vstack([read, sample])
    #print(pygame.time.get_ticks(), np.shape(read))
    # increments to find total number of data points aquired
    data_pts_aq = data_pts_aq+1
    #variable indicates no_of_sec past
    no_of_sec = no_of_sec+1
    #if k == total chunk length then process data
    if data_pts_aq == total_chunk_length-1:
        print("Second-", (no_of_sec+1)/fs)
        print("Shape is", np.shape(read))
        #short_array = read[:,[1,3]]
        #print(short_array)
        y = filtfilt(b, a, read.T)
        cor1 = CCA_RAS(genRef(7.5, fs), y) #idx = 0 
        cor2 = CCA_RAS(genRef(12, fs), y) #idx = 1 
        cor3 = CCA_RAS(genRef(15,fs ), y) #idx = 2
        idx = np.argmax(([cor1, cor2, cor3]))
        print("idx", idx)
        read =  np.zeros((1,8))
        inlet = StreamInlet(streams[0])
        #print(pygame.time.get_ticks())
        data_pts_aq = 0
        flag = 1
        if round((no_of_sec+1)/fs) >= 10:
            gameExit = True
#---------------------------------
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            gameExit = True
        #if event.type == pygame.KEYDOWN:
    if flag == 1:
        pygame.draw.rect(gameDisplay, red, [100,100,20,20])
        flag = 0
    if i <= c: #if i is <= 3 do black
        #draw black
        pygame.draw.rect(gameDisplay, black,[400, 300, 100, 100])
    else:
        #draw white
        pygame.draw.rect(gameDisplay, white,[400, 300, 100, 100])
        gameDisplay.fill(white)
    if i == 6:
        i = 0
    i = i+1
    pygame.display.update()
    #clock.tick(60)
pygame.quit()

sys.exit()
