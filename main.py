# Authors: Rishabh Shukla, Balarama Sridhar Dwadasi, Anupam Bisht
# The game requires a stream of signal
# Run the following command in a new terminal if no signal exists
# python send_data.py
# Then run this file
# python main.py

# Import the libraries
import pygame as pg
import numpy as np
from numba import njit
import math
import pyxdf
from scipy.signal import butter, lfilter, filtfilt
from scipy import signal
from sklearn.cross_decomposition import CCA
import sys
from pylsl import StreamInlet, resolve_stream
from utils_ssvep_bci import *

def main(again, level):

    #Initialize pygame
    pg.init()
    pg.font.init()
    clock = pg.time.Clock()

    # Initialize data streaming
    # first resolve an EEG stream on the lab network
    print("looking for data stream...")
    print()
    streams = resolve_stream('type', 'EEG')
    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])
    #init a matrix to append data
    read =  np.zeros((1,8))
    fs = 256 #sampling freqeuncy
    time_window = 3 #window time - 3 seconds
    #seconds
    no_of_sec = 0
    #data points aquired
    data_pts_aq = 0
    #total_chunk_length
    total_chunk_length = fs*time_window


    # Butterworth filter
    lowcut = 3
    highcut = 30
    b,a = preprocess_filter(lowcut, highcut, fs=fs, order=5)

    #Load music and sound effects
    if level%2 ==1:
        pg.mixer.music.load("audio/Level_odd.mp3")
        pg.mixer.music.set_volume(1.0)
    else:
        # This music is louder, so loaded with reduced sound
        pg.mixer.music.load("audio/Level_even.mp3")
        pg.mixer.music.set_volume(0.6)
    turn_sound = pg.mixer.Sound("audio/Turn.mp3")
    # Set volume of the sound effect
    turn_sound.set_volume(0.1)
    # Play music
    pg.mixer.music.play()
    
    #Set the window size, screen and resolution
    width = 1250
    height = 600
    screen = pg.display.set_mode((width, height))
    hres = 130
    halfvres = 100

    # Keep a track of the number of frame
    i_frame = 1

    # 60 degrees field of view
    mod = hres/60.0

    # Generate a maze
    size, posx, posy, rot, maph, mapc, exitx, exity = gen_map(screen, level)

    # Load the sky and map the horizontal axis to 360 degress
    sky = pg.image.load('images/night_sky.jpg')
    sky = pg.surfarray.array3d(pg.transform.scale(sky, (360, halfvres*2)))
    # Load the floow and wall texture
    floor = pg.surfarray.array3d(pg.image.load('images/wood_floor.jpg'))
    wall = pg.surfarray.array3d(pg.image.load('images/brick_wall.jpg'))

    # Initialize the frame
    frame = np.random.uniform(0, 255, (hres, halfvres*2, 3))

    # Set the time reference
    time_previous = pg.time.get_ticks()/1000

    # Run the while loop until user quits
    running = True
    #print(exitx, exity)
    idx = None

    # The while loop runs once per frame
    while running:
        # Limits the frame rate to 60 Hz
        clock.tick(60)

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
            #print("Second-", (no_of_sec+1)/fs)
            #print("Shape is", np.shape(read))
            #short_array = read[:,[1,3]]
            #print(short_array)
            y = filtfilt(b, a, read.T)
            cor1 = CCA_RAS(genRef(7.5, fs), y) #idx = 0 
            cor2 = CCA_RAS(genRef(10, fs), y) #idx = 1 
            cor3 = CCA_RAS(genRef(12,fs ), y) #idx = 2
            idx = np.argmax(([cor1, cor2, cor3]))
            print("idx", idx)
            read =  np.zeros((1,8))
            inlet = StreamInlet(streams[0])
            #print(pygame.time.get_ticks())
            data_pts_aq = 0


        # Update frame every frame
        frame = new_frame(posx, posy, rot, frame, sky, floor, hres, halfvres, mod, maph, size, wall, mapc, exitx, exity)

                # # Chess board pattern
                # if int(x)%2 == int(y)%2:
                #     frame[i][halfvres*2-j-1] = [0, 0, 0]
                # else:
                #     frame[i][halfvres*2-j-1] = [255, 255, 255]
        
        # Get frame rate
        fps = int(clock.get_fps())
        # Set the window caption
        pg.display.set_caption("Level " + str(level) + " Maze FPS: " + str(fps))  

        # Remake surface every frame
        surf = pg.surfarray.make_surface(frame)
        surf = pg.transform.scale(surf, (width, height))
        # Redraw the surface every frame
        screen.blit(surf, (0,0))

        # Set parameters for the guiding arrow
        arrowx=width/2
        arrowy=height*0.9
        theta = math.atan((exity-posy)/(exitx-posx)) - rot
        theta = theta - math.pi/2
        max_level = 5   # arrows not available after max_level

        # Make the arrow disappear depending on the level
        if (pg.time.get_ticks()/2000%max_level >= level-1):
            arrow(screen, "red", "red", (arrowx, arrowy), (arrowx+20*math.cos(theta), arrowy+20*math.sin(theta)), 10)
        if fps>0:
            i_frame = add_flickers(screen, frame, i_frame, hres, 2*halfvres, width, height, fps, 0.5)

        # Update the display
        pg.display.update()

        # The user reaches the maze exit
        if abs(posx-exitx) <= 0.6 and abs(posy-exity) <= 0.6:
        #if int(posx) == exitx and int(posy) == exity:
            #print(posx, posy, rot)
            pg.mixer.fadeout(100)
            level_time = int(pg.time.get_ticks()/1000 - time_previous)
            # Display the start menu
            again = menu_screen(screen, level, level_time, again)
            running = False
            return again

        #Stream input executed
        if idx == 0:
            key = "left"
        if idx == 1:
            key = "up"
        if idx == 2:
            key = "right"
        # Get user movement
        if idx != None:
            posx, posy, rot = movement(posx, posy, rot, key, 1, maph, turn_sound)
            idx = None

        # The user asked to quit the game
        for event in pg.event.get():
            if event.type==pg.QUIT:
                again = False
                running = False
                return again
            key = None

            if event.type == pg.KEYDOWN:
                if event.key==pg.K_ESCAPE:
                    again = False
                    running = False
                    return again
                elif event.key==pg.K_UP:
                    key = "up"
                elif event.key==pg.K_DOWN:
                    key = "down"
                elif event.key==pg.K_LEFT:
                    key = "left"
                elif event.key==pg.K_RIGHT:
                    key = "right"
                posx, posy, rot = movement(posx, posy, rot, key, 1, maph, turn_sound)
                #print(posx, posy, rot)
        #print(pg.time.get_ticks())

def menu_screen(screen, level, level_time, again):
    # Play the sound for level complete
    pg.mixer.music.load("audio/Complete.mp3")
    pg.mixer.music.play()

    # Set the font for menu text
    my_font = pg.font.SysFont('Comic Sans MS', 30)
    # Draw the text background rectangle
    pg.draw.rect(screen, "white", [0, 0, 500, 200])
    # Set the text
    enter_text = my_font.render("Press Enter key to play again", False, (0, 0, 0))
    escape_text = my_font.render("Press Escape to quit", False, (0, 0, 0))
    time_text = my_font.render("Level " + str(level) + " completed in " + str(level_time) + " s", 
                                False, (0, 0, 0))

    # Draw the text on the screen
    screen.blit(enter_text, (0,0))
    screen.blit(escape_text, (0,50))
    screen.blit(time_text, (0,100))

    # Update display
    pg.display.update()

    # Read if the user wants to continue or quit
    # While loop freezes the frame until one of the expected inputs is received
    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT or event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                again = False
                return again
            if event.type == pg.KEYDOWN and event.key == pg.K_RETURN :
                pg.mixer.fadeout(100)
                return again


def add_flickers(screen, frame, i_frame, hres, vres, width, height, fps, dark):

    size = width/10
    # For a f Hz flash, fps/f is the number of frames in a cycles
    # Alternate between a white square a dark square

    # Left button
    if i_frame%(fps/7.5) < (fps/15): 
        pg.draw.rect(screen, dark*frame[int(hres/7)][int(vres/2)],[width/10, height/2, size, size])
    else:
        pg.draw.rect(screen, "white", [width/10, height/2, size, size])

    # Up button
    if i_frame%(fps/10) < (fps/20): 
        pg.draw.rect(screen, dark*frame[int(hres/2)][int(5*vres/11)],[3*width/7, 5*height/11, size, size])
    else:
        pg.draw.rect(screen, "white", [3*width/7, 5*height/11, size, size])
    
    # Right button
    if i_frame%(fps/12) < (fps/24): 
        pg.draw.rect(screen, dark*frame[int(6*hres/7)][int(vres/2)],[7*width/9, height/2, size, size])
    else:
        pg.draw.rect(screen, "white", [7*width/9, height/2, size, size])

    # Reset the frame counter when it reaches fps
    if i_frame >= fps: # > is required because frame may reduce and the counter goes over fps (not sure)
        i_frame = 0
    i_frame = i_frame+1
    return i_frame

def movement(posx, posy, rot, key, et, maph, turn_sound):
    x, y = posx, posy
    if key == "left":
        rot = rot - math.pi/4 #0.001*et
        pg.mixer.Sound.play(turn_sound)
        #print("Turning left")
    if key == "right":
        rot = rot + math.pi/4 #0.001*et
        pg.mixer.Sound.play(turn_sound)
        #print("Turning right")
    if key == "up":
        #x, y = x + np.cos(rot)*0.002*et, y + np.sin(rot)*0.002*et
        dx = np.cos(rot)/2
        dy = np.sin(rot)/2
        x, y = x + dx, y + dy
        #print("Moving forward")

        # if not(maph[int(x-0.2)][int(y)] or maph[int(x+0.2)][int(y)] or
        #    maph[int(x)][int(y-0.2)] or maph[int(x)][int(y+0.2)]):
        #    posx, posy = x, y
        # elif maph[int(posx+0.2)][int(y)]:
        #     posx = int(x)


    if key == "down":
        #x, y = x - np.cos(rot)*0.002*et, y - np.sin(rot)*0.002*et
        x, y = x - np.cos(rot), y - np.sin(rot)

    # Update the location only if there is no wall in that direction
    if not(maph[int(x-0.2)][int(y)] or maph[int(x+0.2)][int(y)] or
           maph[int(x)][int(y-0.2)] or maph[int(x)][int(y+0.2)]):
        posx, posy = x, y
        
    elif not(maph[int(posx-0.2)][int(y)] or maph[int(posx+0.2)][int(y)] or
             maph[int(posx)][int(y-0.2)] or maph[int(posx)][int(y+0.2)]):
        posy = y
        
    elif not(maph[int(x-0.2)][int(posy)] or maph[int(x+0.2)][int(posy)] or
             maph[int(x)][int(posy-0.2)] or maph[int(x)][int(posy+0.2)]):
        posx = x
    
    # Comment the previous block and uncomment the next line to make walls permeable
    posx, posy = x, y
    return posx, posy, rot

def gen_map(screen, level):

    # Set the size
    size = 5 + level

    # Set the color range of the walls
    mapc = np.random.uniform(0.1, 0.9, (size,size,3))

    # 1 in samples means wall, 0 mean no wall
    # Chances of sampling a 1 increase with each level
    samples = np.append(np.zeros((6, 1), dtype=int), level*[0, 0, 0, 1, 1])
    maph = np.random.choice(samples, (size,size))
    # Make walls all around the map
    maph[0,:], maph[size-1,:], maph[:,0], maph[:,size-1] = (1,1,1,1)

    posx, posy = 1.5, np.random.randint(1, size-1)+.5

    x, y = int(posx), int(posy)
    maph[x][y] = 0
    count = 0
    while True:
        testx, testy = (x, y)
        if np.random.uniform() > 0.5:
            testx = testx + np.random.choice([-1, 1])
        else:
            testy = testy + np.random.choice([-1, 1])
        if testx > 0 and testx < size -1 and testy > 0 and testy < size -1:
            if maph[testx][testy] == 0 or count > 5:
                count = 0
                x, y = (testx, testy)
                maph[x][y] = 0
                if x == size-2:
                    exitx, exity = (x, y)
                    break
            else:
                count = count+1

        # Rotate the view to not face a wall in the first frame
        if maph[int(posx)+1][int(posy)]==1:
            if maph[int(posx)][int(posy)-1]==1:
                # print("Wall on left and front")
                rot = np.pi/4 + 0.01
            elif maph[int(posx)][int(posy)+1]==1:
                # print("Wall on right and front")
                rot = -1*np.pi/4 - 0.01
            else:
                # print("Wall only in front")
                rot = np.pi/4 + 0.01
        else:
            # print("Heading along the x-axis")
            rot = 0.01
    return size, posx, posy, rot, maph, mapc, exitx, exity

# Draw a line and a triangle to make an arrow
def arrow(screen, lcolor, tricolor, start, end, trirad, thickness=4):
    pg.draw.line(screen, lcolor, start, end, thickness)
    rotation = (math.atan2(start[1] - end[1], end[0] - start[0])) + math.pi/2
    pg.draw.polygon(screen, tricolor, ((end[0] + trirad * math.sin(rotation),
                                        end[1] + trirad * math.cos(rotation)),
                                       (end[0] + trirad * math.sin(rotation - 120*math.pi/180),
                                        end[1] + trirad * math.cos(rotation - 120*math.pi/180)),
                                       (end[0] + trirad * math.sin(rotation + 120*math.pi/180),
                                        end[1] + trirad * math.cos(rotation + 120*math.pi/180))))

@njit(cache=True)
def new_frame(posx, posy, rot, frame, sky, floor, hres, halfvres, mod, maph, size, wall, mapc, exitx, exity):
    for i in range(hres):
        rot_i = rot + np.deg2rad(i/mod - 30)
        sin, cos, cos2 = np.sin(rot_i), np.cos(rot_i), np.cos(np.deg2rad(i/mod - 30))
        frame[i][:] = sky[int(np.rad2deg(rot_i)%359)][:]

        x, y = posx, posy
        while maph[int(x)%(size-1)][int(y)%(size-1)] == 0:
            x, y = x +0.01*cos, y +0.01*sin

        n = abs((x - posx)/cos)    
        h = int(halfvres/(n*cos2 + 0.001))

        xx = int(x*3%1*99)        
        if x%1 < 0.02 or x%1 > 0.98:
            xx = int(y*3%1*99)
        yy = np.linspace(0, 3, h*2)*99%99

        shade = 0.3 + 0.7*(h/halfvres)
        if shade > 1:
            shade = 1
            
        ash = 0 
        if maph[int(x-0.33)%(size-1)][int(y-0.33)%(size-1)]:
            ash = 1
            
        if maph[int(x-0.01)%(size-1)][int(y-0.01)%(size-1)]:
            shade, ash = shade*0.5, 0
            
        c = shade*mapc[int(x)%(size-1)][int(y)%(size-1)]
        for k in range(h*2):
            if halfvres - h +k >= 0 and halfvres - h +k < 2*halfvres:
                if ash and 1-k/(2*h) < 1-xx/99:
                    c, ash = 0.5*c, 0
                frame[i][halfvres - h +k] = c*wall[xx][int(yy[k])]
                if halfvres+3*h-k < halfvres*2:
                    frame[i][halfvres+3*h-k] = c*wall[xx][int(yy[k])]
                
        for j in range(halfvres -h):
            n = (halfvres/(halfvres-j))/cos2
            x, y = posx + cos*n, posy + sin*n
            xx, yy = int(x*2%1*99), int(y*2%1*99)

            shade = 0.2 + 0.8*(1-j/halfvres)
            if maph[int(x-0.33)%(size-1)][int(y-0.33)%(size-1)]:
                shade = shade*0.5
            elif ((maph[int(x-0.33)%(size-1)][int(y)%(size-1)] and y%1>x%1)  or
                  (maph[int(x)%(size-1)][int(y-0.33)%(size-1)] and x%1>y%1)):
                shade = shade*0.5

            frame[i][halfvres*2-j-1] = shade*(floor[xx][yy]+0.1*frame[i][halfvres*2-j-1])/2
            #if int(x) == exitx and int(y) == exity and (x%1-0.5)**2 + (y%1-0.5)**2 < 0.2:
            if (x-exitx)**2 + (y-exity)**2 <= 0.2:
                ee = j/(20*halfvres)
                frame[i][j:2*halfvres-j] = (ee*255*np.ones(3)+frame[i][j:2*halfvres-j])/(1+ee)
    return frame


if __name__ == '__main__':

    # The game keeps running if again is True
    again = True
    # Number of current level
    level = 1

    while again:
        print("Level is " + str(level))
        print()
        again = main(again, level)
        level = level + 1

    # If again becomes False, stop the music and quit
    pg.mixer.music.stop()
    pg.quit()
