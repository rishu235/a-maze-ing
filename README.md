# A-maze-ing
An all-in-python BCI-integrated video game built for BCI GameJam 2023

This is an all-in-python BCI-integrated game. We have used the game engine pygame and require a stream of signal from SSVEP acquired through lab streaming layer (pylsl) which interfaces to any compatible EEG headset. We used an 8 channel headset for our demo (video attached). 

For testing without a live stream, run the following command in a new terminal to use a looped test data.
> python send_data.py

To play the game, run the main file [Strean should be on before running this]
> python main.py

# Theme
RAS devil has captured your pet and is moving from one portal to another. In this game, find the portal to reach the end of the level in a quest to find your pet.

# Gameplay instructions:

* Start the game.

* Concept The red arrow shows you the direction to the portal. Beware you may have to navigate around the walls to reach the portal.

# Features

* BCI integration

  The training-free implementation of SSVEP paradigm pre-processes the data using fifth-order butterworth filter. The canonical correlation analysis was used to classify the incoming signal as one of the three target frequencies.

* BCI mechanics

  The streming input is read with pylsl and broken into batches of three seconds. The signal processing creates a lag between the SSVEP event and execution of the related command. We make the batches discontinuous so that the lag doesn’t increase with each batch of the signal data.

* Accesibility

  Three stimulus presenting objects are presented as flickering squares. The SPO frequencies are set to turn left, go straight, or turn right. Alternatively, the game can also be played with keyboard arrows, without BCI.

* Media

  The background music changes between each level. Each turn has a swoosh sound effect. The texture for floor, walls and the sky is generated using static images. The colors randomly generated for each wall.

* Replayability

  The algorithm generates a new maze each level and every time you play. As you reach higher levels, 
   1)	the availability of the guiding arrow progressively reduces to none,
   2)	the maze becomes larger, and
   3)	the maze has more and more walls.
