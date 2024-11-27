#!/bin/bash
###### Install Common Linux Requirements #######

sudo apt-get update
sudo apt-get install libgtk-3-dev
# python install (ubuntu comes with python3)
# sudo apt-get install python3
# sudo apt-get install python3-pip
sudo apt-get install freeglut3-dev
sudo apt-get install freetype*
sudo apt-get install portaudio*
sudo apt-get install libsndfile*
sudo apt-get install xvfb

# Dev packages may be required. Uncomment the correct version.
# Python 3.7
# sudo apt-get install python3.7-tk python3.7-dev 
# Python 3.8
# sudo apt-get install python3.8-tk python3.8-dev 
# Python 3.9
# sudo apt-get install python3.9-tk python3.9-dev

# run headless without a display connected. All commands must have xvfb-run preceeding them. 
# ex. $ xvfb-run pytest