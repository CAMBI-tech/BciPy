#!/bin/bash
###### RUN BCI. #######
# To use, create a virtual env named `bci-env` in your Documents folder. 
# Clone the repo to that folder as well.
# `pip install -r requirements.txt` while that environment is activated. 
# If those assumptions are met, simply click on the run_bci.sh and the gui will 
# 	pop-up allowing you to execute the experiments.
# Change the paths or script name here is needed.


# Change directory to the path of virtual env and activate it
cd
cd Documents
source bci-env/Scripts/Activate

# Change directory to path of bci code
cd bci

# Set the python path
export PYTHONPATH=.

# Execute the gui code
python gui/BCInterface.py
