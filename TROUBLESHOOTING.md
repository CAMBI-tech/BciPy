# How to Install BciPy on Mac M1 Chip

If you are using a computer with the Mac M1 chip, you will run into issues with installing some packages of BciPy including numpy and psychopy. 

The best workaround for this is to use the Rosetta Terminal. 
#

## To create the Rosetta Terminal: 
1. Duplicate the Apple terminal and rename one of the terminals "Rosetta Terminal" 
2. Right click on the terminal you have renamed and click "Get info"
From the "Get info" menu, check the box that says "Open using Rosetta" 

Launch the Rosetta Terminal and use only the Rosetta Terminal for all the following steps/any usage of BciPy. 
#

## To Install BciPy (within the Rosetta Terminal)
1. Git clone https://github.com/BciPy/BciPy.git
2. Run the following command in your terminal:
    
    > `arch -arm64 brew install python@3.9`
3. Create a virtual environment using python 3.9:
    > `python3.9 -m venv venv`
4. Open the virtual environment:
    > `source venv/bin/activate` 
5. Run the following command in your terminal (make sure that your pip is up to date if there are any issues at this step):
    >  `pip install -e .` 
6. Install XCode and enable command line tools:
    > `xcode-select --install`
7. To install the M1 compatible version of PsychoPy, run:
    > `pip install Psychopy==2020.2.10 --install-option="--no-deps"`
8. To install the latest version from PyPi, run: 
    > `pip install BciPy`
#
You can open and use the Rosetta terminal within different IDEs. You just have to make sure that you add it in the settings and then switch to it before you try to run BciPy scripts. The correct packages are only installed in the Rosetta Terminal, so BciPy will not work in zsh/bash, etc. 