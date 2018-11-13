# Language Model Setup
## Installing Docker 

Docker is required in order to run the language model. 
Downloads for Docker Community Edition can be found here: https://store.docker.com/search?type=edition&offering=community
If your operating system is not compatible with any version of Docker Community Edition, download and install Docker Toolbox.
Installation instructions for Docker Toolbox can be found here: [Windows](https://docs.docker.com/toolbox/toolbox_install_windows/)    [Mac](https://docs.docker.com/toolbox/toolbox_install_mac/)

After installing the correct version of Docker, run the Docker Quickstart Terminal. 

If Git is installed in a non-default location on your system, you may need to change the Docker Quickstart Terminal shortcut target to the bash executable located in your Git/bin folder. 

If you receive an error about VT-X or AMD-v being disabled when you run the Quickstart Terminal, you will need to enter your system's BIOS and enable hardware virtualization support. In order to do this:
1. Restart your computer.
1. Enter the computer's BIOS menu. This usually involves repeatedly pressing a key while the computer is starting up, such as F1, F2, Delete, or F10. The key needed differs depending on your system's manufacturer. 
1. Find the virtualization technology setting, and change it to 'Enabled'. This setting will likely be labeled 'Intel Virtualization Technology', or 'AMD-V'. The menu in which this setting is located differs depending on your system's manufacturer. 
1. Save the BIOS settings and exit the BIOS menu. 
1. Start your computer normally, and retry running the Quickstart Terminal. 

## Setting up the language model

**These steps only need to be taken once.** 

Once Docker has installed and loaded successfully, navigate in the regular terminal window to the location of the language model image file(s). 

In the regular terminal window, enter the command:

> docker load --input [image file name].tar 

Wait for the process to complete. When it is finished, in order to check that the image has been successfully loaded, enter: 

> docker images

Repeat these steps for each image file you wish to load. 

## Testing the language model

In order to test that the language model is working correctly, navigate to the root folder of BciPy and run:

> python bcipy/language_model/demo/language_model_demo.py 

If everything is configured correctly, this should print information about the language model file. 
