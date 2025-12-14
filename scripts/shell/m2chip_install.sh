"""Mac M2 install script. 

This script is intended to be run from the root of the BciPy repository. It has been tested on M2 and M3 chips. It may work for M1 chips, but
this has not been tested and may require use of Rosetta 2.

This script will install the necessary dependencies for the M2 chip on a Mac. It will also install
the psychopy version that is compatible with the M2 chip (no deps). This script assumes that you have homebrew
installed. If you do not, please install it before running this script.

This script will install the following dependencies manually or via homebrew:
    - hdf5
    - portaudio
    - pyaudio
    - pyo
    - psychopy

This script will also install the kenlm package with the max_order=12 option. This is necessary for
the language model and tests to run.
"""

xcode-select --install
brew install hdf5
brew install portaudio
# also make sure the path is correct/exported in .zprofile or .bash_profile
export HDF5_DIR=/opt/homebrew/opt/hdf5 
