import sys, os


def force_pypath():

    # Get current files path
    temp = os.path.dirname(os.path.abspath(__file__))

    # Find /bci in temp and change to different os notations
    BCI_PATH = temp[0:temp.rfind('bci')+3] # Notation for windows
    BCI_PATH2 = BCI_PATH.replace('\\','/') # Notation used on unix and mac

    # Add to path if not already included.
    if not BCI_PATH in sys.path:
        sys.path.append(BCI_PATH)

    if not BCI_PATH2 in sys.path:
        sys.path.append(BCI_PATH2)