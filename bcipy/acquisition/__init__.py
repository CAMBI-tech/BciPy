"""Data Acquisition module. This module streams data from the EEG hardware,
persists it in csv files, and makes it available to other systems."""
import sys
from os.path import dirname
sys.path.append(dirname(__file__))
sys.path.append('.')
sys.path.append('..')
