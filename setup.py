#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Modified from https://github.com/kennethreitz/setup.py
# Note: To use the 'upload' functionality of this file, you must:
#   $ pip install twine

import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

# Package meta-data.
NAME = 'bcipy'
DESCRIPTION = 'Python Software for Brain-Computer Interface.'
URL = 'https://github.com/BciPy/BciPy'
EMAIL = 'memmott@ohsu.com'
AUTHOR = 'Tab Memmott'
REQUIRES_PYTHON = '>=3.6.5'

VERSION = '1.3.0'


# What packages are required for this module to be executed?
REQUIRED = [
    'wheel==0.30.0',
    'wxPython==4.0.0',
    'configobj==5.0.6',
    'docker==2.6.1',
    'json-tricks==3.8.0',
    'olefile==0.44',
    'PyOpenGL==3.1.0',
    'pyOpenSSL==17.5.0',
    'mne==0.17.0',
    'lxml==4.1.1',
    'PsychoPy==3.0.4',
    'pytz==2017.2',
    'six==1.11.0',
    'numpy==1.16.2',
    'scipy==1.2.1',
    'future==0.16.0',
    'sklearn==0.0',
    'seaborn==0.9.0',
    'construct==2.8.14',
    'matplotlib==2.1.1',
    'pylsl==1.13.1',
    'psutil==5.4.0',
    'pandas==0.24.1',
    'sounddevice==0.3.10',
    'SoundFile==0.10.1',
    'PySoundCard==0.5.2',
    'PySoundFile==0.9.0',
    'Pillow==4.3.0',
]

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
# with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
#     long_description = '\n' + f.read()

long_description = 'Python Brain-Computer Interface Software'

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    with open(os.path.join(here, NAME, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION


class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPi via Twine…')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')

        sys.exit()


# Where the magic happens:
setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=('tests', 'demo', 'data', )),
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['mypackage'],

    entry_points={
        'console_scripts': ['mycli=mymodule:cli'],
    },
    install_requires=REQUIRED,
    include_package_data=True,
    license='MIT',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
    # $ setup.py publish support.
    cmdclass={
        'upload': UploadCommand,
    },
)
