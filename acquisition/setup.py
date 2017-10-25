from setuptools import setup
from os.path import exists
import daq

setup(name='daq',
      version=daq.__version__,
      description='Data Acquisition module for Brain Computer Interfaces',
      long_description=open('README.md').read() if exists("README.md") else "",
      url='',
      author='OCTRI Clinical Reseach Informatics Apps Team, OHSU',
      author_email='criapps@ohsu.edu',
      license='Apache-2.0',
      keywords='daq data acquisition',
      packages=['daq'],
      install_requires=open('requirements.txt').read().split('\n'),
      tests_require=['pytest'],
      zip_safe=False,
      classifiers=['Development Status :: 2 - Planning',
                   'License :: OSI Approved :: Apache Software License',
                   "Programming Language :: Python",
                   "Programming Language :: Python :: 2.7",
                   "Programming Language :: Python :: 3"])
