# Modified from https://github.com/kennethreitz/setup.py
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

# Package meta-data.
NAME = 'bcipy'
DESCRIPTION = 'Python Software for Brain-Computer Interface.'
URL = 'https://github.com/CAMBI-tech/BciPy'
EMAIL = 'cambi_support@googlegroups.com'
AUTHOR = 'CAMBI'
REQUIRES_PYTHON = '>3.6,<3.9'

VERSION = '2.0.0rc2'


# What packages are required for this module to be executed?
with open('requirements.txt', 'r', encoding='utf-8') as f:
    REQUIRED = f.read().splitlines()

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

long_description = 'Python Brain-Computer Interface Software'

about = {'__version__': VERSION}


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
    entry_points={
        'console_scripts': ['bcipy = bcipy.main:bcipy_main'],
    },
    install_requires=REQUIRED,
    include_package_data=True,
    license='Hippocratic License 2.1',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: Other/Proprietary License',
        'Topic :: Scientific/Engineering :: Human Machine Interfaces',
        'Intended Audience :: Science/Research',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Developers',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    # $ setup.py publish support.
    cmdclass={
        'upload': UploadCommand,
    },
)
