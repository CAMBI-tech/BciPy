import pip, os, sys

cDir = os.path.dirname(os.path.realpath(sys.argv[0]))
print 'Current directory: ' + cDir

f = open("\\..\\requirements.txt",'r')
answer = raw_input('Is your OS 64 Bit?(type y or n and press enter): ')

if answer.lower()[0] == 'y':
    scipy = cDir + '\\whl\\scipy-0.19.1-cp27-cp27m-win_amd64.whl'
    numpy = cDir + '\\whl\\numpy-1.13.1+mkl-cp27-cp27m-win_amd64.whl'
elif answer.lower()[0] == 'n':
    scipy = cDir + '\\whl\\scipy-0.19.1-cp27-cp27m-win32.whl'
    numpy = cDir + '\\whl\\numpy-1.13.1+mkl-cp27-cp27m-win32.whl'

try:
    pip.main(['install', numpy])
    pip.main(['install', scipy])
except Exception as e:
    print(e)
    print('Can\'t find correct wheel files next to moduleLoader.py:')
    print('numpy-1.13.1+mkl-cp27-cp27m-win_amd64.whl')
    print('scipy-0.19.1-cp27-cp27m-win_amd64.whl')

for line in f:
    packagesInstalled = [package.project_name for package in pip.get_installed_distributions()]
    print 'Loading ' + line
    if line[0:line.find('=')] not in packagesInstalled:
        pip.main(['install', line])
    else:
        print 'Already loaded module = ' + line

raw_input('Press enter to exit: ')