import unittest
import shutil
import os
import errno
from mockito import mock, unstub, when, any, kwargs
import psychopy
from bcipy.tasks.rsvp.copy_phrase import RSVPCopyPhraseTask
from bcipy.helpers.save import init_save_data_structure
from bcipy.helpers.load import load_json_parameters
from bcipy.acquisition.device_info import DeviceInfo
from bcipy.acquisition.record import Record

class TestStartTask(unittest.TestCase):
    """This is a Test Case for Starting a copy phrase task."""

    def setUp(self):
        """Set Up."""
        self.parameters = load_json_parameters('bcipy/parameters/parameters.json',
                                               value_cast=True)

        self.parameters['num_sti'] = 2
        self.parameters['len_sti'] = 3

        self.win = psychopy.visual.Window(size=[1,1], screen=0,
                                    allowGUI=False, useFBO=False, fullscr=False,
                                    allowStencil=False, monitor='mainMonitor',
                                    winType='pyglet', units='norm', waitBlanking=False,
                                    color='black')
        self.daq = mock()
        self.daq.offset = 0
        self.daq.marker_writer = mock()
        self.daq.device_info = DeviceInfo(name='LSL', fs=10,
                                              channels=['c1'])
        self.static_clock = mock()
        self.experiment_clock = mock()
        self.classifier = mock()

        try:
            os.makedirs('data/test_user_002')
        except OSError as error:
            self.assertEqual(error.errno, errno.EEXIST)
            
        self.file_save = init_save_data_structure(
            'data/',
            'test_user_002',
            'bcipy/parameters/parameters.json')

    def tearDown(self):
        """Tear Down."""
        shutil.rmtree('data/test_user_002')
        unstub()

    def test_execute_copy_phrase(self):
        when(psychopy.event).getKeys(keyList=any(list)).thenReturn(['space'])
        when(self.daq.marker_writer).push_marker(marker=any(), lsl_time=any()).thenReturn(None)
        when(self.daq).get_data(**kwargs).thenReturn([Record([0] * 10,0,0)] * 2000)
        copy_phrase_task = RSVPCopyPhraseTask(self.win, self.daq, self.parameters, self.file_save, self.classifier, None, True)
        copy_phrase_task.execute()