import unittest
import shutil
import os
import errno
from mockito import mock, unstub, when, any, kwargs
import psychopy
from bcipy.helpers.load import load_json_parameters
from bcipy.tasks.rsvp.icon_to_icon import _init_icon_to_icon_display_task, RSVPIconToIconTask
from bcipy.display.rsvp.rsvp_disp_modes import IconToIconDisplay
from bcipy.acquisition.device_info import DeviceInfo
from bcipy.acquisition.record import Record

class TestIconToIcon(unittest.TestCase):
    def setUp(self):
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

    def tearDown(self):
        shutil.rmtree('data/test_user_002')
        unstub()

    def test_init_icon_to_icon_display(self):
        icon_to_icon_window = _init_icon_to_icon_display_task(self.parameters,
            self.win, self.daq, self.static_clock, self.experiment_clock, False)
        self.assertIsInstance(icon_to_icon_window, IconToIconDisplay)

    def test_execute_icon_to_icon_task(self):
        when(psychopy.event).getKeys(keyList=any(list)).thenReturn(['space'])
        when(self.daq.marker_writer).push_marker(marker=any(), lsl_time=any()).thenReturn(None)
        when(self.daq).get_data(**kwargs).thenReturn([Record([0] * 10,0,0)] * 2000)
        icon_to_icon_task = RSVPIconToIconTask(self.win, self.daq, self.parameters,
            'data/test_user_002', self.classifier, None, True, False, None)
        icon_to_icon_task.execute()
