# INFO: send data from the EyeTracker to labstreaminglayer
#
################################
# SETUP HERE
#

################################
# Preface here
#
import numpy as np
import tobii_research as tr
import time
import random
import os
import pylsl as lsl
import sys

# Find Eye Tracker
ft = tr.find_all_eyetrackers()
#if len(ft) == 0:
#    print "No Eye Trackers found!?"
#    exit(1)

while len(ft) == 0:
     print("No Eye Trackers found!? - retry in 5 seconds...")
     time.sleep(5)
     ft = tr.find_all_eyetrackers()

# Pick first tracker
mt = ft[0]
print(f"Found Tobii Tracker at '{mt.address}'")

sampling_rate = mt.get_gaze_output_frequency() #sampling rate as recorded by the eye tracker
channels = 31 # count of the below channels, incl. those that are 3 or 2 long
gaze_stuff = [
    ('device_time_stamp', 1),

    ('left_gaze_origin_validity',  1),
    ('right_gaze_origin_validity',  1),

    ('left_gaze_origin_in_user_coordinate_system',  3),
    ('right_gaze_origin_in_user_coordinate_system',  3),

    ('left_gaze_origin_in_trackbox_coordinate_system',  3),
    ('right_gaze_origin_in_trackbox_coordinate_system',  3),

    ('left_gaze_point_validity',  1),
    ('right_gaze_point_validity',  1),

    ('left_gaze_point_in_user_coordinate_system',  3),
    ('right_gaze_point_in_user_coordinate_system',  3),

    ('left_gaze_point_on_display_area',  2),
    ('right_gaze_point_on_display_area',  2),

    ('left_pupil_validity',  1),
    ('right_pupil_validity',  1),

    ('left_pupil_diameter',  1),
    ('right_pupil_diameter',  1)
]

def unpack_gaze_data(gaze_data):
    x = []
    for s in gaze_stuff:
        d = gaze_data[s[0]]
        if isinstance(d, tuple):
            x = x + list(d)
        else:
            x.append(d)
    return x

last_report = 0
N = 0

def gaze_data_callback(gaze_data):
    '''send gaze data'''

    '''
    This is what we get from the tracker:

    device_time_stamp

    left_gaze_origin_in_trackbox_coordinate_system (3)
    left_gaze_origin_in_user_coordinate_system (3)
    left_gaze_origin_validity
    left_gaze_point_in_user_coordinate_system (3)
    left_gaze_point_on_display_area (2)
    left_gaze_point_validity
    left_pupil_diameter
    left_pupil_validity

    right_gaze_origin_in_trackbox_coordinate_system (3)
    right_gaze_origin_in_user_coordinate_system (3)
    right_gaze_origin_validity
    right_gaze_point_in_user_coordinate_system (3)
    right_gaze_point_on_display_area (2)
    right_gaze_point_validity
    right_pupil_diameter
    right_pupil_validity

    system_time_stamp
    '''

    try:
        global last_report
        global outlet
        global N
        global halted

        sts = gaze_data['system_time_stamp'] / 1000000. #local machine timestamp - can be mapped to timestamp sent by Presentation
        local_ts = lsl.local_clock()

        outlet.push_sample(unpack_gaze_data(gaze_data), sts)

        if sts > last_report + 5:
            sys.stdout.write("latency: %14.3f seconds \n" % (local_ts-sts))
            sys.stdout.write("timestamp: %14.3f with %10d packets\n" % (local_ts, N))
            last_report = sts
        N += 1

        # print unpack_gaze_data(gaze_data)
    except:
        print("Error in callback: ")
        print(sys.exc_info())

        halted = True


def start_gaze_tracking():
    mt.subscribe_to(tr.EYETRACKER_GAZE_DATA, gaze_data_callback, as_dictionary=True)
    return True

def end_gaze_tracking():
    mt.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, gaze_data_callback)
    return True

halted = False

# Set up lsl stream
def setup_lsl():
    global channels
    global gaze_stuff
    global sampling_rate

    info = lsl.StreamInfo('Tobii', 'ET', channels, sampling_rate, 'float32', mt.address)
    info.desc().append_child_value("manufacturer", "Tobii")
    channels = info.desc().append_child("channels")
    cnt = 0
    for s in gaze_stuff:
        if s[1]==1:
            cnt += 1
            channels.append_child("channel") \
                    .append_child_value("label", s[0]) \
                    .append_child_value("unit", "device") \
                    .append_child_value("type", 'ET')
        else:
            for i in range(s[1]):
                cnt += 1
                channels.append_child("channel") \
                        .append_child_value("label", "%s_%d" % (s[0], i)) \
                        .append_child_value("unit", "device") \
                        .append_child_value("type", 'ET')

    outlet = lsl.StreamOutlet(info)

    return outlet

outlet = setup_lsl()

# Main loop; run until escape is pressed
print(f"{lsl.local_clock()}: LSL Running; press CTRL-C repeatedly to stop")
start_gaze_tracking()
try:
    while not halted:
        time.sleep(1)
        keys = ()  # event.getKeys()
        if len(keys) != 0:
            if keys[0]=='escape':
                halted = True

        if halted:
            break

        # print lsl.local_clock()

except:
    print("Halting...")

print("terminating tracking now")
end_gaze_tracking()