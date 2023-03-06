
import time
import tobii_research as tr

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
print("Found Tobii Tracker at '{mt.address}'")


calibration = tr.ScreenBasedCalibration(mt)

# Enter calibration mode.
calibration.enter_calibration_mode()

# Define the points on screen we should calibrate at.
# The coordinates are normalized, i.e. (0.0, 0.0) is the upper left corner and (1.0, 1.0) is the lower right corner.
points_to_calibrate = [(0.5, 0.5), (0.1, 0.1), (0.1, 0.9), (0.9, 0.1), (0.9, 0.9)]

for point in points_to_calibrate:
 print("Show a point on screen at {0}.".format(point))

 # Wait a little for user to focus.
 time.sleep(1)

 print("Collecting data at {0}.".format(point))
 if calibration.collect_data(point[0], point[1]) != tr.CALIBRATION_STATUS_SUCCESS:
    # Try again if it didn't go well the first time.
     # Not all eye tracker models will fail at this point, but instead fail on ComputeAndApply.
     calibration.collect_data(point[0], point[1])

print("Computing and applying calibration.")
calibration_result = calibration.compute_and_apply()
print("Compute and apply returned {0} and collected at {1} points.".
   format(calibration_result.status, len(calibration_result.calibration_points)))

# Analyze the data and maybe remove points that weren't good.
#recalibrate_point = (0.1, 0.1)
#print("Removing calibration point at {0}.".format(recalibrate_point))
#calibration.discard_data(recalibrate_point[0], recalibrate_point[1])

# Redo collection at the discarded point
#print("Show a point on screen at {0}.".format(recalibrate_point))
#calibration.collect_data(recalibrate_point[0], recalibrate_point[1])

# Compute and apply again.
#print("Computing and applying calibration.")
#calibration_result = calibration.compute_and_apply()
#print("Compute and apply returned {0} and collected at {1} points.".
#   format(calibration_result.status, len(calibration_result.calibration_points)))

# See that you're happy with the result.

# The calibration is done. Leave calibration mode.
calibration.leave_calibration_mode()

print("Left calibration mode.")
# <EndExample>