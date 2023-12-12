import datetime
import logging
import sys

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
# Create handlers for logging to the standard output and a file
stdoutHandler = logging.StreamHandler(stream=sys.stdout)
fileHandler = logging.FileHandler(f"bcipy/simulator/sim_output_{datetime.datetime.now()}.log")

# Set the log levels on the handlers
stdoutHandler.setLevel(logging.INFO)
fileHandler.setLevel(logging.DEBUG)

# Create a log format using Log Record attributes
fmt_file = logging.Formatter(
    "%(levelname)s | %(filename)s:%(lineno)s >> %(message)s"
)

fmt = logging.Formatter(
    "%(levelname)s >> %(message)s"
)


# Set the log format on each handler
stdoutHandler.setFormatter(fmt)
fileHandler.setFormatter(fmt_file)

# Add each handler to the Logger object
log.addHandler(stdoutHandler)
log.addHandler(fileHandler)
