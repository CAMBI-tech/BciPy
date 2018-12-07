""" Progress Bar for not to bore people """

import time


def progress_bar(iteration, total, prefix='', suffix='', decimals=1,
                 length=100, fill='-'):
    """ Progress bar can be used in any finite iteration.
        Args:
            """

    percent = ("{0:." + str(decimals) + "f}").format(
        100 * (iteration / float(total)))
    fill_length = int(length * iteration // total)
    bar = fill * fill_length + ' ' * (length - fill_length)
    print('\r{} {} %{} {}'.format(prefix, bar, percent, suffix)),
    if iteration == total:
        print('')

def _test_pb():
    # A List of Items
    items = list(range(0, 57))
    l = len(items)

    # Initial call to print 0% progress
    progress_bar(0, l, prefix='Progress:', suffix='Complete', length=50)
    for i, item in enumerate(items):
        # Do stuff...
        time.sleep(0.1)
        # Update Progress Bar
        progress_bar(i + 1, l, prefix='Progress:', suffix='Complete',
                     length=50)

def main():
    _test_pb()

    return 0


if __name__ == "__main__":
    main()
