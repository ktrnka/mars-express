from __future__ import unicode_literals
import sys
import argparse
import time


def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def main():
    args = parse_args()


if __name__ == "__main__":
    sys.exit(main())


def number_string(number, singular_unit, plural_unit, format_string="{} {}"):
    return format_string.format(number, singular_unit if number == 1 else plural_unit)


class Timed(object):
    """Decorator for timing how long a function takes"""
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        start_time = time.time()
        self.func(*args, **kwargs)
        elapsed = time.time() - start_time

        hours, seconds = divmod(elapsed, 60 * 60)
        minutes = seconds / 60.
        time_string = number_string(minutes, "minute", "minutes", format_string="{:.1f} {}")
        if hours:
            time_string = ", ".join((number_string(hours, "hour", "hours"), time_string))

        print "{} took {}".format(self.func.__name__, time_string)