#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
General utility functions.
"""

__all__ = ["progressbar", "short_hash"]

import logging
import sys
from time import time
from collections import Iterable
from hashlib import md5

logger = logging.getLogger(__name__)


def short_hash(contents):
    """
    Return a short hash string of some iterable content.

    :param contents:
        The contents to calculate a hash for.

    :returns:
        A concatenated string of 10-character length hashes for all items in the
        contents provided.
    """
    if not isinstance(contents, Iterable): contents = [contents]
    return "".join([str(md5(str(item).encode("utf-8")).hexdigest())[:10] \
        for item in contents])


def progressbar(iterable, message=None, size=100):
    """
    A progressbar.

    :param iterable:
        Some iterable to show progress for.

    :param message: [optional]
        A string message to show as the progressbar header.

    :param size: [optional]
        The size of the progressbar. If the size given is zero or negative,
        then no progressbar will be shown.
    """

    # Preparerise.
    t_init = time()
    count = len(iterable)
    def _update(i, t=None):
        if 0 >= size: return
        increment = max(1, int(count / 100))
        if i % increment == 0 or i in (0, count):
            sys.stdout.write("\r[{done}{not_done}] {percent:3.0f}%{t}".format(
                done="=" * int(i/increment),
                not_done=" " * int((count - i)/increment),
                percent=100. * i/count,
                t="" if t is None else " ({0:.0f}s)".format(t-t_init)))
            sys.stdout.flush()

    # Initialise.
    if size > 0:
        logger.info((message or "").rstrip())
        sys.stdout.flush()

    # Updaterise.
    for i, item in enumerate(iterable):
        yield item
        _update(i)

    # Finalise.
    if size > 0:
        _update(count, time())
        sys.stdout.write("\r\n")
        sys.stdout.flush()

