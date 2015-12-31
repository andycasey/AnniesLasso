#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
General utility functions.
"""

__all__ = ["InterruptiblePool", "progressbar", "short_hash", "wrapper", "_init_pool"]

import functools
import logging
import multiprocessing
import signal
import sys
from time import time
from collections import Iterable
from hashlib import md5
from multiprocessing.pool import Pool
from multiprocessing import Lock, TimeoutError, Value
from six.moves import cPickle as pickle
from tempfile import mkstemp

#mp = multiprocessing.get_context("spawn")



logger = logging.getLogger(__name__)
_counter = Value('i', 0)
_counter_lock = Lock()

import numpy as np

class LargeSerializedObject(object):

    def __init__(self, value):
        # Create a temporary file and pickle the value to the file.
        _, self.path = mkstemp()
        self.shape = value.shape
        m = np.memmap(self.path, dtype=float, mode="w+", shape=self.shape)
        m[:] = value[:]
        m.flush()
        del m

        #with open(self.path, "wb") as fp:
        #    pickle.dump(value, fp, -1)

    def __call__(self):
        #with open(self.path, "rb") as fp:
        #    value = pickle.load(fp)
        m = np.memmap(self.path, dtype=float, mode="c")
        return m.reshape(self.shape)


def _init_pool(args):
    global _counter
    _counter = args

class wrapper(object):
    """
    A generic multiprocessing wrapper with a progressbar.
    """
    def __init__(self, f, args, kwargs, N, message=None, size=100):
        self.f = f
        self.args = list(args if args is not None else [])
        self.kwargs = kwargs if kwargs is not None else {}
        self._init_progressbar(message, N)
        
    def _init_progressbar(self, message, N):
        """
        Initialise a progressbar.
        """
        self.N = N
        self.t_init = time()
        self.message = message
        if message is not None:
            logger.info(message.rstrip())
            sys.stdout.flush()
            with _counter_lock:
                _counter.value = 0
            
    def _update_progressbar(self):
        """
        Increment the progressbar by one iteration.
        """
        global _counter, _counter_lock
        with _counter_lock:
            _counter.value += 1

        index = _counter.value
        if self.message is None: return

        increment = max(1, int(self.N/100))
        t = time() if index >= self.N else None

        #if index % increment == 0 or index in (0, self.N) and self.N > 0:
        status = "({0}/{1})   ".format(index, self.N) if t is None else \
                 "({0:.0f}s)                      ".format(t-self.t_init)
        sys.stdout.write(
            "\r[{done}{not_done}] {percent:3.0f}% {status}".format(
            done="=" * int(index/increment),
            not_done=" " * int((self.N - index)/increment),
            percent=100. * index/self.N,
            status=status))
        sys.stdout.flush()

        if t is not None:
            sys.stdout.write("\r\n")
            sys.stdout.flush()


    def __call__(self, x):
        try:
            result = self.f(*(list(x) + self.args), **self.kwargs)
        except:
            logger.exception("Exception within wrapped function")
            raise

        self._update_progressbar()
        return result
        




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
        #if i % increment == 0 or i in (0, count) and count > 0:
        status = "({0}/{1})   ".format(i, count) if t is None else \
                 "({0:.0f}s)                      ".format(t-t_init)
        sys.stdout.write("\r[{done}{not_done}] {percent:3.0f}% {status}".format(
            done="=" * int(i/increment),
            not_done=" " * int((count - i)/increment),
            percent=100. * i/count,
            status=status))
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

"""
Python's multiprocessing.Pool class doesn't interact well with
``KeyboardInterrupt`` signals, as documented in places such as:

* `<http://stackoverflow.com/questions/1408356/>`_
* `<http://stackoverflow.com/questions/11312525/>`_
* `<http://noswap.com/blog/python-multiprocessing-keyboardinterrupt>`_

Various workarounds have been shared. Here, we adapt the one proposed in the
last link above, by John Reese, and shared as

* `<https://github.com/jreese/multiprocessing-keyboardinterrupt/>`_

Our version is a drop-in replacement for multiprocessing.Pool ... as long as
the map() method is the only one that needs to be interrupt-friendly.

Contributed to `emcee` by Peter K. G. Williams <peter@newton.cx>,
and copied here.
"""

def _initializer_wrapper(actual_initializer, *rest):
    """
    We ignore SIGINT. It's up to our parent to kill us in the typical
    condition of this arising from ``^C`` on a terminal. If someone is
    manually killing us with that signal, well... nothing will happen.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if actual_initializer is not None:
        actual_initializer(*rest)


class InterruptiblePool(Pool):
    """
    A modified version of :class:`multiprocessing.pool.Pool` that has better
    behavior with regard to ``KeyboardInterrupts`` in the :func:`map` method.
    :param processes: (optional)
        The number of worker processes to use; defaults to the number of CPUs.
    :param initializer: (optional)
        Either ``None``, or a callable that will be invoked by each worker
        process when it starts.
    :param initargs: (optional)
        Arguments for *initializer*; it will be called as
        ``initializer(*initargs)``.
    :param kwargs: (optional)
        Extra arguments. Python 2.7 supports a ``maxtasksperchild`` parameter.
    """
    wait_timeout = 3600

    def __init__(self, processes=None, initializer=None, initargs=(),
                 **kwargs):
        new_initializer = functools.partial(_initializer_wrapper, initializer)
        super(InterruptiblePool, self).__init__(processes, new_initializer,
                                                initargs, **kwargs)

    def map(self, func, iterable, chunksize=None):
        """
        Equivalent of ``map()`` built-in, without swallowing
        ``KeyboardInterrupt``.
        :param func:
            The function to apply to the items.
        :param iterable:
            An iterable of items that will have `func` applied to them.
        """
        # The key magic is that we must call r.get() with a timeout, because
        # a Condition.wait() without a timeout swallows KeyboardInterrupts.
        r = self.map_async(func, iterable, chunksize)

        while True:
            try:
                return r.get(self.wait_timeout)
            except TimeoutError:
                pass
            except KeyboardInterrupt:
                self.terminate()
                self.join()
                raise
            # Other exceptions propagate up.
