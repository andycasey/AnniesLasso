#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
General utility functions.
"""

__all__ = ["InterruptiblePool", "short_hash", "wrapper"]

import functools
import logging
import os
import signal
import sys
from six import string_types
from six.moves import cPickle as pickle
from tempfile import mkstemp
from time import time
from collections import Iterable
from hashlib import md5
from multiprocessing.pool import Pool
from multiprocessing import Lock, TimeoutError, Value

logger = logging.getLogger(__name__)


# Initialize global counter for incrementing between threads.
_counter = Value('i', 0)
_counter_lock = Lock()

def _init_pool(args):
    global _counter
    _counter = args

class wrapper(object):
    """
    A generic wrapper with a progressbar, which can be used either in serial or
    in parallel.

    :param f:
        The function to apply.

    :param args:
        Additional arguments to supply to the function `f`.

    :param kwds:
        Keyword arguments to supply to the function `f`.

    :param N:
        The number of items that will be iterated over.

    :param message: [optional]
        An information message to log before showing the progressbar.

    :param size: [optional]
        The width of the progressbar in characters.
    """
    def __init__(self, f, args, kwds, N, message=None, size=100):
        self.f = f
        self.args = list(args if args is not None else [])
        self.kwds = kwds if kwds is not None else {}
        self._init_progressbar(N, message)
        

    def _init_progressbar(self, N, message=None):
        """
        Initialise a progressbar.

        :param N:
            The number of items that will be iterated over.
        
        :param message: [optional]
            An information message to log before showing the progressbar.
        """

        self.N = int(N)
        
        try:
            rows, columns = os.popen('stty size', 'r').read().split()

        except:
            logger.debug("Couldn't get screen size. Progressbar may look odd.")
            self.W = 100

        else:
            self.W = min(100, int(columns) - (12 + 21 + 2 * len(str(self.N))))

        self.t_init = time()
        self.message = message
        if 0 >= self.N:
            return None

        if message is not None:
            logger.info(message.rstrip())
        
        sys.stdout.flush()
        with _counter_lock:
            _counter.value = 0
            

    def _update_progressbar(self):
        """
        Increment the progressbar by one iteration.
        """
        
        if 0 >= self.N:
            return None

        global _counter, _counter_lock
        with _counter_lock:
            _counter.value += 1

        index = _counter.value
        
        increment = max(1, int(self.N/float(self.W)))
        
        eta_minutes = ((time() - self.t_init) / index) * (self.N - index) / 60.0
        
        if index >= self.N:
            status = "({0:.0f}s) ".format(time() - self.t_init)

        elif float(index)/self.N >= 0.05 \
        and eta_minutes > 1: # MAGIC fraction for when we can predict ETA
            status = "({0}/{1}; ~{2:.0f}m until finished)".format(
                        index, self.N, eta_minutes)

        else:
            status = "({0}/{1})                          ".format(index, self.N)

        sys.stdout.write(
            ("\r[{done: <" + str(self.W) + "}] {percent:3.0f}% {status}").format(
            done="=" * int(index/increment),
            percent=100. * index/self.N,
            status=status))
        sys.stdout.flush()

        if index >= self.N:
            sys.stdout.write("\r\n")
            sys.stdout.flush()


    def __call__(self, x):
        try:
            result = self.f(*(list(x) + self.args), **self.kwds)
        except:
            logger.exception("Exception within wrapped function")
            raise

        self._update_progressbar()
        return result


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


def _unpack_value(value):
    """
    Unpack contents if it is pickled to a temporary file.

    :param value:
        A non-string variable or a string referring to a pickled file path.

    :returns:
        The original value, or the unpacked contents if a valid path was given.
    """

    if isinstance(value, string_types) and os.path.exists(value):
        with open(value, "rb") as fp:
            contents = pickle.load(fp)
        return contents
    return value


def _pack_value(value, protocol=-1):
    """
    Pack contents to a temporary file.

    :param value:
        The contents to temporarily pickle.

    :param protocol: [optional]
        The pickling protocol to use.

    :returns:
        A temporary filename where the contents are stored.
    """
    
    _, temporary_filename = mkstemp()
    with open(temporary_filename, "wb") as fp:
        pickle.dump(value, fp, protocol)
    return temporary_filename
