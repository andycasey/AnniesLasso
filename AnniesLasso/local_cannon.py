#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A basic command line interface for a local version of The Cannon.
"""

import argparse
import logging
import multiprocessing as mp
import numpy as np
import os
from random import shuffle
from six.moves import cPickle as pickle
from six import string_types

from astropy.table import Table

import AnniesLasso as tc


logger = logging.getLogger("AnniesLasso")

_DEFAULT_FILENAME_COLUMN = "FILENAME"
_DEFAULT_OUTPUT_SUFFIX = "result"


def get_neighbours(labelled_set, star, label_names, K, exclude=None):
    """
    Return `K` indices of the nearest neighbours of `star` in the `labelled_set`.

    :param labelled_set:
        A table containing high fidelity labels for all stars.

    :param star:
        The star to determine neighbours to.

    :param label_names:
        A list of label names that will be used in the model, and which will be
        used to gauge proximity in label space.

    :param K:
        The number of neighbours to return.

    :param exclude: [optional]
        A list-like of indices to exclude from the list of neighbouring indices.
        For example, this may be the index that corresponds to `star`.

    :returns:
        An array of indices of length `K`.
    """

    # Pivot and rescale the labels.
    D = np.sum(np.abs([(labelled_set[_] - star[_])/np.ptp(labelled_set[_]) \
        for _ in label_names]), axis=0)

    assert np.all(np.isfinite(D))

    if exclude is not None:
        if isinstance(exclude, int): exclude = [exclude]
        max_D = 1 + np.max(D)
        for index in exclude:
            D[index] = max_D

    return np.argsort(D)[:K]


def loocv(labelled_set, label_names, K=None, model_order=1, 
    filename_column=None, output_suffix=None, overwrite=False, **kwargs):
    """
    Perform leave-one-out cross-validation using a local Cannon model at every
    point in the labelled set.

    :param labelled_set:
        The path of a table containing labels of stars, and a column including
        the path of a spectrum for that star.

    :param label_names:
        A list of label names to include in the model.

    :param K: [optional]
        The number of K nearby training set stars to train a model with. If
        `None` is specified, then `K = 2 * N_labels`

    :param model_order: [optional]
        The polynomial order of the model to use. If the `model_order` is given
        as 3, and A is a label, then `A^3` is a term in the model.

    :param filename_column: [optional]
        The name of a column in the `labelled_set` filename that refers to the 
        path of a spectrum for that star.  If `None` is given, it defaults to
        {_DEFAULT_FILENAME_COLUMN}

    :param output_suffix: [optional]
        A string suffix that will be appended to the path of every spectrum
        path. If `None` is given, it defaults to {_DEFAULT_OUTPUT_SUFFIX}

    :param overwrite: [optional]
        Overwrite paths of existing result files.
    """

    filename_column = filename_column or _DEFAULT_FILENAME_COLUMN
    output_suffix = output_suffix or _DEFAULT_OUTPUT_SUFFIX
    K = K or 2 * label_names
    if 1 > model_order:
        raise ValueError("model order must be greater than zero")
    if 2 > K:
        raise ValueError("K must be greater than 1")

    if kwargs.get("shuffle", False):
        labelled_set = shuffle(labelled_set)

    results = []

    failed, N = (0, len(labelled_set))
    for i, star in enumerate(labelled_set):

        spectrum_filename = star[filename_column]
        basename, _ = os.path.splitext(spectrum_filename)
        output_filename = "{}.pkl".format("-".join([basename, output_suffix]))

        logger.info("At star {0}/{1}: {2}".format(i + 1, N, spectrum_filename))

        if os.path.exists(output_filename) and not overwrite:
            logger.info("Output filename {} already exists and not overwriting."\
                .format(output_filename))
            continue

        # [1] Load the spectrum.
        try:
            with open(spectrum_filename, "rb") as fp:
                test_flux, test_ivar = pickle.load(fp)

        except:
            logger.exception("Error when loading {}".format(spectrum_filename))
            failed += 1
            continue

        # [2] What are the K closest neighbours?
        indices = get_neighbours(labelled_set, star, label_names, K, exclude=(i, ))

        # [3] Load those K stars and train a model.
        train_flux = np.ones((K, test_flux.size))
        train_ivar = np.zeros_like(train_flux)

        for j, index in enumerate(indices):

            with open(labelled_set[filename_column][j], "rb") as fp:
                flux, ivar = pickle.load(fp)
            
            train_flux[j, :] = flux
            train_ivar[j, :] = ivar


        # [4] Train a model using those K nearest neighbours.
        model = tc.L1RegularizedCannonModel(labelled_set[indices], train_flux, 
            train_ivar, threads=kwargs.get("threads", 1))
        
        # TODO: Revisit this. Should these default to zero?
        model.s2 = 0
        model.regularization = 0

        model.vectorizer = tc.vectorizer.NormalizedPolynomialVectorizer(
            labelled_set[indices], 
            tc.vectorizer.polynomial.terminator(label_names, model_order))

        model.train()
        model._set_s2_by_hogg_heuristic()

        # [5] Test on that star, using the initial labels.
        result, cov, meta = model.fit(test_flux, test_ivar, 
            initial_labels=[star[label_name] for label_name in label_names],
            full_output=True)
        results.append([spectrum_filename] + list(result.flatten()))

        # Insert a flag as to whether the result is within a convex hull of the
        # labelled set.
        meta = meta[0] # The first (and only) star we tested against.
        meta["in_convex_hull"] = model.in_convex_hull(result)[0]
        
        with open(output_filename, "wb") as fp:
            pickle.dump((result, cov, meta), fp, 2) # For legacy.
        logger.info("Saved output to {}".format(output_filename))

        # Close the pool
        if model.pool is not None:
            model.pool.close()
            model.pool.join()

        del model

    logger.info("Number of failures: {}".format(failed))
    logger.info("Number of successes: {}".format(N - failed))

    # Make the comparisons to the original set!
    t = Table(rows=results, names=["FILENAME"] + list(label_names))
    t.write("cannon-local-loocv-{}.fits".format(output_suffix),
        overwrite=overwrite)

    return None



def _loocv_wrapper(labelled_set, label_names, **kwargs):
    if isinstance(label_names, string_types):
        label_names = label_names.split(",")

    if isinstance(labelled_set, string_types):
        labelled_set = Table.read(labelled_set)

    return loocv(labelled_set, label_names, **kwargs)




def _train_and_test(labelled_set, train_flux, train_ivar, label_names,
    model_order, test_flux, test_ivar, initial_labels, output_filename,
    **kwargs):

    print("Doing {} in parallel".format(output_filename))

    # [4] Train a model using those K nearest neighbours.
    model = tc.L1RegularizedCannonModel(labelled_set, train_flux, train_ivar, 
        **kwargs)
    
    # TODO: Revisit this. Should these default to zero?
    model.s2 = 0
    model.regularization = 0

    model.vectorizer = tc.vectorizer.NormalizedPolynomialVectorizer(
        labelled_set, 
        tc.vectorizer.polynomial.terminator(label_names, model_order))

    model.train(progressbar=False)
    model._set_s2_by_hogg_heuristic()

    # [5] Test on that star, using the initial labels.
    result, cov, meta = model.fit(test_flux, test_ivar, 
        initial_labels=initial_labels, full_output=True)
    
    with open(output_filename, "wb") as fp:
        pickle.dump((result, cov, meta), fp, 2) # For legacy.
    logger.info("Saved output to {}".format(output_filename))

    if model.pool is not None:
        model.pool.close()
        model.pool.join()

    del model

    return None


def test(labelled_set, test_set, label_names, K=None, model_order=1,
    filename_column=None, output_suffix=None, overwrite=False, **kwargs):
    """
    Perform the test step on stars in the test set, by building local Cannon
    models from stars in the labelled set.

    :param labelled_set:
        The path of a table containing labels of stars, and a column including
        the path of a spectrum for that star.

    :param test_set:
        The path of a table containing initial labels of stars to test, as well
        as a column including the path of a spectrum for that star.

    :param label_names:
        A list of label names to include in the model.

    :param K: [optional]
        The number of K nearby training set stars to train a model with. If
        `None` is specified, then `K = 2 * N_labels`

    :param model_order: [optional]
        The polynomial order of the model to use. If the `model_order` is given
        as 3, and A is a label, then `A^3` is a term in the model.

    :param filename_column: [optional]
        The name of a column in the `labelled_set` filename that refers to the 
        path of a spectrum for that star.  If `None` is given, it defaults to
        {_DEFAULT_FILENAME_COLUMN}

    :param output_suffix: [optional]
        A string suffix that will be appended to the path of every spectrum
        path. If `None` is given, it defaults to {_DEFAULT_OUTPUT_SUFFIX}

    :param overwrite: [optional]
        Overwrite paths of existing result files.
    """

    filename_column = filename_column or _DEFAULT_FILENAME_COLUMN
    output_suffix = output_suffix or _DEFAULT_OUTPUT_SUFFIX
    K = K or 2 * label_names
    if 1 > model_order:
        raise ValueError("model order must be greater than zero")
    if 2 > K:
        raise ValueError("K must be greater than 1")

    if kwargs.get("shuffle", False):
        test_set = shuffle(test_set)

    threads = kwargs.get("threads", 1)
    threads = threads if threads > 0 else mp.cpu_count()

    pool = None if threads < 2 else mp.Pool(threads)

    processes = []
    failed, N = (0, len(test_set))
    for i, star in enumerate(test_set):

        spectrum_filename = star[filename_column]
        basename, _ = os.path.splitext(spectrum_filename)
        output_filename = "{}.pkl".format("-".join([basename, output_suffix]))

        logger.info("At star {0}/{1}: {2}".format(i + 1, N, spectrum_filename))

        if os.path.exists(output_filename) and not overwrite:
            logger.info("Output filename {} already exists and not overwriting."\
                .format(output_filename))
            continue

        # [1] Load the spectrum.
        try:
            with open(spectrum_filename, "rb") as fp:
                test_flux, test_ivar = pickle.load(fp)

        except:
            logger.exception("Error when loading {}".format(spectrum_filename))
            failed += 1
            continue

        # [2] What are the K closest neighbours?
        indices = get_neighbours(labelled_set, star, label_names, K)

        # [3] Load those K stars and train a model.
        train_flux = np.ones((K, test_flux.size))
        train_ivar = np.zeros_like(train_flux)

        for j, index in enumerate(indices):

            with open(labelled_set[filename_column][j], "rb") as fp:
                flux, ivar = pickle.load(fp)
            
            train_flux[j, :] = flux
            train_ivar[j, :] = ivar

        # --- parallelism can begin here
        initial_labels = [star[label_name] for label_name in label_names]
        args = (labelled_set[indices], train_flux, train_ivar, label_names,
            model_order, test_flux, test_ivar, initial_labels, output_filename)

        if pool is None:
            _train_and_test(*args)

        else:
            processes.append(pool.apply_async(_train_and_test, args))

            while len(processes) >= threads:
                processes.pop(0).get()

        # --- parallelism can end here

    if pool is not None:
        logger.info("Cleaning up the pool..")
        pool.close()
        pool.join()

    logger.info("Number of failures: {}".format(failed))
    logger.info("Number of successes: {}".format(N - failed))

    return None


def _test_wrapper(labelled_set, test_set, label_names, **kwargs):
    if isinstance(label_names, string_types):
        label_names = label_names.split(",")

    if isinstance(labelled_set, string_types):
        labelled_set = Table.read(labelled_set)

    if isinstance(test_set, string_types):
        test_set = Table.read(test_set)

    return test(labelled_set, test_set, label_names, **kwargs)


def main():
    """ A command line tool parser for the Mini Cannon. """

    # Create the main parser.
    parser = argparse.ArgumentParser(
        description="The Cannon", epilog="http://TheCannon.io")

    # Create parent parser.
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "-v", "--verbose", dest="verbose", action="store_true", default=False, 
        help="Verbose logging mode")
    parent_parser.add_argument(
        "-t", "--threads", dest="threads", type=int, default=1,
        help="The number of threads to use")
    parent_parser.add_argument(
        "--condor", dest="condor", action="store_true", default=False,
        help="Distribute action using Condor")
    parent_parser.add_argument(
        "--condor-chunks", dest="condor_chunks", type=int, default=100,
        help="The number of chunks to distribute across Condor. "\
             "This argument is ignored if --condor is not used")
    parent_parser.add_argument(
        "--condor-memory", dest="memory", type=int, default=2000,
        help="The amount of memory (MB) to request for each Condor job. "\
             "This argument is ignored if --condor is not used")
    parent_parser.add_argument(
        "--condor-check-frequency", dest="condor_check_frequency", default=1,
        help="The number of seconds to wait before checking Condor jobs")

    subparsers = parser.add_subparsers(title="action", dest="action",
        description="Specify the action to perform")

    loocv_parser = subparsers.add_parser("loocv", parents=[parent_parser],
        help="Perform leave-one-out cross-validation")
    loocv_parser.add_argument(
        "labelled_set", type=str,
        help="Path of a file containing labels for stars, as well as a column "\
             "that refers to the path for a spectrum of that star")
    loocv_parser.add_argument(
        "label_names", type=str,
        help="List of label names (in the `labelled_set` file) to include in "\
             "the model. These should be separated by a comma")
    loocv_parser.add_argument(
        "--K", dest="K", type=int, default=None,
        help="The number of nearest labelled set neighbours to train on. "\
             "By default, this will be set to 2 * N_labels")
    loocv_parser.add_argument(
        "--model-order", dest="model_order", type=int, default=1,
        help="The maximum order of the label. For example, if A is a label, "\
             "and `model_order` is 3, then that implies `A^3` is a model term")
    loocv_parser.add_argument(
        "--filename-column", dest="filename_column", type=str,
        help="Name of the column in the `labelled_set` that refers to the "\
             "path location of the spectrum for that star")
    loocv_parser.add_argument(
        "--output-suffix", dest="output_suffix", type=str,
        help="A string suffix that will be added to the spectrum filenames "\
             "when creating the result filename")
    loocv_parser.add_argument(
        "--overwrite", action="store_true", default=False,
        help="Overwrite existing result files")
    loocv_parser.add_argument(
        "--shuffle", action="store_true", default=False,
        help="Shuffle the input spectra (useful for running multiple jobs "\
             "in parallel)")
    loocv_parser.set_defaults(func=_loocv_wrapper)


    test_parser = subparsers.add_parser("test", parents=[parent_parser],
        help="Run the test step (infer labels for stars) from spectra")
    test_parser.add_argument(
        "labelled_set", type=str,
        help="Path of a file containing labels for stars, as well as a column "\
             "that refers to the path for a spectrum of that star")
    test_parser.add_argument(
        "test_set", type=str,
        help="Path of a file containing initial labels for stars, as well as a"\
             " column that refers to the path for a spectrum of that star")
    test_parser.add_argument(
        "label_names", type=str,
        help="List of label names (in the `labelled_set` file) to include in "\
             "the model. These should be separated by a comma")
    test_parser.add_argument(
        "--K", dest="K", type=int, default=None,
        help="The number of nearest labelled set neighbours to train on. "\
             "By default, this will be set to 2 * N_labels")
    test_parser.add_argument(
        "--model-order", dest="model_order", type=int, default=1,
        help="The maximum order of the label. For example, if A is a label, "\
             "and `model_order` is 3, then that implies `A^3` is a model term")
    test_parser.add_argument(
        "--filename-column", dest="filename_column", type=str,
        help="Name of the column in the `labelled_set` that refers to the "\
             "path location of the spectrum for that star")
    test_parser.add_argument(
        "--output-suffix", dest="output_suffix", type=str,
        help="A string suffix that will be added to the spectrum filenames "\
             "when creating the result filename")
    test_parser.add_argument(
        "--overwrite", action="store_true", default=False,
        help="Overwrite existing result files")
    test_parser.add_argument(
        "--shuffle", action="store_true", default=False,
        help="Shuffle the input list of spectra (useful for running parallel "\
             "jobs)")
    test_parser.set_defaults(func=_test_wrapper)

    args = parser.parse_args()
    if args.action is None: return

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    return args.func(**args.__dict__)


if __name__ == "__main__":

    main()
