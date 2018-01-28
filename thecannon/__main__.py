#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" A command line utility for The Cannon. """

import argparse
import logging
import os
from collections import OrderedDict
from numpy import ceil, loadtxt, zeros, nan, diag, ones
from subprocess import check_output
from six.moves import cPickle as pickle
from tempfile import mkstemp
from time import sleep


def fit(model_filename, spectrum_filenames, threads, clobber, from_filename,
    **kwargs):
    """
    Fit a series of spectra.
    """

    import AnniesLasso as tc

    model = tc.load_model(model_filename, threads=threads)
    logger = logging.getLogger("AnniesLasso")
    assert model.is_trained

    chunk_size = kwargs.pop("parallel_chunks", 1000) if threads > 1 else 1
    fluxes = []
    ivars = []
    output_filenames = []
    failures = 0

    fit_velocity = kwargs.pop("fit_velocity", False)

    # MAGIC HACK
    delete_meta_keys = ("fjac", ) # To save space...
    initial_labels = loadtxt("initial_labels.txt")

    if from_filename:
        with open(spectrum_filenames[0], "r") as fp:
            _ = list(map(str.strip, fp.readlines()))
        spectrum_filenames = _

    output_suffix = kwargs.get("output_suffix", None)
    output_suffix = "result" if output_suffix is None else str(output_suffix)
    N = len(spectrum_filenames)
    for i, filename in enumerate(spectrum_filenames):
        logger.info("At spectrum {0}/{1}: {2}".format(i + 1, N, filename))

        basename, _ = os.path.splitext(filename)
        output_filename = "-".join([basename, output_suffix]) + ".pkl"
        
        if os.path.exists(output_filename) and not clobber:
            logger.info("Output filename {} already exists and not clobbering."\
                .format(output_filename))
            continue

        try:
            with open(filename, "rb") as fp:
                flux, ivar = pickle.load(fp)
                fluxes.append(flux)
                ivars.append(ivar)

            output_filenames.append(output_filename)

        except:
            logger.exception("Error occurred loading {}".format(filename))
            failures += 1

        else:
            if len(output_filenames) >= chunk_size:
                
                results, covs, metas = model.fit(fluxes, ivars,
                    initial_labels=initial_labels, model_redshift=fit_velocity,
                    full_output=True)

                for result, cov, meta, output_filename \
                in zip(results, covs, metas, output_filenames):

                    for key in delete_meta_keys:
                        if key in meta:
                            del meta[key]

                    with open(output_filename, "wb") as fp:
                        pickle.dump((result, cov, meta), fp, 2) # For legacy.
                    logger.info("Saved output to {}".format(output_filename))
                
                del output_filenames[0:], fluxes[0:], ivars[0:]


    if len(output_filenames) > 0:
        
        results, covs, metas = model.fit(fluxes, ivars, 
            initial_labels=initial_labels, model_redshift=fit_velocity,
            full_output=True)

        for result, cov, meta, output_filename \
        in zip(results, covs, metas, output_filenames):

            for key in delete_meta_keys:
                if key in meta:
                    del meta[key]

            with open(output_filename, "wb") as fp:
                pickle.dump((result, cov, meta), fp, 2) # For legacy.
            logger.info("Saved output to {}".format(output_filename))
        
        del output_filenames[0:], fluxes[0:], ivars[0:]


    logger.info("Number of failures: {}".format(failures))
    logger.info("Number of successes: {}".format(N - failures))

    return None




def join_results(output_filename, result_filenames, model_filename=None, 
    from_filename=False, clobber=False, errors=False, cov=False, **kwargs):
    """
    Join the test results from multiple files into a single table file.
    """

    import AnniesLasso as tc
    from astropy.table import Table, TableColumns

    meta_keys = kwargs.pop("meta_keys", {})
    meta_keys.update({
        "chi_sq": nan,
        "r_chi_sq": nan,
        "snr": nan,
    #    "redshift": nan,
    })

    logger = logging.getLogger("AnniesLasso")

    # Does the output filename already exist?
    if os.path.exists(output_filename) and not clobber:
        logger.info("Output filename {} already exists and not clobbering."\
            .format(output_filename))
        return None

    if from_filename:
        with open(result_filenames[0], "r") as fp:
            _ = list(map(str.strip, fp.readlines()))
        result_filenames = _

    # We might need the label names from the model.
    if model_filename is not None:
        model = tc.load_model(model_filename)
        assert model.is_trained
        label_names = model.vectorizer.label_names
        logger.warn(
            "Results produced from newer models do not need a model_filename "\
            "to be specified when joining results.")

    else:
        with open(result_filenames[0], "rb") as fp:
            contents = pickle.load(fp)
            if "label_names" not in contents[-1]:
                raise ValueError(
                    "cannot find label names; please provide the model used "\
                    "to produce these results")
            label_names = contents[-1]["label_names"]


    # Load results from each file.
    failed = []
    N = len(result_filenames)

    # Create an ordered dictionary of lists for all the data.
    data_dict = OrderedDict([("FILENAME", [])])
    for label_name in label_names:
        data_dict[label_name] = []
        
    if errors:
        for label_name in label_names:
            data_dict["E_{}".format(label_name)] = []
    
    if cov:
        data_dict["COV"] = []

    for key in meta_keys:
        data_dict[key] = []
    
    # Iterate over all the result filenames
    for i, filename in enumerate(result_filenames):
        logger.info("{}/{}: {}".format(i + 1, N, filename))

        if not os.path.exists(filename):
            logger.warn("Path {} does not exist. Continuing..".format(filename))
            failed.append(filename)
            continue

        with open(filename, "rb") as fp:
            contents = pickle.load(fp)

        assert len(contents) == 3, "You are using some old school version!"
        
        labels, Sigma, meta = contents

        if Sigma is None:
            Sigma = nan * ones((labels.size, labels.size))

        result = [filename] + list(labels) 
        if errors:
            result.extend(diag(Sigma)**0.5) 
        if cov:
            result.append(Sigma)
        result += [meta.get(k, v) for k, v in meta_keys.items()]

        for key, value in zip(data_dict.keys(), result):
            data_dict[key].append(value)

    # Warn of any failures.
    if failed:
        logger.warn(
            "The following {} result file(s) could not be found: \n{}".format(
                len(failed), "\n".join(failed)))

    # Construct the table.
    table = Table(TableColumns(data_dict))
    table.write(output_filename, overwrite=clobber)
    logger.info("Written to {}".format(output_filename))
    



def main():
    """
    The main command line interpreter. This is the console script entry point.
    """

    # Create the main parser.
    parser = argparse.ArgumentParser(
        description="The Cannon", epilog="http://TheCannon.io")

    # Create parent parser.
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("-v", "--verbose",
        dest="verbose", action="store_true", default=False, 
        help="Verbose logging mode.")
    parent_parser.add_argument("-t", "--threads",
        dest="threads", type=int, default=1,
        help="The number of threads to use.")
    
    # Allow for multiple actions.
    subparsers = parser.add_subparsers(title="action", dest="action",
        description="Specify the action to perform.")

    # Fitting parser.
    fit_parser = subparsers.add_parser("fit", parents=[parent_parser],
        help="Fit stacked spectra using a trained model.")
    fit_parser.add_argument("model_filename", type=str,
        help="The path of a trained Cannon model.")
    fit_parser.add_argument("spectrum_filenames", nargs="+", type=str,
        help="Paths of spectra to fit.")
    fit_parser.add_argument("--parallel-chunks", dest="parallel_chunks",
        type=int, default=1000, help="The number of spectra to fit in a chunk.")
    fit_parser.add_argument("--clobber", dest="clobber", default=False,
        action="store_true", help="Overwrite existing output files.")
    fit_parser.add_argument(
        "--output-suffix", dest="output_suffix", type=str,
        help="A string suffix that will be added to the spectrum filenames "\
             "when creating the result filename")
    fit_parser.add_argument("--from-filename", dest="from_filename",
        action="store_true", default=False, help="Read spectrum filenames from file")
    fit_parser.set_defaults(func=fit)


    # Join results parser.
    join_parser = subparsers.add_parser("join", parents=[parent_parser],
        help="Join results from individual stars into a single table.")
    join_parser.add_argument("output_filename", type=str,
        help="The path to write the output filename.")
    join_parser.add_argument("result_filenames", nargs="+", type=str,
        help="Paths of result files to include.")
    join_parser.add_argument("--from-filename", 
        dest="from_filename", action="store_true", default=False,
        help="Read result filenames from a file.")
    join_parser.add_argument(
        "--errors", dest="errors", default=False, action="store_true", 
        help="Include formal errors in destination table.")
    join_parser.add_argument(
        "--cov", dest="cov", default=False, action="store_true", 
        help="Include covariance matrix in destination table.")
    join_parser.add_argument(
        "--clobber", dest="clobber", default=False, action="store_true", 
        help="Ovewrite an existing table file.")

    join_parser.set_defaults(func=join_results)

    # Parse the arguments and take care of any top-level arguments.
    args = parser.parse_args()
    if args.action is None: return

    logger = logging.getLogger("AnniesLasso")
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Do things.
    return args.func(**args.__dict__)


if __name__ == "__main__":

    """
    Usage examples:
    # tc train model.pickle --condor --chunks 100
    # tc train model.pickle --threads 8
    # tc join model.pickle --from-filename files

    """
    _ = main()
