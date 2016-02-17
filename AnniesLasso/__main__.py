#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A basic command line interface for The Cannon.
"""

import argparse
import logging
import os
from numpy import ceil, zeros
from subprocess import check_output
from six.moves import cPickle as pickle
from tempfile import mkstemp
from time import sleep

condor_code = """
executable     = /opt/ioa/software/python/2.7/bin/python 
universe       = vanilla
output         = {logging_dir}/condor-{prefix}-out.log
error          = {logging_dir}/condor-{prefix}-err.log
log            = {logging_dir}/condor-{prefix}.log
request_cpus   = {cpus}
request_memory = {memory}
Notification   = never
# maybe arguments?
## arguments = <arguments your script would typically take >
arguments      = {executable} train {model_filename}
queue
"""

def _condorlock_filename(path):
    return "{0}/.{1}.condorlock".format(
        os.path.dirname(path), os.path.basename(path))


def fit(model_filename, spectrum_filenames, threads, clobber, from_filename,
    **kwargs):
    """
    Fit a series of spectra.
    """

    import AnniesLasso as tc

    model = tc.load_model(model_filename, threads=threads)
    logger = logging.getLogger("AnniesLasso")
    assert model.is_trained

    chunk_size = 1000 if threads > 1 else 1
    fluxes = []
    ivars = []
    output_filenames = []
    failures = 0

    # MAGIC HACK
    initial_labels = np.loadtxt("initial_labels.txt")

    if from_filename:
        with open(spectrum_filenames[0], "r") as fp:
            _ = map(str.strip, fp.readlines())
        spectrum_filenames = _

    N = len(spectrum_filenames)
    for i, filename in enumerate(spectrum_filenames):
        logger.info("At spectrum {0}/{1}: {2}".format(i + 1, N, filename))

        basename, _ = os.path.splitext(filename)
        output_filename = "{}_result.pkl".format(basename)

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
                    initial_labels=initial_labels, full_output=True)

                for result, cov, meta, output_filename \
                in zip(results, covs, metas, output_filenames):
                    with open(output_filename, "wb") as fp:
                        pickle.dump((result, cov, meta), fp, 2) # For legacy.
                    logger.info("Saved output to {}".format(output_filename))
                
                del output_filenames[0:], fluxes[0:], ivars[0:]


    if len(output_filenames) > 0:
        
        results, covs, metas = model.fit(fluxes, ivars, 
            initial_labels=initial_labels, full_output=True)

        for result, cov, meta, output_filename \
        in zip(results, covs, metas, output_filenames):
            with open(output_filename, "wb") as fp:
                pickle.dump((result, cov, meta), fp, 2) # For legacy.
            logger.info("Saved output to {}".format(output_filename))
        
        del output_filenames[0:], fluxes[0:], ivars[0:]


    logger.info("Number of failures: {}".format(failures))
    logger.info("Number of successes: {}".format(N - failures))

    return None


def train(model_filename, threads, condor, chunks, memory, save_training_data,
    condor_check_frequency, re_train, **kwargs):
    """
    Train an existing model, with the option to distribute the work across many
    threads or condor resources.
    """
    
    # It's bad practice not to have all the imports up here, 
    # but I want the CLI to be fast if people are just checking out the help.
    import AnniesLasso as tc

    # Load the model.
    model = tc.load_model(model_filename, threads=threads)
    logger = logging.getLogger("AnniesLasso")
    if model.is_trained and not re_train:
        logger.warn("Model loaded from {} is already trained.".format(
            model_filename))
        logger.info("Exiting..")
        return model

    if condor:
        # We will need some temporary place to put logs etc...
        # MAGIC
        logging_dir = "logs"
        if not os.path.exists(logging_dir):
            logger.info("Creating Condor log directory: {}".format(logging_dir))
            os.mkdir(logging_dir)

        condor_kwds = {
            # Get the path of this executable, since it may not be available to
            # the child resources.
            "executable": check_output("which tc", shell=True).strip(),
            "logging_dir": logging_dir,
            "cpus": 1, # MAGIC
            "memory": memory
        }

        # Split up the model into chunks based on the number of pixels.
        condor_job = "condor.job" # MAGIC
        chunk_size = int(ceil(model.dispersion.size / float(chunks)))
        chunk_filenames = []
        for i in range(chunks):
            # Let's not assume anything about the model (e.g., it may have many
            # attributes from a sub-class that we do not know about).

            # Just make a soft copy and re-specify the data attributes, 
            # since they should not be any different.
            si, ei = (i * chunk_size, (i + 1) * chunk_size) 
            chunk = model.copy()
            chunk._dispersion = chunk._dispersion[si:ei]
            chunk._normalized_flux = chunk._normalized_flux[:, si:ei]
            chunk._normalized_ivar = chunk._normalized_ivar[:, si:ei]
            assert len(model._data_attributes) == 3, \
                "Don't know what to do with your additional data attributes!"

            # Temporary filename
            _, chunk_filename = mkstemp(dir=os.getcwd(),
                prefix='tc.condorchunk.')
            chunk.save(chunk_filename, 
                include_training_data=True, overwrite=True)
            chunk_filenames.append(chunk_filename)
            logger.info("Saved chunk {0} to {1}".format(i, chunk_filename))
            assert os.path.exists(chunk_filename)

            # Now submit the jobs
            kwds = condor_kwds.copy()
            kwds.update({
                "model_filename": chunk_filename,
                "prefix": "{0}-{1}".format(i, os.path.basename(chunk_filename))
            })

            with open(condor_job, "w") as fp:
                fp.write(condor_code.format(**kwds))

            # Create a Condor lock file.
            condorlock_filename = _condorlock_filename(chunk_filename)
            os.system("touch {}".format(condorlock_filename))

            # Submit the Condor job.
            os.system("chmod +wrx {}".format(condor_job))
            os.system("condor_submit {}".format(condor_job))
            logger.info("Submitted job {0} for {1}".format(i, chunk_filename))

        if os.path.exists(condor_job):
            os.remove(condor_job)

        # Wait for completion of all jobs.
        waiting = map(_condorlock_filename, chunk_filenames)
        logger.info("Waiting for completion of {} jobs".format(len(waiting)))
        while True:
            completed = []
            for each in waiting:
                if not os.path.exists(each):
                    logger.info("Finished {}".format(each))
                    completed.append(each)

            waiting = list(set(waiting).difference(completed))
            logger.info("Still waiting on {0} jobs:\n{1}".format(
                len(waiting), "\n".join(waiting)))

            if len(waiting) == 0:
                break

            sleep(condor_check_frequency)
            # Check for Condor log files  to see if they failed!

        logger.info("Collecting results")

        # Collect the results.
        model.theta = zeros((model.dispersion.size, model.design_matrix.shape[1]))
        model.s2 = zeros((model.dispersion.size))

        for i, chunk_filename in enumerate(chunk_filenames):
            logger.info("Loading from chunk {0} {1}".format(i, chunk_filename))
            chunk = tc.load_model(chunk_filename)
            si, ei = (i * chunk_size, (i + 1) * chunk_size)

            model.theta[si:ei] = chunk.theta[:].copy()
            model.s2[si:ei] = chunk.s2[:].copy()

            os.remove(chunk_filename)

    else:
        model.train(
            op_kwargs={"xtol": kwargs["xtol"], "ftol": kwargs["ftol"]},
            op_bfgs_kwargs={"factr": kwargs["factr"], "pgtol": kwargs["pgtol"]})

    # Save the model.
    logger.info("Saving model to {}".format(model_filename))
    model.save(model_filename, include_training_data=save_training_data,
        overwrite=True)

    # Are we a child Condor process?
    condorlock_filename = _condorlock_filename(model_filename)
    if os.path.exists(condorlock_filename):
        logger.info("Removing Condor lock {}".format(condorlock_filename))
        os.remove(condorlock_filename)

    logger.info("Done")
    return model



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
    parent_parser.add_argument("--condor",
        dest="condor", action="store_true", default=False,
        help="Distribute action using Condor.")
    parent_parser.add_argument("--condor-chunks",
        dest="chunks", type=int, default=100,
        help="The number of chunks to distribute across Condor. "\
             "This argument is ignored if --condor is not used.")
    parent_parser.add_argument("--condor-memory",
        dest="memory", type=int, default=2000,
        help="The amount of memory (MB) to request for each Condor job. "\
             "This argument is ignored if --condor is not used.")
    parent_parser.add_argument("--condor-check-frequency",
        dest="condor_check_frequency", type=int, default=1,
        help="The number of seconds to wait before checking for finished Condor jobs.")


    # Allow for multiple actions.
    subparsers = parser.add_subparsers(title="action", dest="action",
        description="Specify the action to perform.")

    # Training parsers.
    train_parser = subparsers.add_parser("train", parents=[parent_parser],
        help="Train an existing Cannon model.")
    train_parser.add_argument("--save_training_data", default=False,
        action="store_true", dest="save_training_data",
        help="Once trained, save the model using the training data.")
    train_parser.add_argument("--re-train", default=False,
        action="store_true", dest="re_train",
        help="Re-train the model if it is already trained.")
    train_parser.add_argument("model_filename", type=str,
        help="The path of the saved Cannon model.")
    train_parser.add_argument("--factr", default=10000000.0, dest="factr",
        help="BFGS keyword argument")
    train_parser.add_argument("--pgtol", default=1e-5, dest="pgtol",
        help="BFGS keyword argument")
    train_parser.add_argument("--xtol", default=1e-6, dest="xtol",
        help="fmin_powell keyword argument")
    train_parser.add_argument("--ftol", default=1e-6, dest="ftol",
        help="fmin_powell keyword argument")
    train_parser.set_defaults(func=train)


    # Fitting parser.
    fit_parser = subparsers.add_parser("fit", parents=[parent_parser],
        help="Fit stacked spectra using a trained model.")
    fit_parser.add_argument("model_filename", type=str,
        help="The path of a trained Cannon model.")
    fit_parser.add_argument("spectrum_filenames", nargs="+", type=str,
        help="Paths of spectra to fit.")
    fit_parser.add_argument("--clobber", dest="clobber", default=False,
        type=bool, help="Overwrite existing output files.")
    fit_parser.add_argument("--from-filename", dest="from_filename",
        action="store_true", default=False, help="Read spectrum filenames from file")
    fit_parser.set_defaults(func=fit)

    # Parse the arguments and take care of any top-level arguments.
    args = parser.parse_args()
    if args.action is None: return

    logger = logging.getLogger("AnniesLasso")
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if args.chunks != parent_parser.get_default("chunks") and not args.condor:
        logger.warn("Ignoring chunks argument because Condor is not in use.")

    # Do things.
    return args.func(**args.__dict__)


if __name__ == "__main__":

    """
    Usage examples:
    # tc train model.pickle --condor --chunks 100
    # tc train model.pickle --threads 8
    """
    _ = main()