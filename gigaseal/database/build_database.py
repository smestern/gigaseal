###########
# This script builds the database for the web visualization
# Here we we are going to take a folder of ABF or NWB files, and extract some features
# from each file. We will then save the features in a database file.
# The database file will be used by the web visualization to display the data.
# The database file is a JSON file, or csv
#
# The per-file extraction is not implemented yet. The DANDI explorer's long square
# pipeline (data_for_specimen_id, plot_data, build_dataset_traces) now lives in
# intraSPCA_DANDI_explorer/dandi_scraper; the stimulus/sweep primitives it uses are in
# gigaseal.database.protocol_detect and re-exported here.
###########
import glob
import argparse
import logging
from multiprocessing import freeze_support

from gigaseal.utils import arg_wrap
from gigaseal.database.protocol_detect import *
from gigaseal.database.protocol_detect import (
    _ONTOLOGY, _UNIT_ONTOLOGY, _holding_current, _longest_stim_epoch, _group_long_square_runs,
    _plausible_vm, _choose_volt_scale, _iter_current_clamp_traces, _step_stim_amp,
)

log = logging.getLogger(__name__)


def glob_files(folder, ext="nwb"):

    #this function will take a folder and a file extension, and return a list of files
    # with that extension in that folder
    return glob.glob(folder + "/**/*." + ext, recursive=True)


def run_analysis(folder, backend="ipfx", outfile='out.csv', ext="nwb", parallel=False, n_jobs=4):
    #TODO: per-file feature extraction for a folder of ABF/NWB files, returning a DataFrame indexed by specimen_id.
    #For long squares: loadNWB(f, return_obj=True) -> long_square.select_long_square_run -> long_square.long_square_features
    raise NotImplementedError("per-file extraction not implemented yet; see dandi_scraper.extract for the DANDI pipeline")


def main(argv=None):
    #main function to be called when running this script
    # Handle command line arguments
    parser = argparse.ArgumentParser(description='Build database for web visualization')
    parser.add_argument('folder', type=str, nargs='?', default=None, help='Folder containing ABF or NWB files')
    parser.add_argument('--data_folder', type=str, default=None, help='Folder containing ABF or NWB files, supersedes folder')
    parser.add_argument('--backend', type=str, default="ipfx", help='Backend to use for feature extraction')
    parser.add_argument('--outfile', type=str, default="out.csv", help='Output file name') 
    parser.add_argument('--ext', type=str, default="nwb", help='File extension to search for in the folder')

    if argv is not None: #called from code, e.g. run_web_viz, so don't prompt
        args = parser.parse_args(argv)
    else:
        args = arg_wrap(parser, cli_prompt=True).parse_args()  # Wrap the parser to catch exceptions
    #args should have folder or data_folder
    if args.folder is None and args.data_folder is None:
        raise ValueError("You must provide a folder with ABF or NWB files, either with the folder argument or the --data_folder argument")
    if args.data_folder is not None: #data_folder will supercede folder
        args.folder = args.data_folder
    # Run analysis
    return run_analysis(args.folder, args.backend, args.outfile, ext=args.ext)


if __name__ == "__main__":
    freeze_support()
    #call main 
    main()
