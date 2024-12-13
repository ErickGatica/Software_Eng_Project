# Author: rjw
"""
Python code base for manipulating DAQ interferogram files
into phase-corrected data.

Quick user guide of useful commands:
IG = open_daq_files(filename_string_without_extension)
len(IG) # number of interferograms
SNR_peak_to_peak = (np.max(IG[0]) - np.min(IG[0])) / np.std(IG[0][:500])
spectrum_raw = np.abs(np.fft.fft(IG[0])[:len(IG[0])//2])
IG.set_mode(PC) # work from .cor phase corrected file instead of raw file
IG.set_mode(RAW) # work from raw file, this is default
s = Spectra(IG, num_pcs = len(IG)) # phase-correct entire interferogram into one spectrum over Nyquist = 0.1:0.4
transmission = np.abs(s[0])
# Or just phase-correct first 100 IGs over Nyquist window 0.17-0.44
# (search for 'DEFAULTS[' in this script to see default values)
s_quick_look = Spectra(IG, num_pcs = 100, stop = 100, pc_lim_low = .17, pc_lim_high = .44)
transmission = np.abs(s[0])

Word to the wise:
Be careful about real vs complex values when converting from interferogram <-> spectra
.cor files are complex interferograms, and .raw files are real interferograms

updates:
        8/10/2021    Adria    -Program automatically changes the DateFormat in VC707 Constants depending on version
"""
# TODO:
#   Add in DCSLoaded data for IGs, not just spectra
#       How to handle DCS data that is either spectra or IGs?
#   Move phase correction to functions instead of in class initializations?
#       Might make custom phase correction routines easier?
#   Redo file organization methods to work with new daqfiles class
#
#   How to calc p2p based on input data?
#       NI raw data range +/-; VC707 depends on num_hwavgs

import os
import time
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from shutil import copy
from zipfile import ZipFile, ZIP_DEFLATED
from itertools import repeat
from math import ceil

# from scipy.signal import hilbert
from enum import IntEnum
from re import search as re_search
from numbers import Integral, Number
from traceback import format_exc, print_exc

from pldspectrapy.constants import SPEED_OF_LIGHT, M_2_CM
from pldspectrapy.misc_tools import compare_end_and_total


# Define modes for DAQ file organization
class FileModes(IntEnum):
    COPY = 0
    MOVE = 1
    ZIPCOPY = 2
    ZIPMOVE = 3


# Data mode for DAQFiles
class DataMode(IntEnum):
    RAW = 0
    PC = 1


class IterableInt(int):
    def __iter__(daq_file):
        return repeat(daq_file)

    def __getitem__(daq_file, i):
        return daq_file


def ceildiv(a, b):
    return -(-a // b)


def intify(string):
    return int(round(float(string)))


def unwrap(a, r=2 * np.pi, thresh=0.5):
    """
    An unwrap function with an arbitrary range (numpy locked at 2pi)
    Currently assumes a is 1d array
    """
    try:  # If range r is not integer but a is make sure b is not integer
        if isinstance(a[0], Integral) and not isinstance(r, Integral):
            b = np.array(a, np.float64)
        else:
            b = np.copy(a)
    except IndexError:
        return np.empty(0)
    r = abs(r)
    thresh_value = r * abs(thresh)
    d = np.diff(a)
    b[1:] -= (
        np.cumsum(np.subtract(d > thresh_value, d < -thresh_value, dtype=np.int8)) * r
    )
    return b
    # Iterative solution
    correction = 0
    for ii in range(1, len(a)):
        diff = a[ii] - a[ii - 1]
        if abs(diff) > thresh_value:
            correction += np.sign(diff) * r
        b[ii] -= correction
    return b


def force_fit_line(x, y, thresh=0.05):
    """
    2 step line fit_td: fits a linear model to a set of data, then removes
    outliers and refits a line.

    INPUTS:
        x = array_like, shape (M,): x-coordinates of the M sample
            points (x[i], y[i]).
        y = array_like, shape (M,): y-coordinates of the sample points.
        thresh = fractional threshold, in terms of the max residual_td from the
            initial line fit_td, above which those points will be ignored in the
            second line fit_td. Can be set to None to weight fits based on
            1/residual_td.

    OUTPUTS:
        p = Fit line coefficients (slope, y-intercept)

    """
    linefit = np.polyfit(x, y, 1)
    line = np.poly1d(linefit)
    r = np.abs(line(x) - y)

    if thresh is not None:
        thresh_value = thresh * np.max(r)
        weights = (r < thresh_value).astype(int)
        if np.sum(weights) < 5:
            thresh = None
    if thresh is None:
        weights = 1 / r

    return np.polyfit(x, y, 1, w=weights)


def str_lst_search(list_of_strings, substring):
    """
    Looks for a string within a list of strings

    INPUTS:
        list_of_strings = list of strings to search within
        substring       = target string to find

    OUTPUTS:
        (row, location) = tuple containing the row and location, both int,
                          of the target string. Otherwise returns (-1,-1) as
                          failure condition
    """
    for row, string in enumerate(list_of_strings):
        location = string.find(substring)
        if location > -1:
            return (row, location)
    return (-1, -1)


def get_lines(item):
    try:
        item.seek(0)
    except AttributeError:
        try:
            return item.readlines()
        except AttributeError:
            try:
                with open(item, "r") as f:
                    return f.readlines()
            except (IOError, TypeError):
                try:
                    return item.split("\n")
                except AttributeError:
                    return item  # assume item is list of string settings
    else:
        return item.readlines()


# Define constants for concatenation, parsing, and testing
EXT_LOG_FILE = "log"
EXT_RAW_FILE = "raw"
EXT_PCD_FILE = "cor"
EXT_PCP_FILE = "corpar"
EXTS = (EXT_LOG_FILE, EXT_RAW_FILE, EXT_PCD_FILE, EXT_PCP_FILE)


def remove_extension(fname):
    for ext in EXTS:  # try to remove any known spectroscopy extension
        f_ext = fname[-(len(ext) + 1) :].lower()
        if f_ext == "." + ext or f_ext == "_" + ext:
            return fname[: -(len(ext) + 1)]
    return fname


def is_populated_log(fname):
    fname = fname.strip()
    ext = fname[-len(EXT_LOG_FILE) - 1 :].lower()
    if ext == "." + EXT_LOG_FILE or ext == "_" + EXT_LOG_FILE:
        try:
            return bool(os.path.getsize(fname))
        except OSError:
            pass
    return False


class Constants:
    dtype_raw = None
    dtype_pc = None

    @classmethod
    def parse_log(cls, log):
        raise NotImplementedError


class NIConstants(Constants):
    start_time = "Start time"
    start_time_fmt_1 = "Start time: %m/%d/%Y %H:%M:%S\n"
    start_time_fmt_2 = "Start time: %m/%d/%Y %I:%M:%S %p\n"
    frame_length = "Frame length"
    num_hwavgs = "Num of Avg"
    num_pcs_realtime = "# phasecorrections"
    pc_lim_low = "LowFreq"
    pc_lim_high = "HighFreq"

    # The LabVIEW software uses big-endian style
    # More information: https://en.wikipedia.org/wiki/Endianness
    dtype_raw = np.dtype(np.float64).newbyteorder(">")
    dtype_pc = np.dtype(np.complex128).newbyteorder(">")

    @classmethod
    def parse_log(cls, log):
        line_list = get_lines(log)

        # Grab the start time row
        time_row, _ = str_lst_search(line_list, cls.start_time)
        # Parse to datetime
        try:
            s = {
                "start_time": datetime.strptime(
                    line_list[time_row], cls.start_time_fmt_1
                )
            }
        except ValueError:
            s = {
                "start_time": datetime.strptime(
                    line_list[time_row], cls.start_time_fmt_2
                )
            }

        for setting, cast in (
            ("frame_length", int),
            ("num_hwavgs", int),
            ("num_pcs_realtime", int),
            ("pc_lim_low", float),
            ("pc_lim_high", float),
        ):
            name_row, _ = str_lst_search(line_list, getattr(cls, setting))
            search_result = re_search("<Val>(.*)</Val>", line_list[name_row + 1])
            s[setting] = cast(search_result.group(1))
        return s


class VC707Constants(Constants):
    file_version = "file version"
    num_hwavgs = "num_hwavgs"
    num_hwavgs_prop = "@num_hwavgs"
    frame_length = "frame_length"
    p2p_range = "p2p_range"
    p2p_min = "p2p_min"
    p2p_max = "p2p_max"
    p2p_total = "p2p_total"
    pc_win = "pc_win"
    pc_lim_low = "pc_lim_low"
    pc_lim_high = "pc_lim_high"
    start_time = "start_time_local"
    start_time_utc = "start_time_utc"
    timezone = "timezone"
    byte_order = "byte_order"
    trigger_name = "trigger_name"
    timeout = "timeout"
    max_time = "max_time"
    runtime = "runtime"
    num_igs_accepted = "igms_accepted"
    num_igs_received = "igms_received"
    time_fmt_2 = "%Y%m%d%H%M%S%z"
    time_fmt = "%Y%m%d%H%M%S"
    notes = "notes"
    num_pcs_realtime = "num_pcs"
    pc_time = "pc_time"
    igs_per_pc = "igs_per_pc"
    # Parameters for file version >= 1.0.1
    p2p_rejects = "igms_filtered_p2p"
    ghost_rejects = "igms_filtered_ghosts"

    fr1_setpoint = "fr1_setpoint"
    fr1 = "fr1"
    fr2 = "fr2"
    dfr = "dfr"

    cb_loc = "cb_loc"
    cb_loc_std = "cb_loc_std"
    daq_bits = 14

    dtype_raw = np.dtype(np.int32)
    dtype_pc = np.dtype(np.complex128)

    # parameters for file version >= 1.0.4
    daq_info = "daq_info"

    @classmethod
    def parse_log(cls, log):
        line_list = get_lines(log)
        a = {}
        s = {"file_version": (1, 0, 0)}
        for ldx in line_list:
            # TODO: if notes is multiple lines will only read in 1st line
            ldx = ldx.strip()
            split = ldx.split(" = ", 1)
            if len(split) < 2:
                split = ldx.lower().split(cls.file_version)
                if len(split) > 1:
                    version = split[1].split(".")
                    if len(version) == 3:
                        s["file_version"] = tuple(int(v) for v in version)
                        continue
                    raise ValueError("Unrecognized file version identifier: '%s'" % ldx)
                continue
            name = split[0].strip().lower()
            a[name] = split[1]
        if s["file_version"] > (1, 0, 1):
            cls.time_fmt_2 = "%Y%m%d%H%M%S%z"
        else:
            cls.time_fmt_2 = "%Y%m%d%H%M%S"

        s["frame_length"] = intify(a.pop(cls.frame_length))
        s["p2p_total"] = float(a.pop(cls.p2p_total))
        s["start_time"] = datetime.strptime(a.pop(cls.start_time), cls.time_fmt_2)
        s["start_time_utc"] = datetime.strptime(a.pop(cls.start_time_utc), cls.time_fmt)
        s["timezone"] = str(a.pop(cls.timezone))
        s["byte_order"] = a.pop(cls.byte_order)
        s["runtime"] = float(a.pop(cls.runtime))
        s["num_igs_accepted"] = intify(a.pop(cls.num_igs_accepted))
        s["num_igs_received"] = intify(a.pop(cls.num_igs_received))
        try:
            s["timeout"] = float(a.pop(cls.timeout))
        except (KeyError, ValueError):
            s["timeout"] = float(a.pop(cls.max_time))
        try:
            s["num_hwavgs"] = intify(a.pop(cls.num_hwavgs))
        except (KeyError, ValueError, OverflowError):
            s["num_hwavgs"] = intify(a.pop(cls.num_hwavgs_prop))
        try:
            p2p_range = a.pop(cls.p2p_range).strip("()[] ")
            split = p2p_range.split(",")
            if len(split) != 2:
                split = p2p_range.split(" ")
                if len(split) != 2:
                    raise ValueError("Unknown p2p_range '%s'" % p2p_range)
            p2p_range = tuple(float(x) for x in split)
            s["p2p_min"], s["p2p_max"] = p2p_range
        except (KeyError, ValueError):
            s["p2p_min"] = float(a.pop(cls.p2p_min))
            s["p2p_max"] = float(a.pop(cls.p2p_max))
        try:
            pc_win = a.pop(cls.pc_win).strip("()[] ")
            split = pc_win.split(",")
            if len(split) != 2:
                split = pc_win.split(" ")
                if len(split) != 2:
                    raise ValueError()
            pc_win = tuple(float(x) for x in split)
            s["pc_lim_low"], s["pc_lim_high"] = pc_win
        except (KeyError, ValueError):
            s["pc_lim_low"] = float(a.pop(cls.pc_lim_low))
            s["pc_lim_high"] = float(a.pop(cls.pc_lim_high))

        try:
            s["num_pcs_realtime"] = intify(a.pop(cls.num_pcs_realtime))
        except (KeyError, ValueError, OverflowError):
            s["pc_time"] = float(a.pop(cls.pc_time))
            igs_per_pc = a.pop(cls.igs_per_pc).strip("()[] ")
            split = igs_per_pc.split(",")
            if len(split) < 2:
                # split = igs_per_pc.split(' ')
                split = [0]
            s["igs_per_pc"] = tuple(intify(x) for x in split)
        if s["file_version"] > (1, 0, 0):
            s["p2p_rejects"] = intify(a[cls.p2p_rejects])
            s["ghost_rejects"] = intify(a[cls.ghost_rejects])
        else:
            s["p2p_rejects"] = s["num_igs_received"] - s["num_igs_accepted"]
            s["ghost_rejects"] = 0
        # Parameters termed to be optional
        try:
            s["trigger_name"] = a.pop(cls.trigger_name)
            if s["trigger_name"].lower() == "none":
                s["trigger_name"] = None
        except KeyError:
            pass
        try:
            s["notes"] = a.pop(cls.notes)
        except KeyError:
            pass
        try:
            s["fc_setpoint"] = float(a.pop(cls.fr1_setpoint))
        except (KeyError, ValueError):
            pass
        try:
            s["fr1"] = float(a.pop(cls.fr1))
        except (KeyError, ValueError):
            pass
        try:
            s["fr2"] = float(a.pop(cls.fr2))
        except (KeyError, ValueError):
            pass
        try:
            s["dfr"] = float(a.pop(cls.dfr))
        except (KeyError, ValueError):
            pass
        try:
            s["cb_loc"] = float(a.pop(cls.cb_loc))
        except (KeyError, ValueError):
            pass
        try:
            s["cb_loc_std"] = float(a.pop(cls.cb_loc_std))
        except (KeyError, ValueError):
            pass

        try:
            s["daq_info"] = str(a.pop(cls.daq_info))
        except (KeyError, ValueError) as e:
            pass
        # parameters not currently loaded:
        # pc_mos, p2p_thresh, pc_width, Fr1_setpoint
        return s


class DCSData:
    start_time = None
    start_time_utc = None
    timezone = None
    timeout = None
    runtime = None
    num_igs_accepted = None
    num_igs_received = None
    frame_length = None
    num_hwavgs = None
    num_pcs = None
    p2p_min = None
    p2p_max = None
    p2p_scale = None
    pc_lim_low = None
    pc_lim_high = None
    p2p_total = None
    pc_time = None
    byte_order = None
    trigger_name = None
    data_source = None
    igs_per_pc = None
    file_version = None
    p2p_total = None
    p2p_rejects = None
    ghost_rejects = None
    fc_setpoint = None
    fr1 = None
    fr2 = None
    dfr = None
    cb_loc = None
    cb_loc_std = None
    daq_info = None
    notes = None

    def __init__(daq_file, params):
        # NOTE: this modifies params!
        # TODO: make more efficient
        for name, val in tuple(params.items()):
            if name[0] == "_" or name.upper() == name:
                # reserved names, cannot set
                continue
            try:
                default = getattr(daq_file.__class__, name)
            except AttributeError:
                # not a class attribute, therefore not a valid param
                continue
            else:
                if default is not None and not isinstance(default, (Number, str)):
                    # currently params all default to None, strings, or numbers
                    # TODO: change this is it becomes invalid
                    continue
            del params[name]
            daq_file.__setattr__(name, val)
        if len(params):
            raise ValueError(
                "Unknown parameters requested for %s: %s"
                % (daq_file.__class__.__name__, params)
            )

    def __len__(daq_file):
        return len(daq_file.data)

    def __iter__(daq_file):
        return iter(daq_file.data)

    def __getitem__(daq_file, i):
        return daq_file.data.__getitem__(i)


class DAQFiles(DCSData):
    num_pcs_realtime = None

    CONSTANTS = Constants

    def __init__(daq_file, filename):
        """
        INPUTS:
            filename = string of the filename of DAQ file

        OUTPUTS:
            None

        ERRORS:
            TODO
        """
        # daq_file.base_path = directory + base file name of log file (no extension)
        # daq_file.base_dir = directory (or zip file) conatining DAQ files
        # daq_file.base_fname = log file name with all extensions stripped
        # base_dir + base_fname == base_path
        daq_file.base_path = os.path.realpath(filename.strip())

        daq_file.file_log = None
        daq_file.data_raw = None
        daq_file.data_raw_source = None
        daq_file.data_pc = None
        daq_file.data_pc_source = None
        daq_file.file_corpar = None
        daq_file.zfile = None

        try:
            # If it's a .zip, then use the zfile library to peak inside
            if daq_file.base_path[-4:].lower() == ".zip":
                daq_file.zfile = ZipFile(daq_file.base_path, "r")
                # Iterate through file heirarchy
                for fname in daq_file.zfile.namelist():
                    f_stripped = fname.strip()
                    for ext in EXTS:
                        f_ext = f_stripped[-(len(ext) + 1) :].lower()
                        if f_ext == "." + ext or f_ext == "_" + ext:
                            if ext == EXT_LOG_FILE and daq_file.file_log is None:
                                daq_file.file_log = daq_file.zfile.open(fname, "r")
                                daq_file.base_fname = f_stripped[: -len(ext)]
                                daq_file.base_fname.rstrip("_.")
                            elif (
                                ext == EXT_RAW_FILE and daq_file.data_raw_source is None
                            ):
                                daq_file.data_raw_source = fname
                            elif (
                                ext == EXT_PCD_FILE and daq_file.data_pc_source is None
                            ):
                                daq_file.data_pc_source = fname
                            elif ext == EXT_PCP_FILE and daq_file.file_corpar is None:
                                daq_file.file_corpar = fname
                            break
                if daq_file.file_log:
                    daq_file.base_dir = daq_file.base_path
                    daq_file.base_path = os.path.join(
                        daq_file.base_dir, daq_file.base_fname
                    )

            else:
                daq_file.base_path = remove_extension(daq_file.base_path)
                daq_file.base_dir, daq_file.base_fname = os.path.split(
                    daq_file.base_path
                )
                # Open files as read-only, or read-only-binary defined by extension type
                for ext in EXTS:
                    for fname in (
                        daq_file.base_path + "." + ext,
                        daq_file.base_path + "_" + ext,
                    ):
                        if os.path.isfile(fname):
                            if ext == EXT_LOG_FILE:
                                daq_file.file_log = open(fname, "r")
                            elif ext == EXT_RAW_FILE:
                                daq_file.data_raw = np.memmap(
                                    fname,
                                    dtype=daq_file.CONSTANTS.dtype_raw,
                                    mode="r",
                                )
                                daq_file.data_raw_source = fname
                            elif ext == EXT_PCD_FILE:
                                daq_file.data_pc = np.memmap(
                                    fname,
                                    dtype=daq_file.CONSTANTS.dtype_pc,
                                    mode="r",
                                )
                                daq_file.data_pc_source = fname
                            elif ext == EXT_PCP_FILE:
                                daq_file.file_corpar = fname
                            break

            if not any(
                (
                    daq_file.file_log,
                    daq_file.data_raw_source,
                    daq_file.data_pc_source,
                    daq_file.file_corpar,
                )
            ):
                raise ValueError("No valid DAQ files at '%s'" % daq_file.base_path)
        except:  # noqa: E722
            daq_file.close()
            raise

        if not daq_file.file_log:
            daq_file.failure = "No log file"
        elif not os.path.getsize(daq_file.file_log.name):
            daq_file.failure = "Empty log file"
        elif all(all(c == "\x00" for c in ldx) for ldx in daq_file.file_log):
            daq_file.failure = "Log file filled with null characters"
        else:
            daq_file.failure = False
        if daq_file.failure:
            return
        log_params = daq_file.CONSTANTS.parse_log(daq_file.file_log)
        super().__init__(log_params)
        daq_file._finalize_init()

        # TODO: For now assume a rep rate clock
        # This assumes a stable, correctly locked DCS
        daq_file.is_fc_known = False
        daq_file.fc = 200e6  # Hz of clock signal
        if daq_file.fc_setpoint is None:
            daq_file.dfr = daq_file.fc / daq_file.frame_length
        else:
            daq_file.dfr = daq_file.fc_setpoint / daq_file.frame_length
        daq_file.nyquist_window = 2.0

        daq_file.has_data = False
        if daq_file.zfile:
            if daq_file.data_raw_source is not None:
                daq_file.has_data = True
            elif daq_file.data_pc_source is not None:
                daq_file.has_data = True
            else:
                daq_file.mode = None
                daq_file.data = None
        else:
            if daq_file.data_raw is not None:
                daq_file.data_raw.shape = (-1, daq_file.frame_length)
                if len(daq_file.data_raw):
                    daq_file.has_data = True
                    daq_file.set_mode(DataMode.RAW)
                else:
                    daq_file.data_raw = None
                    daq_file.data_raw_source = None
            if daq_file.data_pc is not None:
                daq_file.data_pc.shape = (-1, daq_file.frame_length)
                if len(daq_file.data_pc):
                    if not daq_file.has_data:
                        daq_file.has_data = True
                        daq_file.set_mode(DataMode.PC)
                else:
                    daq_file.data_pc = None
                    daq_file.data_pc_source = None
            if not daq_file.has_data:
                # TODO: how to handle this, what to set things to
                daq_file.mode = None
                daq_file.data = None

    def _finalize_init(daq_file):
        raise NotImplementedError

    def set_mode(daq_file, mode):
        # TODO: handle zip_source mode set
        if mode == DataMode.RAW:
            if daq_file.data_raw is None:
                raise ValueError("Has no raw IGs")
            daq_file.data = daq_file.data_raw
            daq_file.num_pcs = 1
            daq_file.num_igs_per = IterableInt(1)
        elif mode == DataMode.PC:
            if daq_file.data_pc is None:
                raise ValueError("Has no phase corrected IGs")
            daq_file.data = daq_file.data_pc
            daq_file.num_pcs = daq_file.num_pcs_realtime
            # TODO: fix me!
            # daq_file.num_igs_per = np.empty(len(daq_file.data), dtype=np.uint32)
            # daq_file.num_igs_per[:] = daq_file.num_pcs
            # if len(daq_file.data)*daq_file.num_pcs > daq_file.num_igs_accepted:
            #     daq_file.num_igs_per[-1] = daq_file.num_igs_accepted - (len(daq_file.data) - 1)*daq_file.num_pcs
        else:
            raise ValueError("Data reading mode (%r) unknown" % mode)
        daq_file.mode = mode
        daq_file.data_source = daq_file.data.filename

    def calc_p2p_total(daq_file):
        if daq_file.p2p_total is not None:
            return
        if daq_file.data_raw is not None:
            daq_file.p2p_total = 0
            for ig in daq_file.data_raw:
                p2p = (max(ig) - min(ig)) * daq_file.p2p_scale
                daq_file.p2p_total += p2p
        elif daq_file.data_pc is not None:
            daq_file.p2p_total = 0
            for ig in daq_file.data_pc:
                daq_file.p2p_total += max(abs(ig)) * 2 * daq_file.p2p_scale
        else:
            raise ValueError("Has no data")

    def is_open_elsewhere(daq_file):
        """
        Determine if the file is open elsewhere by trying to rename it

        TODO:
            - Modify so this works on *Unix
            - Does this work if we have the files open? probably not
            - And for the zip files?
        """
        try:
            os.rename(daq_file.file_log.name, daq_file.file_log.name)
            return False
        except WindowsError:
            return True

    def close(daq_file):
        """
        Close everything
        This helps to make sure that other programs can gain access, and Windows won't freak out
        """
        daq_file.data = None
        daq_file.data_raw = None
        daq_file.data_pc = None
        log_file = daq_file.file_log
        if log_file:
            log_file.close()
            daq_file.file_log = None
        zfile = daq_file.zfile
        if zfile:
            zfile.close()
            daq_file.zfile = None

    def __del__(daq_file):
        """
        If this object is deleted, then make sure to close everything
        """
        daq_file.close()


class DAQFilesNI(DAQFiles):
    byte_order = "big"

    CONSTANTS = NIConstants
    DTYPE_RAW = CONSTANTS.dtype_raw
    DTYPE_PC = CONSTANTS.dtype_pc

    def _finalize_init(daq_file):
        if daq_file.data_raw is None:
            # NI DAQ puts no extension on raw data, try to find it
            if daq_file.zfile:
                if daq_file.base_fname in daq_file.zfile.namelist():
                    daq_file.data_raw_source = daq_file.base_fname
            else:
                try:
                    daq_file.data_raw = np.memmap(
                        daq_file.base_path, dtype=daq_file.DTYPE_RAW, mode="r"
                    )
                except (IOError, ValueError):
                    pass
        if daq_file.data_raw is not None:
            daq_file.num_igs_accepted = len(daq_file.data_raw)
        elif daq_file.data_pc is not None:
            daq_file.num_igs_accepted = (
                len(daq_file.data_pc) * daq_file.num_pcs_realtime
            )
        daq_file.num_igs_received = daq_file.num_igs_accepted
        daq_file.p2p_scale = 1


class DAQFilesVC707(DAQFiles):
    CONSTANTS = VC707Constants

    def _finalize_init(daq_file):
        if daq_file.byte_order == "little":
            bo = "<"
        elif daq_file.byte_order == "big":
            bo = ">"
        else:
            raise ValueError("Unrecognized byte order %r" % daq_file.byte_order)
        if daq_file.data_raw is not None:
            daq_file.data_raw.dtype = daq_file.data_raw.dtype.newbyteorder(bo)
        if daq_file.data_pc is not None:
            daq_file.data_pc.dtype = daq_file.data_pc.dtype.newbyteorder(bo)
        daq_file.p2p_scale = 1 / (
            (pow(2, daq_file.CONSTANTS.daq_bits - 1) - 1) * daq_file.num_hwavgs
        )
        daq_file.raw_time = abs(daq_file.dfr**-1) * daq_file.num_hwavgs
        daq_info = daq_file.daq_info
        daq_file.avg_pc_data_info = None
        daq_file.data_raw_pc = None
        daq_file.base_time = None
        daq_file.data_len = None
        daq_file.x_wvn = None
        daq_file.pc_type = daq_file.pc_single_point
        daq_file.pc_kwargs = {}
        if daq_info is not None and "rp" in daq_info.lower():
            raise Exception
        else:
            pass

    def phase_correct(daq_file):
        """
        This is the wrapper function for initiating phase correction
        Parameters
        ----------
        kwargs: dictionary
        Optional arguments to pass to phase correction functions

        Returns
        -------

        """
        pc_type = daq_file.pc_type
        try:
            pc_type()
        except:
            print("Valid pc_type not specified: " + str(pc_type))
            sys.exit()

    def pc_single_point(daq_file):
        """
        This function performs the phase correction of raw data collected by a
        VC707 or Red Pitaya DAQ file.

        Parameters
        ----------
        kwargs:
            sample_width: int
                Apodization width for quickly determining the phase of the IGs
            slide_flag: bool
                Flag indicating whether or not the data should be rotated such that all
                the centerbursts are pre-aligned around 0
        Returns
        -------
        Adds the DAQ file attribute 'data_raw_pc' which contains the phase corrected
        (but not averaged) raw data
        """

        if daq_file.data_raw is None:
            raise Exception("No raw data available for phase correcting")

        slide_flag = daq_file.pc_kwargs.get("slide_flag", False)

        try:
            sample_width = pow(2, int(np.log2(daq_file.pc_kwargs["sample_width"])))
        except:
            sample_width = 2**11

        if slide_flag:
            cbs = find_cb(daq_file.data_raw, daq_file.pc_lim_low, daq_file.pc_lim_high)

            # TODO This function fails with raw data when it should not - this needs to
            #  be investigated
            # If the centerbursts are are too spread we go process the next file
            if check_spread_centerbursts(cbs):
                raise Exception("The centerbursts are too spread")

            # Rotate data around centerburst when flag is true
            daq_file.data_raw = slide_centerburst(daq_file.data_raw, cbs)

        # This portion finds the strongest IG to use for the reference phase
        # The alternative is to just use the first IG - it likely makes little
        # difference
        # ref_ig = np.argmax(np.amax(np.abs(daq_file.data_raw), axis=1))

        ref_ig = 0

        ig0 = daq_file.data_raw[ref_ig]
        is_input_complex = np.iscomplexobj(ig0)
        center = np.argmax(ig0)
        start = center - sample_width // 2
        if start < 0:
            start = 0
        stop = start + sample_width

        if is_input_complex:
            s = np.fft.fft(ig0[start:stop])
            pc_start = int(round(daq_file.pc_lim_low * len(s)))
            pc_stop = int(round(daq_file.pc_lim_high * len(s)))
        else:
            s = np.fft.rfft(ig0[start:stop])
            len_2 = 2 * len(s)
            pc_start = int(round(daq_file.pc_lim_low * len_2))
            pc_stop = int(round(daq_file.pc_lim_high * len_2))
        r = s.real[pc_start:pc_stop]
        i = s.imag[pc_start:pc_stop]
        mag = r * r + i * i
        ref_index = pc_start + np.argmax(mag)
        phase_ref = np.angle(s[ref_index])

        if is_input_complex:
            s = np.fft.fft(daq_file.data_raw[:, start:stop])
        else:
            s = np.fft.rfft(daq_file.data_raw[:, start:stop])
        dphase = np.angle(s[:, ref_index]) - phase_ref
        daq_file.data_raw_pc = daq_file.data_raw * np.exp(-1j * dphase[:, np.newaxis])

    def mobile_axis(daq_file, config_variables, fr2_set=None, df_tooth=None):
        """
        Calculate frequency axis for daq_file-referenced dual-comb.
        Based on setpoint and dfr rather than instantaneous fr1, fr2 which shift by 5 Hz

        f_opt = optical lock frequency and sign (MHz). Where f_CW = f_tooth + f_opt
        wvn_spectroscopy = one frequency inside filtered spectra (cm-1)

        IG is an object from pldspectrapy/igtools.py with all the log file information from comb locking
        IG.fc = repetition rate setpoint of clock comb (sets DAQ clock)
        IG.fr2 = repetition rate of unclocked comb
        IG.frame_length = points per interferogram. If interferogram centerburst is not moving,
                                                    this tells us 1/df_rep
        """
        # Calculate CW laser frequency from individual comb f_reps, IG length, optical beat offset

        f_opt = config_variables["fitting"]["lock_frequency"]
        band_fit = config_variables["fitting"]["band_fit"]
        wvn_spectroscopy = np.mean(band_fit)  # Figure out which nyquist window to use

        f_cw_approx = 191.5e12  # approximate CW reference laser frequency (Hz)
        if fr2_set is None or df_tooth is None:
            fr2_set = daq_file.fc_setpoint - daq_file.dfr
            df_tooth = 0.5 * (daq_file.fc_setpoint + fr2_set)

        df_nyq = daq_file.frame_length * fr2_set
        nyq_cw = np.argmin(np.abs(f_cw_approx - df_nyq * np.asarray(range(20))))
        f_cw = df_nyq * nyq_cw + f_opt

        # determine Nyquist window of spectroscopy
        f_spectroscopy = wvn_spectroscopy * SPEED_OF_LIGHT * M_2_CM
        nyq, sign = divmod(f_spectroscopy, df_nyq)
        if sign / df_nyq < 0.5:
            flip_spectrum = False
        else:
            nyq += 1
            flip_spectrum = True

        # Last make frequency axis array
        x_hz = np.arange(daq_file.frame_length / 2 + 1)
        if flip_spectrum:
            x_hz *= -1
        x_hz = x_hz * df_tooth
        x_hz += nyq * df_nyq - f_opt
        daq_file.x_wvn = x_hz / SPEED_OF_LIGHT / M_2_CM

        return daq_file.x_wvn

    def prep_for_processing(daq_file, config_variables):
        """
        Calls the functions to create the frequency axis, identify data to process (and
        phase correct if needed),and create averaging information - store all as DAQFILE attributes

        Parameters
        ----------
        config_variables : dictionary
            dictionary containing all the values parsed from the config file
            entry examples:
                "lock_freq": 32e6,
                "band_fit": [6007, 6155]

        Returns
        -------

        """

        if daq_file.x_wvn is None:
            daq_file.mobile_axis(config_variables)

        if (daq_file.data is None) or (daq_file.base_time is None):
            daq_file.get_data_to_process(config_variables)

    def get_data_to_process(daq_file, config_variables):
        """
        This function collects the data that will be processed (i.e., phase-correctd
        vs raw) and calculates the base-level time represented by the individual IGs
        Parameters
        ----------
        config_variables: dictionary
            Dictionary containing the data process/handling parameters

        Returns
        -------
        Adds attributes to the DAQFILE object: 'data' -> data to be processing
        (pc vs raw); 'base_time' -> time in seconds for each IG in 'data'; 'data_len' ->
        number of IGs in 'data'
        """
        process_raw = config_variables["input"]["process_raw"]
        if process_raw:
            if daq_file.data_raw_pc is None:
                daq_file.phase_correct()
            daq_file.data = daq_file.data_raw_pc
            daq_file.base_time = daq_file.raw_time
            daq_file.data_len = np.shape(daq_file.data_raw_pc)[0]
        else:
            daq_file.data = daq_file.data_pc
            daq_file.base_time = daq_file.pc_time
            daq_file.data_len = np.shape(daq_file.data_pc)[0]

        daq_file.set_averaging_parameters(len(daq_file.data))

    def set_averaging_parameters(
        daq_file,
        value,
        method="number",
        start=0,
        end=0,
        total=1,
    ):
        """
        This function creates the dictionary (DaqFile attribute) based on the input
        parameters. It allows for number-based or time-based averaging, and then converts all
        time-based parameters to index-based parameters.

        Parameters
        ----------
        daq_file: DaqFile object
            DAQ file object
        value: float
            Base number controlling the averaging - if indexed-based, this is the number
            of IGs to average; if time-based, this is the averaging time (in seconds)
        method: string
            String indicating time- or indexed-based averaging. Use 'number' for
            indexed-based, and 'time' for time-based. Upon
        start: int, optional
            Start value for averaging (default is 0)
        end: int, optional
            End value for averaging (default is 0)
        total: int, optional
            Total averaging (default is 1)

        Returns
        -------

        """
        if method not in ["number", "time"]:
            raise ValueError("Averaging method must be 'number' or 'time'")

        # To preserve back-compatibility we will allow the user to specify a "zero" value
        # This will be interpreted as the full length of the data
        if value == 0:
            value = len(daq_file.data)

        daq_file.avg_pc_data_info = {
            "indices": [],
            "num": int(value) if method == "number" else float(value),
            "idx_start": int(start) if method == "number" else float(start),
            "idx_end": int(end) if method == "number" else float(end),
            "total": int(total) if method == "number" else float(total),
        }

        if method == "time":
            daq_file.convert_time_params()

        daq_file.check_averaging_parameters()

        daq_file.set_averaging_indices()

    def set_averaging_indices(daq_file):
        """
        This function creates the indices for the data to be averaged based on the
        averaging parameters specified in the averaging_variables dictionary.

        Returns
        -------
        Updates the averaging_variables dictionary with the indices of the data to be
        averaged.

        """
        indices = daq_file.avg_pc_data_info["indices"]

        for idx in range(daq_file.avg_pc_data_info["total"]):
            current_start_index = daq_file.avg_pc_data_info["idx_start"] + (
                idx * daq_file.avg_pc_data_info["num"]
            )
            current_end_idx = (
                current_start_index + (daq_file.avg_pc_data_info["num"]) - 1
            )

            if current_end_idx > daq_file.avg_pc_data_info["idx_end"]:
                current_end_idx = daq_file.avg_pc_data_info["idx_end"]
                indices.append([current_start_index, current_end_idx])

                break

            indices.append([current_start_index, current_end_idx])

        daq_file.avg_pc_data_info.update(
            {
                "indices": indices,
                "time_indices": convert_idx_to_time(daq_file.base_time, indices),
            }
        )

    def convert_time_params(daq_file):
        """
        This function updates the number/index-based parameters in averaging_variables
        based on the user-inputs for the time-based averaging (since the data is index-
        rather than time-based).

        Parameters
        ----------
        averaging_variables : dictionary
            Dictionary containing averaging parameters
        base_time : float
            Length of time associated with a single data IG

        Returns
        -------
        averaging_variables : dictionary
            Returns original dictionary with updated number/index-based parameters
        """
        if daq_file.base_time > daq_file.avg_pc_data_info["num"]:
            # TODO I didn't see this warning pop-up...
            warnings.warn(
                f"Base time for Allan Deviation analysis ("
                f"{daq_file.avg_pc_data_info['num']}) is less than the data "
                f"resolution ({daq_file.base_time}). Value has been adjusted to {daq_file.base_time}."
            )
            daq_file.avg_pc_data_info["num"] = daq_file.base_time

        avg_num = int(daq_file.avg_pc_data_info["num"] / daq_file.base_time)
        start_idx = int(daq_file.avg_pc_data_info["idx_start"] / daq_file.base_time)
        end_idx = (
            int(daq_file.avg_pc_data_info["idx_end"] / daq_file.base_time)
            if (daq_file.avg_pc_data_info["idx_end"] > 0)
            else 0
        )
        daq_file.avg_pc_data_info["num"] = avg_num
        daq_file.avg_pc_data_info["idx_start"] = start_idx
        daq_file.avg_pc_data_info["idx_end"] = end_idx

    def check_averaging_parameters(daq_file):
        """
        This function checks the number/index-based parameters contoined within
        the averaging_variables dictionary.

        Parameters
        ----------
        daq_file : igtools.DAQFile
            DAQ file object

        Returns
        -------
        averaging_variables : dictionary
            Returns original dictionary after checking and updating elements
        """
        if (daq_file.data is None) or (daq_file.base_time is None):
            raise Exception(
                "Data must be initialized prior to checking/validating "
                "averaging_variables!"
            )

        if (
            daq_file.avg_pc_data_info["idx_start"]
            > daq_file.avg_pc_data_info["idx_end"]
        ):
            raise Exception("Start index greater than end index!")

        daq_file.check_values()

    def check_values(daq_file):
        """
        This function checks the 'end' and 'totol' parameters of the
        specified data range and validates the value for compatibility.

        If the end value is not specified, it is set to the end of the data matrix.
        If there is a conflict between the end and total values, an exception is raised.
        If the total duration is longer than the available data duration, the total
        duration is set to the remaining length of the data matrix.

        Parameters
        ----------
        daq_file: DaqFile object

        Returns
        -------

        """

        start_val = daq_file.avg_pc_data_info["idx_start"]
        end_val = daq_file.avg_pc_data_info["idx_end"]
        avg_val = daq_file.avg_pc_data_info["num"]
        total_val = daq_file.avg_pc_data_info["total"]

        data_len = len(daq_file.data)

        # Case where we didn't specify an end or total value
        if (end_val == 0) & (total_val == 0):
            end_val = data_len
            if avg_val > 1:
                total_val = int(((end_val - start_val) / avg_val) + 1)
            else:
                total_val = int((end_val - start_val) / avg_val)

        # Case where we didn't specify an end value but did specify a total value
        elif (end_val == 0) & (total_val > 0):
            end_val = np.min([(start_val + (total_val * avg_val)), (data_len)])

            if (total_val * avg_val) > (end_val - start_val):
                # If the total value is longer than the available data, set the total
                # value to the remaining length of the data matrix.
                percent_requested = (
                    (end_val - start_val) / (total_val * avg_val)
                ) * 100
                total_val = ceil((end_val - start_val) / avg_val)

                warnings.warn(
                    f"Total value exceeds available data length. Setting data length to remaining data length ({percent_requested:.2f}% of requested total)."
                )

        # Case where we specified an end value but not a total value
        elif (end_val >= start_val) & (total_val == 0):
            total_val = ceil((end_val - start_val) / avg_val)

        # Case where we specified both an end and total value
        elif ((end_val >= start_val)) & (total_val > 0):
            try:
                compare_end_and_total(start_val, end_val, avg_val, total_val)
            except Exception as e:
                raise Exception(e)

        daq_file.avg_pc_data_info["idx_start"] = start_val
        daq_file.avg_pc_data_info["idx_end"] = end_val
        daq_file.avg_pc_data_info["total"] = total_val

    def update_pc_lims(daq_file, lim_low, lim_high):
        """
        This function changes the DaqFile attributes pc_lim_low and pc_lim_high to
        user defined values (from the values found in the .log file)
        Parameters
        ----------
        lim_low: float
            Lower limit for the phase correction window
        lim_high: float
            Upper limit for the phase correction window

        Returns
        -------

        """
        daq_file.pc_lim_low = lim_low
        daq_file.pc_lim_high = lim_high

    def update_pc_kwargs(daq_file, **kwargs):
        """
        This function updates the DaqFile pc_kwargs attribute to contain user defined
        parameters to pass to the phase correction function
        Parameters
        ----------
        kwargs: dictionary
            Dictionary of user-defined keyword arguments for the phase correction
            function

        Returns
        -------

        """
        for key, value in kwargs.items():
            daq_file.pc_kwargs[key] = value

    def update_pc_func(daq_file, func):
        """
        This function updates the configured phase correction function for the
        DaqFile object
        Parameters
        ----------
        func: function
            User-specified phase correction function

        Returns
        -------

        """
        daq_file.pc_type = func


class DAQFilesRP(DAQFilesVC707):
    def _finalize_init(daq_file):
        if daq_file.byte_order == "little":
            bo = "<"
        elif daq_file.byte_order == "big":
            bo = ">"
        else:
            raise ValueError("Unrecognized byte order %r" % daq_file.byte_order)
        if daq_file.data_raw is not None:
            daq_file.data_raw.dtype = daq_file.data_raw.dtype.newbyteorder(bo)
        if daq_file.data_pc is not None:
            daq_file.data_pc.dtype = daq_file.data_pc.dtype.newbyteorder(bo)
        daq_file.p2p_scale = 1 / (
            (pow(2, daq_file.CONSTANTS.daq_bits - 1) - 1) * daq_file.num_hwavgs
        )
        daq_file.raw_time = abs((daq_file.dfr**-1) / 2) * daq_file.num_hwavgs
        daq_info = daq_file.daq_info
        daq_file.avg_pc_data_info = None
        daq_file.data_raw_pc = None
        daq_file.base_time = None
        daq_file.data_len = None
        daq_file.x_wvn = None
        daq_file.pc_type = daq_file.pc_single_point
        daq_file.pc_kwargs = {}
        if daq_info is not None and "rp" in daq_info.lower():
            pass
        else:
            raise Exception

    def mobile_axis(daq_file, config_variables, fr2_set=None, df_tooth=None):
        """
        Calculate frequency axis for daq_file-referenced dual-comb.
        Based on setpoint and dfr rather than instantaneous fr1, fr2 which shift by 5 Hz

        f_opt = optical lock frequency and sign (MHz). Where f_CW = f_tooth + f_opt
        wvn_spectroscopy = one frequency inside filtered spectra (cm-1)

        IG is an object from pldspectrapy/igtools.py with all the log file information from comb locking
        IG.fc = repetition rate setpoint of clock comb (sets DAQ clock)
        IG.fr2 = repetition rate of unclocked comb
        IG.frame_length = points per interferogram. If interferogram centerburst is not moving,
                                                    this tells us 1/df_rep
        """

        # Calculate CW laser frequency from individual comb f_reps, IG length, optical beat offset

        f_opt = config_variables["fitting"]["lock_frequency"]
        band_fit = config_variables["fitting"]["band_fit"]
        wvn_spectroscopy = np.mean(band_fit)  # Figure out which nyquist window to use

        f_cw_approx = 191.5e12  # approximate CW reference laser frequency (Hz)

        mod_fc_setpoint = daq_file.fc_setpoint  # * 2
        mod_dfr = daq_file.dfr  # * 2

        if fr2_set is None or df_tooth is None:
            fr2_set = mod_fc_setpoint - mod_dfr
            df_tooth = 0.5 * (mod_fc_setpoint + fr2_set)

        df_nyq = 2 * daq_file.frame_length * fr2_set
        nyq_cw = np.argmin(np.abs(f_cw_approx - df_nyq * np.asarray(range(20))))
        f_cw = df_nyq * nyq_cw + f_opt

        # determine Nyquist window of spectroscopy
        f_spectroscopy = wvn_spectroscopy * SPEED_OF_LIGHT * M_2_CM
        nyq, sign = divmod(f_spectroscopy, df_nyq)
        if sign / df_nyq < 0.5:
            flip_spectrum = False
        else:
            nyq += 1
            flip_spectrum = True

        # Last make frequency axis array
        # x_hz = np.arange(IG.frame_length / 2 + 1)
        x_hz = np.arange(daq_file.frame_length + 1)
        if flip_spectrum:
            x_hz *= -1
        x_hz = x_hz * df_tooth
        x_hz += nyq * df_nyq - f_opt
        daq_file.x_wvn = x_hz / SPEED_OF_LIGHT / M_2_CM

        daq_file.x_wvn = daq_file.x_wvn[0 : int((len(daq_file.x_wvn) + 1) / 2)]

        return daq_file.x_wvn


def open_daq_files(filename):
    # TODO: should we move it to be a class method of DAQFiles that all of the types below inherit?

    """
    Open a DAQ file and return the appropriate DAQFiles object
    Parameters
    ----------
    filename: str
        The filename of the DAQ file to open

    Returns
    -------
    daq_files: DAQFiles
        The DAQFiles object for the opened file

    """

    err_str = "\nAttempt to open %s as DAQFiles gave following exceptions\n" % filename
    types = (DAQFilesVC707, DAQFilesNI, DAQFilesRP)
    for t in types:
        try:
            # BUG: produces ValueError when file is not found
            return t(filename)
        except:  # noqa: E722
            err_str += "\nException initializing as %s:" % t.__name__
            err_str += format_exc().lstrip("Traceback (most recent call last):")
    raise ValueError(err_str.replace("\n", "\n\t"))


def open_dir(dirname, recursion_levels=0, verbose=False):
    fnames = {}
    recursion_levels -= 1
    for fname in os.listdir(dirname):
        full_file = os.path.join(dirname, fname)
        if os.path.isdir(full_file):
            if recursion_levels >= 0:
                fnames.update(open_dir(full_file, recursion_levels, verbose))
            continue
        if not is_populated_log(full_file):
            continue
        fname = remove_extension(fname)
        if fname in fnames:
            continue
        try:
            daq_files = open_daq_files(os.path.join(dirname, fname))
        except ValueError:
            if verbose > 1:
                raise
            if verbose:
                print_exc()
        else:
            fnames[fname] = daq_files
    return fnames


###############################################################################
# TODO: nothing below this has been fixed since 2020-04-22 rewrite


class DCSDataLoaded(DCSData):
    # TODO: set start, stop , step to reflect which igs from raw data are being used?
    num_pcs = 1
    start = None
    stop = None
    step = None

    def __init__(daq_file, data_source, **kwargs):
        for setting in DCSData.DEFAULTS:
            if setting in (
                "p2p_min",
                "p2p_max",
                "num_pcs",
                "byte_order",
                "start",
                "stop",
                "step",
            ):
                continue
            if setting in ("pc_lim_low", "pc_lim_high"):
                if setting in kwargs:
                    continue
            kwargs[setting] = data_source.settings[setting]
        kwargs["byte_order"] = sys.byteorder
        super(DCSDataLoaded, daq_file).__init__(**kwargs)

        if daq_file.pc_lim_low < 0:
            daq_file.pc_lim_low = 0.0
        elif daq_file.pc_lim_low > 0.5:
            daq_file.pc_lim_low = 0.5
        if daq_file.pc_lim_high > 0.5:
            daq_file.pc_lim_high = 0.5
        elif daq_file.pc_lim_high < 0.0:
            daq_file.pc_lim_high = 0.0
        if daq_file.pc_lim_high < daq_file.pc_lim_low:
            t = daq_file.pc_lim_high
            daq_file.pc_lim_high = daq_file.pc_lim_low
            daq_file.pc_lim_low = t

        s = slice(daq_file.start, daq_file.stop, daq_file.step)
        daq_file.start, daq_file.stop, daq_file.step = s.indices(len(data_source))
        data = data_source[s]
        num_pcs = daq_file.num_pcs
        if num_pcs > len(data):
            num_pcs = len(data)
        elif num_pcs < 1:
            num_pcs = 1
        daq_file.num_pcs = data_source.num_pcs * num_pcs
        daq_file._load_data(data, data_source.num_igs_per[s], num_pcs)

    def _load_data(daq_file, data, input_igs_per, num_pcs):
        raise NotImplementedError


class Spectra(DCSDataLoaded):
    def _load_data(daq_file, data, input_igs_per, num_pcs):
        """
        Phase correction happens here
        TODO: assumes data is IG, but should be able to accept spectrum
        """

        is_input_complex = np.iscomplexobj(data)
        num_spectra = ceildiv(len(data), num_pcs)
        len_spectrum = daq_file.frame_length // 2 + 1  # based on rfft result
        daq_file.spectra = np.empty((num_spectra, len_spectrum), dtype=np.complex128)
        # Have to make this 2d in order to iterate over rows
        daq_file.num_igs_per = np.zeros((num_spectra, 1), dtype=np.uint32)
        daq_file.p2p_total = 0

        pc_start_pt = int(round(daq_file.pc_lim_low * len_spectrum * 2))
        pc_end_pt = int(round(daq_file.pc_lim_high * len_spectrum * 2))

        iter_data = iter(data)
        iter_num = iter(input_igs_per)

        try:
            for spectrum, num_igs in zip(daq_file.spectra, daq_file.num_igs_per):
                for i in range(num_pcs):
                    ig = next(iter_data)
                    num_igs_in = next(iter_num)
                    p2p = np.max(ig.real)
                    if p2p < 0:
                        p2p = 0
                    else:
                        p2p *= 2 * daq_file.p2p_scale
                    if p2p < daq_file.p2p_min or p2p > daq_file.p2p_max:
                        continue
                    if is_input_complex:
                        real = ig.real
                        imag = ig.imag
                        p2p = (
                            np.sqrt(np.max(real * real + imag * imag))
                            * 2
                            * daq_file.p2p_scale
                        )
                        if p2p < daq_file.p2p_min or p2p > daq_file.p2p_max:
                            continue
                        fft = np.fft.fft(ig)[:len_spectrum]
                    else:
                        p2p = np.max(ig.real)
                        if p2p < 0:
                            p2p = 0
                        else:
                            p2p *= 2 * daq_file.p2p_scale
                        if p2p < daq_file.p2p_min or p2p > daq_file.p2p_max:
                            continue
                        fft = np.fft.rfft(ig)
                    if num_igs == 0:
                        if num_pcs > 1:
                            if pc_start_pt == pc_end_pt:
                                phase_ref_index = pc_start_pt
                            else:
                                real = fft.real[pc_start_pt:pc_end_pt]
                                imag = fft.imag[pc_start_pt:pc_end_pt]
                                mag2 = (real * real) + (imag * imag)
                                phase_ref_index = np.argmax(mag2) + pc_start_pt
                            phase_ref = np.angle(fft[phase_ref_index])
                        spectrum[:] = fft
                    else:
                        dphase = np.angle(fft[phase_ref_index]) - phase_ref
                        spectrum += fft * np.exp(-1j * dphase)
                    num_igs += num_igs_in
                    daq_file.p2p_total += p2p
                if num_igs == 0:
                    spectrum[:] = 0
        except StopIteration:
            pass
        # Turn this back to 1d array
        daq_file.num_igs_per.shape = (num_spectra,)
        daq_file.data = daq_file.spectra

    def fliplr(daq_file):
        daq_file.data = np.fliplr(daq_file.data)

    def phase_average_further(daq_file):
        """
        After first calling Spectra object and producing complex phase-averaged FFTs,
        further phase-correct FFTs
        Same phase-correction algorithm as above, except start w/ ffts

        TODO: Nate wrote this code and it doesn't work well on his data.
        """
        len_spectrum = daq_file.frame_length // 2 + 1  # based on rfft result
        pc_start_pt = int(round(daq_file.pc_lim_low * len_spectrum * 2))
        pc_end_pt = int(round(daq_file.pc_lim_high * len_spectrum * 2))
        spectrum_out = daq_file.data[0]
        try:
            for spectrum, num_igs in daq_file.data:
                if num_igs == 0:
                    if num_pcs > 1:
                        if pc_start_pt == pc_end_pt:
                            phase_ref_index = pc_start_pt
                        else:
                            mag2 = np.abs(spectrum[pc_start_pt:pc_end_pt])
                            phase_ref_index = np.argmax(mag2) + pc_start_pt
                        phase_ref = np.angle(spectrum[phase_ref_index])
                    spectrum_out[:] = spectrum
                else:
                    dphase = np.angle(spectrum[phase_ref_index]) - phase_ref
                    spectrum_out += spectrum * np.exp(-1j * dphase)

        except StopIteration:
            pass

        daq_file.data = spectrum_out


class IGs(DCSDataLoaded):
    def _load_data(daq_file, data, input_igs_per, num_pcs):
        """
        Phase correction happens here (TODO: currently only coadd)
        TODO: assumes data is IG, but should be able to accept spectrum
        """
        num_igs_coor = ceildiv(len(data), num_pcs)
        daq_file.igs = np.zeros((num_igs_coor, daq_file.frame_length), dtype=np.float64)
        # Have to make this 2d in order to iterate over rows
        daq_file.num_igs_per = np.zeros((num_igs_coor, 1), dtype=np.uint32)

        iter_data = iter(data)
        iter_num = iter(input_igs_per)

        try:
            for ig, num_igs in zip(daq_file.igs, daq_file.num_igs_per):
                for i in range(num_pcs):
                    ig_in = next(iter_data)
                    num_igs_in = next(iter_num)
                    p2p = np.max(ig_in.real)
                    if p2p < 0:
                        p2p = 0
                    else:
                        p2p *= 2 * daq_file.p2p_scale
                    if p2p < daq_file.p2p_min or p2p > daq_file.p2p_max:
                        continue
                    if num_igs == 0:
                        if num_pcs > 1:
                            pass
                        ig[:] = ig_in
                    else:
                        ig += ig_in
                    num_igs += num_igs_in

                if num_igs == 0:
                    ig[:] = 0
        except StopIteration:
            pass
        # Turn this back to 1d array
        daq_file.num_igs_per.shape = (num_igs_coor,)
        daq_file.data = daq_file.igs


def pc_crosscorrelate(igs, sample_width=2**11):
    sample_width = pow(2, int(round(np.log2(sample_width))))
    i_center = None
    igc = np.zeros(len(igs[0]), dtype=np.complex128)
    x_range = np.arange(2)
    for ig in igs:
        # ig = ig - np.mean(ig)
        if i_center is None:
            i_center = np.argmax(ig)
            start = i_center - sample_width // 2
            stop = start + sample_width
            centerburst = ig[start:stop]
            # make ig_center 1 point longer so result of convolve will be 2^n in length
            stop += 1
        ig_center = ig[start:stop]
        xcreal = np.correlate(centerburst, ig_center, "full")
        xc = hilbert(xcreal)
        xcmag = abs(xc)
        i_max = np.argmax(xcmag)

        fit = np.polyfit(range(i_max - 1, i_max + 2), xcmag[i_max - 1 : i_max + 2], 2)
        delay = fit[1] / (-2 * fit[0])
        delay0 = int(delay)
        delay2 = delay0 + 2
        phase = np.interp(delay % 1, x_range, np.angle(xc[delay0:delay2]))
        igc += ig * np.exp(1j * phase)
    return igc


def pc_hilbert(igs, sample_width=2**8):
    sample_width = pow(2, int(round(np.log2(sample_width))))
    i_center = None
    igc = np.zeros(len(igs[0]), dtype=np.complex128)
    x_range = np.arange(2)
    for ig in igs:
        # ig = ig - np.mean(ig)
        if i_center is None:
            i_center = np.argmax(ig)
            start = i_center - sample_width // 2
            stop = start + sample_width
        h = hilbert(ig[start:stop])
        hmag = abs(h)
        i_max = np.argmax(hmag)

        fit = np.polyfit(range(i_max - 1, i_max + 2), xcmag[i_max - 1 : i_max + 2], 2)
        delay = fit[1] / (-2 * fit[0])
        delay0 = int(delay)
        delay2 = delay0 + 2
        phase = np.interp(delay % 1, x_range, np.angle(xc[delay0:delay2]))
        igc += ig * np.exp(1j * phase)
    return igc


def angle(z, axis=-1):
    return np.unwrap(np.angle(z), axis=axis)


def find_cb(data_pc, pc_lim_low, pc_lim_high):
    """
    This function finds the max points in the centerbursts

    Args:
        daq_file (object): Object with data from the daq file (.cor)

    Returns:
        indices (list): List of indices for the max points

    """
    length = 2**12
    cut = 0.2
    cb_locs = []
    weights = []

    cb_loc = np.argmax(data_pc, axis=-1)
    # cb_loc = the location of the max point in the centerburst
    cb_locs.append(cb_loc)

    w = [np.abs(data_pc[i, j]) for i, j in enumerate(cb_loc)]
    # The relative strength of the IGs in the file, as calc'd by their size
    weights.append(np.divide(w, sum(w)))

    if any(length > cb_loc for cb_loc in cb_locs[0]):
        cbs = [i for i in data_pc]

    else:
        cbs = [
            data_pc[i, start : start + length]
            for i, start in enumerate(cb_loc - length // 2)
        ]

    # ifftshift takes 0 frequency placed at the center of the array and moves to to the
    # 0th position! (NOT fftshift, although behavior for even len array is same)
    cbs = np.fft.ifftshift(cbs, axes=-1)

    center = (pc_lim_low + pc_lim_high) / 2
    w = (pc_lim_high - pc_lim_low) * cut
    start = int(round((center - w / 2) * length))
    stop = int(round((center + w / 2) * length))
    F = np.fft.fft(cbs, axis=-1)
    a = angle(F[:, start:stop], axis=-1)
    slopes = np.polyfit(np.arange(stop - start), np.transpose(a), 1)[0]
    p = length * slopes / (2 * np.pi)

    indices = (cb_loc - p).astype(int)

    return indices


def slide_centerburst(data_pc, centerbursts):
    """
    This function situates a value in the middle of an array, and rotates
    around it

    Args:
        data_pc (np.array): Array with data that needs to be rotated

        centerbursts (list): Indices of the maximum values for the array of
        data, (the value corresponding to that index is placed in the middle of the array)

    Returns:
        data_pc_slided (np.array): Array with the data rotated

    """
    data_pc_slided = []
    for i, cb in enumerate(centerbursts):
        to_rotate = abs(cb - len(data_pc[i]) // 2)
        if cb > len(data_pc[i] // 2):
            to_rotate = -to_rotate

        data_pc_slided.append(np.roll(data_pc[i], to_rotate))

    data_pc_slided = np.array(data_pc_slided)
    return data_pc_slided


def check_spread_centerbursts(centerbursts):
    """
    This function checks if the centerbursts in a measurement are too different
    by calculating the abs differences between the elements in the array (indices)

    Args:
        centerbursts (list): Indices of the maximum values for the array of
        data

    Returns:
        ret (bool): True if centerburst are too spread

    """
    CB_DIFF_THRESHOLD = 25
    # List with differences between all elements of the list
    diffs = [abs(i - j) for i in centerbursts for j in centerbursts if i != j]
    ret = any(CB_DIFF_THRESHOLD < i for i in diffs)
    return ret


def avg_igs_by_idx(data, idxs):
    """
    Generic function to average (sum) data based on indices

    Parameters
    ----------
    data: numpy array
        The input array containing all the IGs for data processing
    idxs: list
        List of indices indicating how to slice the data array for averaging
    Returns
    -------
    avg_data: averaged (summed) data
    """

    data_subset = data[idxs[0] : idxs[1] + 1]
    # TODO: If this is super slow, explore using dask
    avg_data = np.sum(data_subset, 0)

    return avg_data


def convert_idx_to_time(base_time, indices):
    """
    This function converts a list of index values to a time-based array using the
    base-time variable

    Parameters
    ----------
    base_time: float
        Amount of time represented by individual IGs
    indices: list
        List of indices to be converted to a time-base

    Returns
    -------
    time_indices: array of time-values corresponding to the input indices; same
    length as idxs input
    """
    time_indices = []
    for idx, idxs in enumerate(indices):
        start_idx = idxs[0]
        end_idx = idxs[1]
        start_time = timedelta(seconds=(base_time * start_idx)).total_seconds()
        end_time = timedelta(
            seconds=((base_time * end_idx) + base_time)
        ).total_seconds()
        time_indices.append([start_time, end_time])
    return time_indices


def return_p2p(igs, idx):
    """
    Return the peak-to-peak value for a given interferogram

    Parameters
    ----------
    igs: numpy.ndarray
        2D array of interferograms. (Comes from daq_file.data_pc

    idx: int
        Index of the interferogram to be analyzed

    Returns
    -------
    p2p: float
        Peak-to-peak value of the interferogram
    """

    adc_scale = 2**-16
    width = 128
    ig = igs[idx, :]
    argmax = ig.argmax()
    mx = ig[argmax]
    cb = ig[int(argmax - (width / 2)) : int(argmax + (width / 2))]
    mn = cb.min()
    p2p = (mx - mn) * adc_scale / 50
    return p2p


def pc_truncated_RP(igs, pc_lim_low=0.1, pc_lim_high=0.4, sample_width=2**11):
    sample_width = pow(2, int(np.log2(sample_width)))

    ref_ig = 0
    ref_max = 0
    for index, ig in enumerate(igs):
        max_ = np.amax(ig)
        if max_ > ref_max:
            ref_max = max_
            ref_ig = index

    ig0 = igs[ref_ig]
    is_input_complex = np.iscomplexobj(ig0)
    center = np.argmax(ig0)
    start = center - sample_width // 2
    if start < 0:
        start = 0
    stop = start + sample_width
    igc = np.array(ig0, dtype=np.complex128)

    if is_input_complex:
        s = np.fft.fft(ig0[start:stop])
        pc_start = int(round(pc_lim_low * len(s)))
        pc_stop = int(round(pc_lim_high * len(s)))
    else:
        s = np.fft.rfft(ig0[start:stop])
        len_2 = 2 * len(s)
        pc_start = int(round(pc_lim_low * len_2))
        pc_stop = int(round(pc_lim_high * len_2))
    r = s.real[pc_start:pc_stop]
    i = s.imag[pc_start:pc_stop]
    mag = r * r + i * i
    ref_index = pc_start + np.argmax(mag)
    phase_ref = np.angle(s[ref_index])

    for ig in igs[1:]:
        if is_input_complex:
            s = np.fft.fft(ig[start:stop])
        else:
            s = np.fft.rfft(ig[start:stop])
        dphase = np.angle(s[ref_index]) - phase_ref
        igc += ig * np.exp(-1j * dphase)
    return igc


def pc_truncated_vc707(
    igs, pc_lim_low=0.1, pc_lim_high=0.4, sample_width=2**11, slide_flag=False
):
    cbs = find_cb(igs, pc_lim_low, pc_lim_high)

    # If the centerbursts are are too spread we go process the next file
    # if check_spread_centerbursts(cbs):
    #     raise Exception("The centerbursts are too spread")

    # Rotate data around centerburst when flag is true
    if slide_flag:
        igs = slide_centerburst(igs, cbs)

    sample_width = pow(2, int(np.log2(sample_width)))
    ig0 = igs[0]
    is_input_complex = np.iscomplexobj(ig0)
    center = np.argmax(ig0)
    start = center - sample_width // 2
    if start < 0:
        start = 0
    stop = start + sample_width
    igc = np.array(ig0, dtype=np.complex128)

    if is_input_complex:
        s = np.fft.fft(ig0[start:stop])
        pc_start = int(round(pc_lim_low * len(s)))
        pc_stop = int(round(pc_lim_high * len(s)))
    else:
        s = np.fft.rfft(ig0[start:stop])
        len_2 = 2 * len(s)
        pc_start = int(round(pc_lim_low * len_2))
        pc_stop = int(round(pc_lim_high * len_2))
    r = s.real[pc_start:pc_stop]
    i = s.imag[pc_start:pc_stop]
    mag = r * r + i * i
    ref_index = pc_start + np.argmax(mag)
    phase_ref = np.angle(s[ref_index])

    for ig in igs[1:]:
        if is_input_complex:
            s = np.fft.fft(ig[start:stop])
        else:
            s = np.fft.rfft(ig[start:stop])
        dphase = np.angle(s[ref_index]) - phase_ref
        igc += ig * np.exp(-1j * dphase)

    return igc


def pc_nearest_2(igs, pc_lim_low=0.1, pc_lim_high=0.4):
    ig0 = igs[0]
    ig_width = pow(2, int(np.log2(len(ig0))))
    spectrum_width = ig_width // 2 + 1  # based on output of rfft
    is_input_complex = np.iscomplexobj(ig0)
    center = np.argmax(ig0)
    start = center - ig_width // 2
    stop = start + ig_width

    if is_input_complex:
        sc = np.fft.fft(ig0[start:stop])[:spectrum_width]
    else:
        sc = np.fft.rfft(ig0[start:stop])
    len_2 = 2 * len(sc)
    pc_start = int(round(pc_lim_low * len_2))
    pc_stop = int(round(pc_lim_high * len_2))
    r = sc.real[pc_start:pc_stop]
    i = sc.imag[pc_start:pc_stop]
    mag = r * r + i * i
    ref_index = pc_start + np.argmax(mag)
    phase_ref = np.angle(sc[ref_index])

    for ig in igs[1:]:
        if is_input_complex:
            s = np.fft.fft(ig[start:stop])[:spectrum_width]
        else:
            s = np.fft.rfft(ig[start:stop])
        dphase = np.angle(s[ref_index]) - phase_ref
        sc += s * np.exp(-1j * dphase)

    return sc


def get_walking_rate2(data_source, igs_to_use=1000, plot=False):
    """
    INPUTS:
        data_source: DCSData object to calc values on
        igs_to_use = the max number of igs to use in the ref laser calc
        plot = boolean, whether to plot the ig walking results

    OUTPUTS:
        walk_rate = the number of points the peak of IG moves each
            delta fc time step

    ERRORS:
        None
    """
    if igs_to_use > len(data_source):
        igs_to_use = len(data_source)

    # Have to make this 2d in order to iterate over rows
    maxes = np.empty((igs_to_use, 1), dtype=np.uint32)
    for max, ig in zip(maxes, data_source.data[:igs_to_use]):
        max[0] = np.argmax(ig.real)
        # Turn this back to 1d array
    maxes.shape = (igs_to_use,)
    maxes = unwrap(maxes, data_source.frame_length, 0.95)

    x = tuple(range(igs_to_use))
    linefit = force_fit_line(x, maxes)

    num_igs_avgd = data_source.num_pcs * data_source.num_hwavgs
    walk_rate = linefit[0] / num_igs_avgd

    if plot:
        line = np.poly1d(linefit)
        plt.figure()
        plt.plot(x, maxes, x, line(x))

    return walk_rate


def get_walking_rate(data_source, igs_to_use=1000, plot=False):
    """
    INPUTS:
        data_source: DCSData object to calc values on
        igs_to_use = the max number of igs to use in the ref laser calc
        plot = boolean, whether to plot the ig walking results

    OUTPUTS:
        walk_rate = the number of points the peak of IG moves each
            delta fc time step

    ERRORS:
        None
    """
    if igs_to_use > len(data_source):
        igs_to_use = len(data_source)

    data = data_source.data[:igs_to_use].real
    fl = data_source.frame_length
    dtype = data.dtype.newbyteorder("N")

    means = np.mean(data, axis=1)
    maxlocs = np.argmax(data, axis=1)
    maxes = np.empty(igs_to_use, dtype=dtype)
    for i, (d, maxloc) in enumerate(zip(data, maxlocs)):
        maxes[i] = d[maxloc]
    # Move halfway around the ig from the max and assume it is all noise
    # at this point
    noiselocs = (maxlocs + fl // 2) % fl
    # Unwrap the maxlocs to have walk be a continuous line
    maxlocs = unwrap(maxlocs, fl, 0.95)

    # Take noise_pts num of points around each noiseloc and find the max;
    # call this the noise level
    noise_pts = fl // 100
    start_largest = fl - noise_pts
    starts = noiselocs - noise_pts // 2
    stops = starts + noise_pts
    noise = np.empty(igs_to_use, dtype=dtype)
    for i, (d, start, stop) in enumerate(zip(data, starts, stops)):
        if start < 0:
            start = 0
            stop = noise_pts
        elif stop > fl:
            start = start_largest
            stop = fl
        noise[i] = np.amax(d[start:stop])

    # sf = signal factor, is a measure of how noisy the spectrum is
    sf = (maxes - means) / (noise - means) - 1

    # Begin fitting lines; first one weight by the noise factor
    x = tuple(range(igs_to_use))
    linefit = np.polyfit(x, maxlocs, 1, w=pow(sf, 2.5))
    line1 = np.poly1d(linefit)

    # Second line weighting is binary based on residual_td thresholding
    residuals = np.abs(line1(x) - maxlocs)
    thresh = 0.05
    thresh_value = thresh * np.amax(residuals)
    weights = (residuals < thresh_value).astype(int)
    linefit = np.polyfit(x, maxlocs, 1, w=weights)
    line2 = np.poly1d(linefit)

    if plot:
        plt.figure()
        plt.plot(x, maxlocs, x, line1(x), x, line2(x))
        plt.legend(("raw", "signal fit_td", "final fit_td"))

    num_igs_avgd = data_source.num_pcs * data_source.num_hwavgs
    walk_rate = linefit[0] / num_igs_avgd
    return walk_rate


def estimate_ref_laser(data_source, igs_to_use=1000, plot=False):
    """
    INPUTS:
        data_source: DCSData object to calc values on
        igs_to_use = the max number of igs to use in the ref laser calc
        plot = boolean, whether to plot the ig walking results

    OUTPUTS:
        laser_wl = the estimated wavelength of the reference laser used
            in generating these IGs

    ERRORS:
        None
    """
    walk_rate = get_walking_rate(data_source, igs_to_use, plot)
    frame_length_true = data_source.frame_length + walk_rate
    if data_source.dfr < 0:  # clocking on faster comb
        dfNYQ = (frame_length_true - 1) * data_source.fc
    else:  # clocking on slower comb
        dfNYQ = (frame_length_true + 1) * data_source.fc

    # Estimate which nyqueist window the laser is locked to
    laser_wl_est = 1565.0  # nm
    laser_f_est = 1e9 * c / laser_wl_est  # hz
    NYQref_estimate = int(round(laser_f_est / dfNYQ))

    laser_wl = 1e9 * c / (dfNYQ * NYQref_estimate)
    return laser_wl


def organize_daq_files(
    input_dir,
    output_dir,
    dirfmt="%Y%m%d",
    filefmt="%Y%m%d%H%M%S",
    time_thresh=1000,
    use_utc=False,
    mode=FileModes.MOVE,
):
    """
    Organize the DAQ files into a logical structure of subdirectories based on date

    INPUTS:
        input_dir   = string of DAQ log directory, does not need ending '\'
        output_dir  = string of directory to move everything to
        dirfmt      = string of subdirectory format to reorganize things
        filefmt     = string of datetime format stored in DAQ file to parse
        time_thresh = timeout time to find files, [sec]
        mode        = int of what to do with the files, defined in top of file

    OUTPUTS:
        None
    """
    if use_utc:
        raise NotImplementedError
    # Create directory strings
    input_dir = os.path.realpath(input_dir)
    output_dir = os.path.realpath(output_dir)
    # Look for the input files
    try:
        input_files = os.listdir(input_dir)
    except WindowsError:  # TODO: make os independant
        print("Do not have access to input directory '%s'" % input_dir)
        return
    # Initialize log list, and start timing for information and debug
    now = time.time()
    daq_logs = []

    # Look for log files in input_dir
    # Doing 2 interations is not most efficient, but allows you to determine
    # number of files that must be moved before heavy lifting
    for f in input_files:
        ext = f[-(len(EXT_LOG_FILE) + 1) :].lower()
        if ext != ("." + EXT_LOG_FILE) and ext != ("_" + EXT_LOG_FILE):
            continue
        if now - os.path.getmtime(os.path.join(input_dir, f)) < time_thresh:
            continue
        daq_logs.append(f)

    num_daq_files = len(daq_logs)
    if not num_daq_files:
        print("No DAQ file sets found to organize in '%s'" % input_dir)
        return
    print("Found %i DAQ file sets; begin organization" % num_daq_files)
    percent_factor = 100.0 / num_daq_files

    # Rearrange DAQ files into logical directories
    for index, daq_log in enumerate(daq_logs):
        if index:
            time_elapsed = (time.time() - now) / 60  # minutes
            time_total = time_elapsed / index * num_daq_files
            time_remaining = "%0.2f" % (time_total - time_elapsed)
        else:
            time_remaining = "N/A"
        print(
            "%05.2f%% complete, %s minutes remaining"
            % (index * percent_factor, time_remaining)
        )
        print("   Processing " + os.path.join(input_dir, daq_log))
        # Get DAQFilesNI object for this particular daq_log, which contains the open fileid, and close it
        daq_files = open_daq_files(os.path.join(input_dir, daq_log))
        daq_files.close()
        # Skip the file if it's open somewhere else
        if daq_files.is_open_elsewhere():
            continue
        # Don't worry about empty files
        if not daq_files.has_data:
            if mode == FileModes.MOVE or mode == FileModes.ZIPMOVE:
                print("   has no data; remove files")
                for f in (
                    daq_files.file_log,
                    daq_files.data_raw_source,
                    daq_files.data_pc_source,
                    daq_files.file_pcp,
                ):
                    if f is not None:
                        if isinstance(f, basestring):
                            os.remove(f)
                        else:
                            os.remove(f.name)
            else:
                print("   has no data; skip")
            continue

        # Grab log info for generating subdirectories and create them
        start_time = daq_files.start_time
        save_path = os.path.join(output_dir, start_time.strftime(dirfmt))
        save_name = start_time.strftime(filefmt)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        # COPY or MOVE the file
        if mode == FileModes.COPY or mode == FileModes.MOVE:
            save_name = os.path.join(save_path, save_name)
            for f, ext in zip(
                (
                    daq_files.file_log,
                    daq_files.data_raw_source,
                    daq_files.data_pc_source,
                    daq_files.file_pcp,
                ),
                EXTS,
            ):
                if not f:  # File not included in this set of DAQ Files
                    continue
                if isinstance(f, basestring):
                    fname = os.path.realpath(f)
                else:
                    fname = os.path.realpath(f.name)
                fname_new = save_name + "." + ext
                if mode == FileModes.COPY:
                    copy(fname, fname_new)
                else:
                    os.rename(fname, fname_new)
        # Do the same, only perform operation on the .zip
        elif mode == FileModes.ZIPCOPY or mode == FileModes.ZIPMOVE:
            with ZipFile(
                os.path.join(save_path, save_name + ".zip"),
                "a",
                ZIP_DEFLATED,
                True,
            ) as zfile:
                for f, ext in zip(
                    (
                        daq_files.file_log,
                        daq_files.file_raw,
                        daq_files.file_pc,
                        daq_files.file_pcp,
                    ),
                    EXTS,
                ):
                    if not f:  # File not included in this set of DAQ Files
                        continue
                    if isinstance(f, basestring):
                        fname = os.path.realpath(f)
                    else:
                        fname = os.path.realpath(f.name)
                    zfile.write(fname, save_name + "." + ext)
                    if mode == FileModes.ZIPMOVE:
                        os.remove(fname)
    print("100.0% complete")


def organize_subdirectories(
    base_dir,
    output_dir,
    dirfmt="%Y%m%d",
    filefmt="%Y%m%d%H%M%S",
    time_thresh=1000,
    use_utc=False,
    mode=FileModes.MOVE,
):
    """
    Organize the subdirectories of DAQ files into the structure defined in "organize_daq_files"

    INPUTS:
        base_dir    = string of the root directory of the directory tree to reorganize
        output_dir  = string of the root directory for the output
        dirfmt      = string of subdirectory format to reorganize things
        filefmt     = string of datetime format stored in DAQ file to parse
        time_thresh = timeout time to find files, [sec]
        mode        = int of what to do with the files, defined in top of file

    OUTPUTS:
        None

    ERRORS:
        None
    """
    for folder in os.listdir(base_dir):
        folder = os.path.join(base_dir, folder)
        if not os.path.isdir(folder):
            continue
        organize_daq_files(
            folder, output_dir, dirfmt, filefmt, time_thresh, use_utc, mode
        )


if __name__ == "__main__":
    d = r"D:\spectro"
    t0 = time.time()
    files = open_dir(d, verbose=2)
    dt = time.time() - t0
    print("Opened %i files in %0.3f seconds" % (len(files), dt))
    newest = ""
    for name, daq_file in files.items():
        if not daq_file.failure:
            non_empty = sum(bool(i) for i in daq_file.igs_per_pc)
            if len(daq_file.data_pc) != non_empty:
                newest = max(newest, name)
