#!/usr/bin/env python

from optparse import OptionParser
import os
import subprocess
import sys
import re
import shutil
import glob
import datetime
import yaml
import common

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"

def extract_version( exec_path  ):
        objdump_out_filename = this_directory + "so_objdump_out.{0}.txt".format(os.getpid())
        objdump_out_file = open(objdump_out_filename, 'w+')
        subprocess.call(["strings", exec_path], stdout=objdump_out_file)
        objdump_out_file.seek(0)
        returnStr = re.sub( r".*(gpgpu-sim_git-commit[^\s]+).*", r"\1", objdump_out_file.read().strip().replace("\n", " ")  )
        objdump_out_file.close()
        os.remove(objdump_out_filename)
        return returnStr

#-----------------------------
# main script start
#-----------------------------
(optionsm args) = common.parse_run_simulation_options()

cuda_version = common.get_cuda_version( this_directory  )
options.run_directory = os.path.join(this_directory, "../../sim_run_%s"%cuda_version)

version_string = extract_version( so_path  )
print(version_string)
running_so_dir = os.path.join( options.run_directory, "gpgpu-sim-builds", version_string  )

if not os.path.exists( running_so_dir  ):
    # In the very rare case that concurrent builds try to make the directory at the same time
    # (after the test to os.path.exists -- this has actually happened...)
    try:
        os.makedirs( running_so_dir  )
    except:
        pass

# Don't need to copy so_path to runnning_so_dir : Line#277

options.so_dir = os.path.join(os.get.env("SST_CORE_HOME"), "bin")

common.load_defined_yamls()

# Test for the existance of torque on the system
if not any([os.path.isfile(os.path.join(p, "qsub")) for p in os.getenv("PATH").split(os.pathsep)]):
    exit("ERROR - Cannot find qsub in PATH... Is torque installed on this machine?")

if not any([os.path.isfile(os.path.join(p, "nvcc")) for p in os.getenv("PATH").split(os.pathsep)]):
    exit("ERROR - Cannot find nvcc PATH... Is CUDA_INSTALL_PATH/bin in the system PATH?")


benchmarks = []
benchmarks = common.gen_apps_from_suite_list(options.benchmark_list.split(","))

cfgs = common.gen_configs_from_list( options.configs_list.split(",")  )
configurations = []
for config in cfgs:
    configurations.append( ConfigurationSpec( config  )  )

print("Running Simulations with GPGPU-Sim built from \n{0}\n ".format(version_string) +
      "\nUsing configs: " + options.configs_list +
      "\nBenchmark: " + options.benchmark_list)

for config in configurations:
    config.my_print()
    config.run(version_string, benchmarks, options.run_directory, cuda_version, options.so_dir)

