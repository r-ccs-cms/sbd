"""MPI launcher shared by the shell and Python CLI tests."""
import os
import shlex
import subprocess
import sys


def flags(name):
    # CMake lists use semicolons; standalone Make callers may use shell words.
    value = os.environ.get(name, "")
    return value.split(";") if ";" in value else shlex.split(value)


def mpi_command(ranks, executable):
    return ([os.environ.get("MPIEXEC", "mpirun"),
             os.environ.get("MPIEXEC_NUMPROC_FLAG", "-np"), str(ranks)]
            + flags("MPIEXEC_PREFLAGS") + [executable]
            + flags("MPIEXEC_POSTFLAGS"))


if __name__ == "__main__":
    result = subprocess.run(mpi_command(sys.argv[1], sys.argv[2]) + sys.argv[3:],
                            timeout=60)
    sys.exit(result.returncode)
