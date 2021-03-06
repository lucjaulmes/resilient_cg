# Compiling

To compile, be sure to have in your path
- a C11 able compiler (e.g. gcc-4.9)
- the mercurium compiler for OmpSs
and to have installed
- nanos
- extrae
- papi

See [BSC’s OmpSs Download page](https://pm.bsc.es/ompss-downloads) if you are unsure how to get the
programming model’s software.

Adjust variables if they don't comply to the following defaults:
```Makefile
CC  = gcc-4.9                            # C11 compiler
MCC = mcc                                # mercurium compiler
NANOS_HOME = /apps/PM/ompss/2014-04-10
PAPI_HOME = /apps/PAPI/5.0.1
EXTRAE_HOME = /apps/CEPBATOOLS/extrae/latest/default/64
```
e.g. :
```bash
make NANOS_HOME=/usr/local
```

Each different configuration will be compiled in a subdirectory, with 6 targets

- cg, parallel version that is compiled with the instrumentation library of nanox
- cg_seq, sequential : same as above ignoring pragmas (entirely compiled with $(CC))

- cg_speedup, parallel version that only measures speed (compiled with performance library)
- cg_seq_speedup, sequential : same as above ignoring pragmas (entirely compiled with $(CC))

- cg_conv, parallel version that writes convergence and resilience values to a trace (perf)
- cg_seq_conv, sequential : same as above ignoring pragmas (entirely compiled with $(CC))


# Resilience

The default make target will be a non-resilient textbook Conjugate Gradient.
You may then choose resilience strategies for Detected Uncorrected Errors (DUE).

Options are :
- DUE: none async path lossy rollback

  If using the rollback DUE strategy, you will need to choose a (not "none") checkpointing
  method as well.
- CKPT: none disk mem

The name of each configuration's directory is then DUE_CKPT

Some valid examples :

| configuration           | directory      | name in paper                      |
|-------------------------|----------------|------------------------------------|
| DUE=none                | none_none      | baseline, trivial                  |
| DUE=async               | async_none     | AFEIR  (recover asynchronously)    |
| DUE=path                | path_none      | FEIR   (recover in critical path)  |
| DUE=lossy               | lossy_none     | Lossy                              |
| DUE=rollback CKPT=disk  | rollblack_disk | ckpt                               |
| DUE=rollback CKPT=mem   | rollblack_mem  | N/A                                |


# Running

You have to supply a file containing a spd matrix in [Matrix Market](https://math.nist.gov/MatrixMarket/) format.
All other parameters are optional and described below. Some are only available for some flavours of the resilient CG.

```
Usage: ./$directory/cg [options] <matrix-market-filename> [, ...]
Possible options are :
 ===  fault injection  ===
  -nf               Disabling faults simulation (default).
  -l     lambda     Inject errors with lambda meaning MTBE in usec.
  -nerr  N duration Inject N errors over a period of duration in usec.
                    Note : the options -nf, -l and -nerr are mutually exclusive.
  -mfs   strategy   Select an alternate (cf Agullo2013) strategy for multiple faults.
                    'strategy' must be one of global, uncorrelated, decorrelated.
                    Note : has no effect without errors. global is default.
 === run configuration ===
  -th    threads    Manually define number of threads.
  -nb    blocks     Defines the number of blocks in which to divide operations ;
                    their size will depend on the matrix' size.
  -r     runs       number of times to run a matrix solving.
  -cv    thres      Run until the error verifies ||b-Ax|| < thres * ||b|| (default 1e-10).
  -maxit N          Run no more than N iterations (default no limit).
  -seed  s          Initialize seed of each run with s. If 0 use different (random) seeds.
 === resilience method ===
  -ps    size       Defines page size (used on failure, in bytes, defaults to 4K).
                    Must be a multiple of the system page size (and a power of 2).
  -ckpt  N          Checkpointing frequency (expressed in iterations).
  -path  /path/dir  Path to a directory on local disk for checkpointing (default $TMPDIR).
  -prefix           Prefix of the name of checkpoint files.
All options apply to every following input file. You may re-specify them for each file.
```

e.g. the following command will run CG once with a MTBE of 1e8 us, and once without error
injection (without reloading the matrix from file) :

```
./cg -l 1e8 <file>.mtx -nf <file>.mtx
```

# Viewing

In the subdirectory utils/ you will find paraver configuration files that allow to view
symbols outputted to traces that are specific to this application.

The trace_to_plot.py script extracts these informations from configuration files into a
format that can be fed to any viewing tool, to plot convergence over time.

