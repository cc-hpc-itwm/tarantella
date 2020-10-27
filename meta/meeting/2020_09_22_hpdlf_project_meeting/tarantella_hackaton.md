
# HP-DLF Project Meeting - Tarantella Hackaton

## location: Heidelberg
## date: 23/09/2020
## attendees:
  * Alexandra Carpen-Amarie [ACA]
  * Martin Kuehn [MK]
  * Peter Labus [PL]
  * Lynton Ardizzone 


### TODOs

* TF2.3
* build/compile - > move bcast/add call in model wrapper?
* CMake: add compile variables for python executable/make sure cmake detects the conda environment python 
* Batch size: check if larger than nranks
* Distributed datasets 
  - check whether batch size is a multiple of nranks/micro_batch
  - wrap datasets in `model.fit` and do not expose them to the user
    - users should batch datasets using `batch_size` and this will be automatically converted to `micro_batch_size` in the wrapper


### Tarantella_run
* option to print logging messages on the master rank only or on every rank
* redirect stdout/stderr to separated files for each rank
* use working directory for writing files

* run options
  - `tnt_run ./resnet.py -epochs 100 -bs 256` - localhost, 1 rank on GPUs if TF detects any
  - `tnt_run -no-gpu ./resnet.py` - localhost, 1 rank on CPU
  - `tnt_run -m machinefile ./resnet.py` - localhost, 1 rank per node, use GPUs if TF detects any
  - `tnt_run -m machinefile -no-gpus ./resnet.py` - localhost, 1 rank per node on CPUs
  - `tnt_run -ngpus ngpus ./resnet.py` - localhost, ranks use ngpus GPUs 
  - `tnt_run -n nranks ./resnet.py` - localhost, nranks 
    

### Python implementation
  - `sys.exit` return error exit code
  - write errors to stderr 
  - BatchNorm: show warning if micro-batch size is too small (16)
  - Logging: Print the microbatch/batch sizes
  - Callbacks 
      - Filter callbacks given by user in `model.fit` by single-rank or distributed execution (e.g. save modelonly on rank0)
      - Add `distributed_callbacks` arg to `model.fit` (execute on all ranks)
  - Extend `model.fit` with `local_batch_size` argument
  - Tarantella.init - rename `ngpus` and document usage


### Documentation

* Tarantella_init - make sure users call it before any computation
  - in particular disable the GPUs we don't need
* Add explanations on how to check for rank if you want code executed only on one node (or rank-specific)
* Use only one keras model in the program
* Custom training loops are not supported

* All TF optimizers are supported
* Document custom optimizer 
  - has to inherit from keras.optimizers

* Document usage of Batchnorm
  - write in the documentation that Batchnorm is computed over the `local_batch_size`, which should have a min value

* Batchsize and hyper-parameters expect global variables
  - document writing your code with the mb in mind (bs = mbs * nranks)


### Tutorials

* Extend/cleanup examples
* add GAN to our model examples


### Potential features
* Eager execution supported?
* Suport for custom training loop (gradient tape)?


### Pipelining API
* log the partitions description
* add CutLayers in the model at partition boundaries


