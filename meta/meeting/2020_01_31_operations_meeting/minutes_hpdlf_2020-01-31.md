
HP-DLF Meeting 2020 --- Meeting Minutes
======

## date: 01/31/2020
## time: 10.00 -- 11.00
## attendees:
  * Alexandra Carpen-Amarie [ACA]
  * Martin Kuehn [MK]
  * Peter Labus [PL]

## Tasks for HPDLF Project Report (due Feb 10)
* Benchmarks - ResNet CIFAR10
  - scaling on CPUs on Taurus (up to 128 nodes)
  - scaling on GPUs on our cluster (up to 32 GPUs)
  - compare to Horovod (with MPI)
* **TODOs**
  - Fix dataset split into train/validation/test sets
  - Avoid validation at every training epoch to obtain only training timings

## Code review meeting
* Fixed for **March 9** with the Aloma team
* **TODOs**
    - Set up CI [@ACA]
    - Clean up build system
    - Add unit tests
    - Add scaling tests
    - Set up environment for the Aloma team to run our code (conda packages, GPI-2, etc.)
* Can we move our CI to Schmidthuber04 (including GPUs)? [@PL]

## Dataset issues with distributed training
* Issues
    - processes were loading random micro-batches, could not ensure they were disjoint
    - final evaluation was done on a subset of the validation dataset 
* Specific solution for ResNet56 with CIFAR10
    - shuffle entire dataset with a fixed seed on all processes
    - use the `shard` option to select only samples corresponding to the current rank
* TODO: devise a generic solution transparent for the users
    



