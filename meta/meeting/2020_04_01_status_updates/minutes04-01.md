## date: 04/01/2020
## time: 09.30 -- 10.00
## attendees:
  * Alexandra Carpen-Amarie [ACA]
  * Martin Kuehn [MK]
  * Peter Labus [PL]

## code review

* ACA proposed a coding guidedline based on the old HP-DLF guideline
* MK will have a look, so we can discuss it next time

## updates

@Martin:

* benchmarks ImageNet/ResNet on STYX up to 16 GPUs
  * time per iteration 0.5s (1GPU) -> 0.7s (16GPUs)
  * no stalling (as seen before on Taurus by ACA)
  * correct results? -> TODO: compare test set accuracy (full 100 epoch run in future?)
  * TODO: repeat with TensorBoard traces for further analysis
  * TODO: repeat with Horovod
  * TODO: redo on 32 GPUs

@Alex:

* refactor gpi_comm_lib
