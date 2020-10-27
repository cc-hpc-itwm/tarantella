## date: 03/31/2020
## time: 10.00 -- 10.20
## attendees:
  * Alexandra Carpen-Amarie [ACA]
  * Martin Kuehn [MK]
  * Peter Labus [PL]

## Status updates

@Martin:

* setting up ImageNet/ResNet on GPU-Cluster (styx)
  * single GPU works with TF2
  * multi-GPU: need to fix gaspirun/SSH
* plan: implement multi-threaded SynchCommunicator unit test
* idea: unit test for double buffering scheme

@Alex:

* change SynchCommunicator interface ('wait_all' -> finish(id))
* merged code review changes

@Peter:

* idea: daily short meeting (15min max) for status update in the morning
