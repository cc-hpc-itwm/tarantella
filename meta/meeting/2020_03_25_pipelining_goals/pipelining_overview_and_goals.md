## date: 03/25/2020
## time: 15.00 -- 17.00
## attendees:
  * Alexandra Carpen-Amarie [ACA]
  * Martin Kuehn [MK]
  * Peter Labus [PL]

## Goal: Conclude pipelining study with Martin's resources

**https://gitlab.itwm.fraunhofer.de/labus/epigram-hs/-/blob/master/reports_EU/deliverable5.3_01_2020/D5.3.pdf**
Chapters 2.1, 2.2, 3.1, 3.2, 4.5, 5.2, 5.3

**https://gitlab.itwm.fraunhofer.de/labus/epigram-hs/-/blob/master/docs/pipelining/pipelining.pdf**


### Take home messages

* data parallelism and model parallelism communicate mutually exclusive parts of model
* *convolutional layers*: small weights, big activations
* *fully connected layers*: big weights, small activations
* *CNN*: big images small channels at begin, small images big channels at end
* pipelining has around 2x overhead compared to data parallelism 
* model has to be split homogeniously versus flop, memory, network communication
* Load balance: might be hard to get, maybe use data or model parallelism to split single layer

### Aims

* support fully synchronous distributed SGD
* use pipelining to support large models (gpipe scheme with 1F1B scheduling)
* use data parallelism on top of pipelining (for performance)
* no splitting of single layer for now

### Implementation

* Can we do (TensorFlow) subgraphs and merge them later on?
* Can we create small example and distribute it manually to two pieces?
* cf. meeting 03/27/2020 for technical details
