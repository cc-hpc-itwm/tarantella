# Proposal

## Parts of HP-DLF
1. compiler:
  * system independent graph (ONNX) to hardward-optimized graph (now ONNX++)
2. performance model:
  * estimate cost (runtime, energy consumption) for single operators on different hardware plattforms
3. runtime system:
  * task based, dynamic, fault-tolerant (?) execution on optimal hardware
4. monitoring system:
  * analysis & optimization of framework and models

## Goals

### Technical desiderata
1. scalability
  * strong scaling
  * weak scaling (large, distributed models)
  * distribution of large single layers
2. portability
  * import of existing models from other frameworks (via ONNX) 
  * execution on various types of hardware
3. automatic parallelization
  * use all possible layers of parallelization
  * automatic partitioning of model
  * automatic distribution of hardware (possible metrics: runtime, energy efficiency, compute budget)
4. fault tolerance:
  * should be able to deal with crashes
  * possible stop of execution and continuation on other plattform
5. automatic data flow:
  * use data locality
  * efficient data flow
6. simulation of computational cost:
  * metics: runtime, energy/compute budget
7. monitoring

### Conceptual & political goals
* enable DL experts to use HPC system without prior knowledge
* abstraction of learning algorithms -> optimize HPC system w/o knowledge of DL
* create a counter pole to development in the US

### Parallelization strategies
known:
* data parallel in shared memory (mini-batch SGD)
* data parallel in distributed setting (copy of model: distributed, synchronous, mini-batch SGD)
want:
* asynchronous
* task parallel, i.e. model/pipeline parallel

### Existing Frameworks
* Caffe, Torch, Theano
* TensorFlow, CNTK, IDLF (Intel, discontinued), Caffe2

### Differences to TF
* TF is bottom-up (compute graph composed out of atomic tensor operations)
* HP-DLF is supposed to be *top-down* (split on net, layer, data when needed)

### Disadvantages of existing solutions
* data parallelism:
  * minimal batch size (for efficiency of BLAS operations)
  * maximal batch size (diminishing returns in reducing iteration counts/FLOPs to accuracy)
* synchronous optimizers:
  * bandwidth becomes bottle neck

## Solution strategies

### architecture of HP-DLF
* translate DNN into Petri Net with compiler ("in a way transparent to the user"... how??)
* static performance model may help the compiler
* execute dynamically, fault tolerant on heterogeneous cluster with GSPC
* dynamic (!) performance model aids GSPC with scheduling decision

### Key components
compiler:
* generates Petri Net
* hints for data flow optimization for GSPC (based on availability of GPUs, SIMD, etc.)
* use and extent AnyDSL to generate highly optimized code for parts of DNN
* *should enable parallelization of single layers* (!!)
* goal: reduce memory bandwidth

performance model:
* checks admissibility of execution for a  given operation depending on requirements of
  compiler & scheduler
* gives performance estimation for given operation
* performance metrics: runtime (1st), energy consumption & budget (2nd)
* estimation based on: BLAS/conv operations & network parameter (1st), then dynamically
  during the runtime

runtime system (GSPC):
* takes data locality & performance estimation for optimal scheduling into account
* reschedules in case of failure (how does that conflict with divergence of algorithm??)
* overlap of computation and communication ("using single-sided communication via GPI2) (!!)

solvers:
* synchronous SGD (Forrest et al. '16, arXiv:1511.00175)
* asynchronous solvers (literature study & implemenation,
  in particular Keuper et al., arXiv: 1505.04956)

simulator:
* based on runtime system & performance model
* simulate execution of Petri nets (could we simulate ONNX++ nets instead?!)
* estimate runtime of full DNNs
* should enable improvement of performance model, s.t. scheduling can be improved

monitoring:
* extension of GSPC monitor & debugger
* use monitoring & tracing infrastructure of ZiH
* goal: enable user to track/debug execution of DNN

### scientific applications
* huge DNNs, huge data sets
* domains: autonomes driving, medical imaging (stereo videos of liver operations)
* data sets: existing: CityScape, generate: using mixed reality

## Milestones

### Milestone 1 (01.11.2018)

* first working prototype
* HP-DLF is usable by DL and HPC experts
* ONNX models are executable (originally: Caffe), but functionality is limited
* performance model gives hints to optimize *training time*
* parallelisation granularity: single layers
* scalability up to 64 nodes
* making code open source + documentation & tutorials

### Milestone 2 (01.11.2019)

* execute any ONNX model
* asynchronous solver
* performance model gives hints to optimize *energy efficient execution*
* parallelisation granularity: distributed layers
* scalability up to 128 nodes
* monitoring of individual components of DNNs (for detailed analysis during training)

### Milestone 3 (01.11.2020)

* sparse gradient communication
* simulator for time and energy estimation before execution
* benchmark of cutting edge computer vision applications
* user documentation/tutorials/installation tool

## Packages

* total PMs ITWM: (12 + 12 + 4) * 3 years

**AP0: Project management & coordination (12PM)**
  AP0.1 (6PM)
  * coordination partners
  * evaluation time/work plan
  * organisation project meetings (4x/year)
  * website & social media
  * organisation tutorials
  AP0.2 (6PM)
  * technical documentation (API, Installation guide)
  * examples
  * workshops (HPC & ML centres, online)

**AP1: Framework and integration (33PM)**
  AP1.1 (12PM)
  * integration layer implementations
  * integration numerical solver
  * integration monitoring
  * integration simulator
  AP1.2 (18PM)
  * synchronous SGD
  * gradient quantization
  * sparse gradient communication
  * asynchronous SGD (with sparse gradients)
  AP1.3 (3PM)
  * design and benchmark of large DNNs

**AP2: Compiler (6PM)**
  * translate ONNX to Petri Nets (NOW: to ONNX++)
  AP2.1 (3PM)
  * intermediate representation of DNN with hardware annotation (ONNX++)
  * ONNX frontend
  * performance model frontend
  * granularity: single DNN layers
  AP2.2 (2PM)
  * granularity: distributed layers
  * code generation (AnyDSL)
  AP2.3 (1PM)
  * use optimiser depending on cost model

**AP3: Performance Model (6PM)**
  * offline & online information for runtime system & compiler
  AP3.1 (2PM)
  * performance and data model for hardware and framework
  AP3.2 (2PM)
  * model cost for data transfers
  AP3.3 (2PM)
  * add metrics: energy effciency & dollar cost
  AP3.4 (0PM)
  * online (dynamic) performance information

**AP4: Runtime system/GPI-Space (24PM)**
  AP4.1 (12PM)
  * GSPC: API for performance model and hardware class
  * GSPC: use performance model information in scheduler
  * collect performance information in data base
  AP4.2 (3PM)
  * online monitoring of DNN execution performance
  * visualization of performance metrics
  AP4.3 (6PM)
  * simulator to approximate trainings time
  AP4.4 (3PM)
  * bundeling for deployment ("one click")

**AP5: Application & Training (3PM)**
  AP5.1 (0PM)
  * generate huge, annotated data sets with mixed reality
  AP5.2 (3PM)
  * benchmark of training & inference for large data sets & DNNs
  AP5.3 (0PM)
  * installation & user support

## Usage

* open source
* GPI-Space: open source until 31.10.2019
* documentation & tutorials (video tutorials!!)
* industrial use, e.g. autonomes driving
* workshops at conferences, ML-Kompentenzzentren, Gauss-Alliance
* tutorials at European HPC centers & HPC conferences
* publications at NIPS/ICML/CVPR & ISC/SC
* growing number of users
* website
