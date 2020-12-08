Distributed Deep Learning Resources
======

## Overview

* Demystifying Parallel and Distributed Deep Learning,
  [paper](https://arxiv.org/abs/1802.09941) and [video](https://www.youtube.com/watch?v=xtxxLWZznBI)
* Scalable Deep Learning on Distributed Infrastructures: Challenges, Techniques and Tools,
  [paper](https://arxiv.org/abs/1903.11314)
* Distributed Learning Systems with First-Order Methods, [book](https://www.nowpublishers.com/article/Details/DBS-062)

## Distributed Training

### Model parallelism
  * [Mesh Tensorflow, Google Brain](https://arxiv.org/abs/1811.02084)
    * [Github](https://github.com/tensorflow/mesh)
    * implemented in TensorFlow
  * [TF Replicator, Google Deepmind](https://arxiv.org/abs/1902.00465)
    * part of tf.distribute.Strategy of TensorFlow
  * [LBANN, LLNL](https://lbann.readthedocs.io/en/latest/publications.html)
    * inspired by PyTorch API
  * [FlexFlow](https://arxiv.org/abs/1802.04924.pdf), Stanford, Microsoft

### Pipelining
  * [Hy-Par-Flow, Ohio State University](https://arxiv.org/abs/1911.05146.pdf)
    * Data and pipelining parallelism based on Tensorflow, Keras and MPI
  * [PipeDream, Microsoft Research](https://github.com/msr-fiddle/pipedream)
    * implemented in PyTorch
  * [GPipe, Google Brain](https://arxiv.org/abs/1811.06965)
    * [Code available for TensorFlow](https://github.com/tensorflow/lingvo/blob/master/lingvo/core/gpipe.py)
    * also cf. https://github.com/tensorflow/lingvo/issues/112
    * [PyTorch implementation](https://github.com/kakaobrain/torchgpipe)

### Data parallelism (distributed)
  * [Horovod, Uber](https://arxiv.org/abs/1802.05799) 
    * [Github](https://github.com/horovod/horovod)
    * support for TensorFlow, Keras, PyTorch, Apache MXNet

### Native support in frameworks
  * PyTorch
    * [Official tutorials](https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html)
    * [Medium.com blog post](https://medium.com/intel-student-ambassadors/distributed-training-of-deep-learning-models-with-pytorch-1123fa538848)
  * TensorFlow
    * [Official documentation](https://www.tensorflow.org/guide/distributed_training#overview)

### Hybrid & special purpose
* Microsoft [DeepSpeed](https://github.com/microsoft/DeepSpeed)
* single GPU optimization, reduce memory impact on single GPU
    * MSR, [Gist: Efficient Data Encoding for Deep Neural Network Training](https://www.microsoft.com/en-us/research/publication/gist-efficient-data-encoding-deep-neural-network-training/)
* data loading optimzation
    * MSR, [THEMIS: Fair and Efficient GPU Cluster Scheduling, USENIX NSDI'20](https://www.microsoft.com/en-us/research/publication/themis-fair-and-efficient-gpu-cluster-scheduling/)
    * MSR, [The Case for Unifying Data Loading in Machine Learning Clusters, USENIX HotCloud'19](https://www.microsoft.com/en-us/research/publication/the-case-for-unifying-data-loading-in-machine-learning-clusters/)


## DNN optimization

### 1st order gradient-based
* LARS
  * UC Berkeley, NVidia, [paper](https://arxiv.org/abs/1708.03888)
* LAMB
  * [blog post](https://towardsdatascience.com/an-intuitive-understanding-of-the-lamb-optimizer-46f8c0ae4866)
  * Google, [paper](https://openreview.net/pdf?id=Syx4wnEtvH)
* ZeRO
  * MSR, [paper](https://www.microsoft.com/en-us/research/publication/zero-memory-optimization-towards-training-a-trillion-parameter-models/)
* Adasum
  * MSR, [paper](https://arxiv.org/abs/2006.02924)
  * [Horovod documentation](https://horovod.readthedocs.io/en/latest/adasum_user_guide_include.html)
* SWAP
  * U Berkeley & Apple, [paper](https://arxiv.org/abs/2001.02312)
* NovoGrad
  * NVidia, [paper](https://arxiv.org/pdf/1905.11286.pdf)

### 2nd order gradient-based

* K-FAC
* distributed K-FAC, [paper](https://arxiv.org/pdf/1811.12019.pdf)

### Asynchronous/stale gradient-based

* https://arxiv.org/pdf/1802.09941.pdf
  * Hoefler's review about parallel DL. In particular: Fig. 18 & 7.1 & 7.2 (and references therein)

* https://arxiv.org/abs/1604.00981
  * Some interesting comparison between synch and asynch distributed SGD.
  * This paper seem to have led to asynchronous optimizers falling somewhat out of favour

* https://proceedings.neurips.cc/paper/2013/hash/b7bb35b9c6ca2aee2df08cf09d7016c2-Abstract.html
  * Probably the first paper on stale Parameter Server approaches

* https://arxiv.org/pdf/1805.09767.pdf and https://arxiv.org/abs/1808.07217
  * Two papers about local SGD (which is sort of the collectives equivalent to the stale parameter server approach).
  * The introduction in the first paper explains the difference between asynchronous optimization using collectives and parameters servers, respectively, quite nicely.

### Tool boxes
* BackPACK (pyTorch)
  * [webpage](https://backpack.pt/)
  * [paper](https://openreview.net/forum?id=BJlrF24twB)

## Benchmarks

* [MLPerf](https://mlperf.org/)
* [MLBench](https://mlbench.github.io/)
* [Deep500](https://www.deep500.org/)
* [DAWNBench](https://dawn.cs.stanford.edu/benchmark/) -> culminated into MLPerf

## Datasets

### Vision

* ImageNet
* [openImages](https://storage.googleapis.com/openimages/web/index.html)
* [YFCC100m](https://arxiv.org/abs/1503.01817)
* [cityscapes](https://www.cityscapes-dataset.com/)

## Large scale training of (large) models

### Generell
* Microsoft Research [Project Fiddle](https://www.microsoft.com/en-us/research/project/fiddle/)
* OpenAI & Microsoft [OpenAI blog post](https://openai.com/blog/microsoft/), [Microsoft blog post](https://news.microsoft.com/2019/07/22/openai-forms-exclusive-computing-partnership-with-microsoft-to-build-new-azure-ai-supercomputing-technologies/)

### Computer Vision:
* ResNet50 benchmarks/records
  * [Sony in 224s blog post](https://news.developer.nvidia.com/sony-breaks-resnet-50-training-record-with-nvidia-v100-tensor-core-gpus/)
* [AmoebaNet: Regularized Evolution for Image Classifier Architecture Search](https://arxiv.org/abs/1802.01548)
* [EfficientNet: Rethinking Model Scaling for CNNs](https://arxiv.org/abs/1905.11946) [Tan, et al. 2019]

### NLP
* NVidia [BERT & GPT-2 Training](https://devblogs.nvidia.com/training-bert-with-gpus/)
* NVidia [Project Megatron](https://github.com/nvidia/megatron-lm)
* Microsoft [Turing-NLG](https://www.microsoft.com/en-us/research/blog/turing-nlg-a-17-billion-parameter-language-model-by-microsoft/)
* OpenAI [GPT-2: 1.5B](https://openai.com/blog/gpt-2-1-5b-release/)
* OpenAI [GPT-3: 175B](https://towardsdatascience.com/gpt-3-the-new-mighty-language-model-from-openai-a74ff35346fc), [paper](https://arxiv.org/abs/2005.14165)
* OpenAI [Image-GPT](https://openai.com/blog/image-gpt/)

### Deep Learning Scaling
* Baidu paper [Deep Learning Scaling is Predictable, Empirically](https://arxiv.org/abs/1712.00409)
* OpenAI blog [How AI training scales](https://openai.com/blog/science-of-ai/)
* OpenAI paper [Scaling Laws for Autoregressive Generative Modeling](https://arxiv.org/abs/2010.14701)

### Large Models:
* ZeRO & DeepSpeed [blog post](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)

## Large Batch Size (Problem)

### Papers
* OpenAI's [An Empirical Model of Large-Batch Training](https://arxiv.org/abs/1812.06162.pdf)
* Google's [Measuring the effects of data parallelism on neural network training](https://arxiv.org/abs/1811.03600)
* [Pipelined Backpropagation at Scale: Training Large Models without Batches](https://arxiv.org/pdf/2003.11666.pdf)
* [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/pdf/2002.05709.pdf) -> large batch sizes may be beneficial in self-supervised learning

## Collective communication

### Communication Libraries for Deep Learning
  * [NCCL, NVidia](https://github.com/nvidia/nccl)
      * collectives for GPUs
      * Allreduce, Reduce, Broadcast, Allgather etc.
      * implementation based on ring algorithms
  * [Gloo, Facebook](https://github.com/facebookincubator/gloo)
      * collectives for CPUs, GPUs
      * Allreduce, Reduce, Broadcast, Allgather etc.
      * mutiple algoritms for each collective, CUDA-aware, overlapping
  * [Blink](https://arxiv.org/abs/1910.04940), Microsoft Research UC Berkeley
      * comm library for heterogeneous networks for GPUs
      * Allreduce, Broadcast
      * NCCL API
  * [Tensorflow tf.distribute module](https://www.tensorflow.org/api_docs/python/tf/distribute)
      * collectives for GPUs
      * Allreduce, Reduce, Broadcast
  * [Aluminum](https://ieeexplore.ieee.org/document/8638639), LLNL (LBANN)
      * collectives for GPUs
      * Allreduce, Allgather, Reduce-scatter, Reduce
      * blocking and non-blocking Allreduce (Rabenseifner alg), support for multiple, concurrent Allreduce ops
      * based on NCCL, MPI
  * [BlueConnect](https://mlsys.org/Conferences/2019/doc/2019/130.pdf), IBM, SysML2019
      * hierarchical (topology-aware) Allreduce for GPUs
      * based on MPI and NCCL
  * [Baidu Allreduce](https://github.com/baidu-research/baidu-allreduce)
      * Ring allreduce implementation for GPUs
      * based on MPI


### Allreduce algorithms papers

* [Sparse Allreduce: Efficient Scalable Communication for Power-Law Data [H. Zhao, JF Canny]](https://www.researchgate.net/publication/259239833_Sparse_Allreduce_Efficient_Scalable_Communication_for_Power-Law_Data)
* [More efficient reduction algorithms for non-power-of-two number of processors in message-passing parallel systems [R Rabenseifner, JL Träff]](https://fs.hlrs.de/projects/rabenseifner/publ/myreduce_pvmmpi2004_talk.pdf)
* [Performance Analysis of MPI Collective Operations [J. Pjesivac–Grbovic, et al.]](ftp://ftp2.uk.i-scream.org/sites/ftp.netlib.org/utk/people/JackDongarra/PAPERS/collective-cc-2006.pdf)
* [BandwidthOptimalAll-reduceAlgorithmsforClustersof Workstations [Pitch Patarasuk, Xin Yuan]](http://websrv.cs.fsu.edu/~xyuan/paper/09jpdc.pdf)
* [Butterfly-like Algorithms for GASPI Split Phase Allreduce [V. End, et al.]](https://www.researchgate.net/profile/Vanessa_End/publication/315669443_Butterfly-like_Algorithms_for_GASPI_Split_Phase_Allreduce/links/58ddfeff92851cd2d3e3748e/Butterfly-like-Algorithms-for-GASPI-Split-Phase-Allreduce.pdf)
* [Optimization of Collective Communication Operations in MPICH [R. Thakur, et al.]](https://www.mcs.anl.gov/~thakur/papers/ijhpca-coll.pdf)

### Sparse/Compressed communication
* SC. Renggli, D. Alistarh, M. Aghagolzadeh, T. Hoefler, 
[SparCML: High-Performance Sparse Communication for Machine Learning](https://spcl.inf.ethz.ch/Publications/index.php?pub=351)
* Nvidia Blogs, 
[Using Tensor Cores for Mixed-Precision Scientific Computing](https://devblogs.nvidia.com/tensor-cores-mixed-precision-scientific-computing/)


## Tarantella development

### C++ resources
* [fffaraz/awesome-cpp](https://github.com/fffaraz/awesome-cpp)
* [rigtorp/awesome-modern-cpp](https://github.com/rigtorp/awesome-modern-cpp)
* [C++ FAQ](https://isocpp.org/faq)
* [Herb Sutter's blog](https://herbsutter.com/)
* [Fluent{C++}](https://www.fluentcpp.com/)

### TensorFlow
* [Official documentation](https://www.tensorflow.org/guide)
* special topics
  * [TF op in C++](https://www.tensorflow.org/guide/create_op)
  * [Profiling TF](https://www.tensorflow.org/guide/profiler)
  * [TF Graph Optimization](https://www.tensorflow.org/guide/graph_optimization)
* [Reproducibility in TF](https://github.com/NVIDIA/framework-determinism)

### CMake programming
* [**Modern CMake** book and additional resources](https://cliutils.gitlab.io/modern-cmake/)
* https://pabloariasal.github.io/2018/02/19/its-time-to-do-cmake-right


