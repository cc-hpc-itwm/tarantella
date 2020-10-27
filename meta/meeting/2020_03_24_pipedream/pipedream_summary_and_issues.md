# PipeDream: Fast and Efficient Pipeline Parallel DNN Training

Harlap et al., Microsoft Research, Carnegie Mellon, Stanford<br>
**arxiv:1806.03377v1 [cs.DS] 8 Jun 2018**

Narayanan et al., SOSP 2019<br>
**PipeDream: Generalized Pipeline Parallelism for DNN Training**

### Metrics of performance

* *statistical efficiency:* # of epochs to accuracy
* *hardware efficiency:* time per epoch
* *time to accuracy:* product of statistical & hardware efficiency

### Terminology [p.2-4]

* **Data Parallelism**
    - inputs are partitioned across workers
    - each worker maintains a local copy of the model weights and trains on its own partition of
      inputs while periodically synchronizing weights with other workers
* **Model Parallelism (intra-batch parallelism/ "naive" pipelining)**
    - layers in a DNN model are partitioned across the available workers
* **Hybrid Intra-batch Parallelism**
    - data parallelism for some layers combined with model parallelism
* **Inter-batch Parallelism** - GPipe
    - split a mini-batch into micro-batches which are then fed to a pipeline

## Pipeline parallelism

* partition into "stages" (consecutive set of layers) include "input" & "output stage"
* communication can be done asynchronously
* challenges:
  1. efficient (automatic) partitioning
  1. scheduling of micro-batches for throughput optimization
  1. maintaining synchronous SGD (-> statistical efficiency)

### [A] Automatic partitioning

**1. Profiling**
  * benchmark tests on single machine<br>
  [**our assumptions:** each & every layer fits in single memory alone, all machines are equal]
  * for each layer *l* measure:
    1. T_l = time for fwd & bwd combined (estimate with 1000 batches)
    1. a_l = size of output tensor (e.g. in bytes)
    1. w_l = size of parameters (e.g. in bytes)

**2. Partitioning**
  * goals of the partitioning scheme **should lead to minimization of overall training time:**
    1. approximately same amount of work (e.g. FLOPs?) for each stage ("balanced pipeline")
    1. minimize combined size of all output (activation) tensors of stages
  * output of algorithm:
    1. partitioning of layers into stages
    1. model replication factor of each stage (for fine-grained data parallelism)
    1. optimal number of mini-batches (for busy pipeline)
  * partitioning has **optimal sub-problem property**:<br>
    - optimal pipeline for given machine count is composed of *optimal*
      sub-pipelines for some smaller machine counts
    - **steady state throughput** of the pipeline is the throughput of the slowest stage
    - the partitioning problem is equivalent to minimizing the time taken by the slowest stage of the pipeline

**3. Algorithm**
* **output A(N, M)**, i.e. *time of slowest stage* as a function of # of layers *N* and # of machines *M*
* Let A(j, m) be slowest stage in *optimal* pipeline for layers 1 -> j, with data parallelism on *m* machines
* **time for single stage:**<br>
  T(i->j, m) = 1/m \max(\sum_{l=i}^j T_l, \sum_{l=i}^j W_l^m),
  where W_l^m ... communication time for layer *l* to reduce gradients of *m* machines
  * **Q: Is this a good approximation if we overlap most of the reduction time?**
  * **Q: What would W_l^m be for our reduce algorithm?**
  * **Q: Shall we allow for data parallelism in stages at all??**
* **slowest stage in optimal pipeline** on *m* machines:
  1. single stage: A(j, m) = T(1->j, m)
  1. multi-stage: A(j, m) = min_{1<=m'<m, 1<=i<j} max (A(i,m'), 2C_i, T(i+1->j, m-m'), where C_i is communication time from layer i to i+1
* **initialization:**
  * A(1,m) = T(1->1, m)
  * A(1->j, 1) = T(1->j, 1)
* solve with dynamic programming, **runtime:** \mathcal O(N^2 M^2)
* optimal # of mini-batches **(NOAM):**

  NOAM = \ceil{m/(#machines in input stage)}

### [B] Work scheduling

* bi-directional pipeline **Q: Is "unrolling" fwd & bwd equivalent to GPipe?**
* trade-offs:
  * prioritize fwd: cannot update weights so often
  * prioritize bwd: may result in more idle machines
* proposed solution (static):
  * 1 fwd + 1 bwd (1F1B) -> full throughput in steady state (when balanced pipeline)
  * 1F1B-RR with round-robin data parallelism (miniBatchID MOD stageReplicaID => same machine process fwd & bwd for given batch)
* **Q: Other schedules possible (e.g. 1F1B-RR w/ flushes)?**

### [C] Keeping the synchronous SGD scheme ("efficient learning")

* problems:
  * weights are stale: fwd & bwd are carried out w/ different versions of model
  * different stages have different degrees of staleness (last stage has minimum, first stage has maximum)
* solutions:
  1. weights stashing
  1. vertical synchronization
1. **weight stashing:**
  * fwd: use "newest" weights available for given batch i
  * bwd: do updates w/ these weights
  * need to copy local weights NOAM-times
  * **update scheme:** w^t = w^{t-1} - \eta \nabla f (w_1^{t-n+1}, w_1^{t-n+2},..., w_n^{t}), n ... # stages
2. **vertical synchronization:**
  * fwd & bwd on *all* stages: use weights w^{(i-n)}
  * same amount of additional memory as weight stashing, put more meta information to communicate
  * **update scheme:** w^t = w^{t-1} - \eta \nabla f (w_1^{t-n+1}, ..., w_n^{t-n+1})
  * **Q: Does this have the same convergence properties as sync SGD?**
