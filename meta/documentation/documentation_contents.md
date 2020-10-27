# Documentation

software tools:
* https://readthedocs.org/

## Contents

### Very quick overview (landing page)
* pictures
* 1-2 sentences what it does

### Installation
* dependencies/requirements
* build from source
* [installation from pip/conda packages]
* [running in the cloud]

### Features
* distributed data parallel training
  * what is it?
  * thinks to know/mind:
    * global vs local batch size
    * adaption of learning rate
    * BatchNorm layers (computed over local batch size)
    * the notion of ranks -> c.f. advanced topics
* distributed data sets

### Quick start
* code example: simple TF Keras model with lines to add
* explanation with the following important points:
  * when to call tarantella_init
  * only one Keras model per program
  * no support for custom training loops
  * all TF optimizers are supported
  * how to store/load model
  * datasets (again?)
* how to use `tnt_run`
  * machine files
  * logging and how to redirect it
  * all run options

### Tutorials
* Image recognition
  * ResNet50 (for SC)
* Segmentation
  * Mask-RCNN
* Videos
  * ??? (ask @Ricard, @Kalun)
* Text2Speech
  * WaveNet
* NLP
  * Transformers (for SC)
  * RNN-based (which?)
  * BERT
* Generative Models
  * GAN (which one? ask @Ricard)
  * VAE (ask @Kalun)

### FAQ
* installation issues
* environment issues

### Advanced topics
* using ranks
* using callbacks
* using custom optimizers (inherit from Keras.optimizers)
* setting local batch size
* setting fusion threshold
