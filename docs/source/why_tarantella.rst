Why Tarantella?
===============

Tarantella is an open-source Deep Learning framework that focuses on providing fast, scalable and
efficient training of Deep Neural Networks (DNNs) on High Performance Computing (HPC) clusters.

Goals
-----

Tarantella is designed to meet the following goals:

.. code-block:: text

  Tarantella...

    1. ...provides strong scalability
    2. ...is easy to use
    3. ...follows a synchronous training scheme
    4. ...integrates well with existing models
    5. ...provides support for GPU and CPU systems

Tarantella provides close to linear speed-up for the training of common Deep Learning architectures,
thus considerably reducing the required time-to-accuracy in many Deep Learning workflows.
To make this capability accessible to as many users as possible, Tarantella's interface
is designed such that its use does not require any expertise in HPC or parallel computing.

To allow integrating Tarantella into any TensorFlow-based Deep Learning workflow,
we put special emphasis on strictly following the synchronous optimization scheme
used to train DNNs. This guarantees that results obtained in serial execution can be
reproduced when using distributed training
(cf. however :ref:`these guidelines <points-to-consider-label>`),
so that computation can be scaled up at any point in time without losing reproducibility
of the results.

Furthermore, we made sure that existing TensorFlow 2/Keras
models can be made ready for distributed training with minimal effort
(follow the :ref:`Quick Start guide <quick-start-label>` to learn more).
Tarantella supports distributed training on GPU and pure CPU clusters,
independently of the hardware vendors.

.. todo::

   Performance Results

