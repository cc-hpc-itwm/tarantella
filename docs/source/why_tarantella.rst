Why Tarantella?
===============

Tarantella is an open-source Deep Learning framework that focuses on providing fast, scalable and
efficient training of Deep Neural Networks (DNNs) on High Performance Computing (HPC) clusters.

Goals
-----

Tarantella was designed to meet the following goals:

.. code-block:: text

  Tarantella...

    1. ...provides strong scalability
    2. ...is easy to use
    3. ...follows a synchronous training scheme
    4. ...extends existing models with minimal effort
    5. ...provides support for GPU and CPU systems

Tarantella provides close to linear speed-up for the training of common Deep Learning architectures,
thus reducing the required time-to-accuracy in many Deep Learning workflows considerably.
To make this capability accessible to as many users as possible, Tarantella's user interface
was designed in such a way that no knowledge of HPC or parallel computing is required
to use it.

To allow integrating Tarantella into any TensorFlow-based Deep Learning workflow,
we put special emphasis on strictly following the synchronous scheme of the optimization process
used to train DNNs. This guarantees that results obtained in serial execution can be
reproduced when using distributed training
(c.f. however the guidelines :ref:`below <points-to-consider-label>`),
so that computation can be scaled up at any point in time without losing reproducibility
of the results.

Furthermore, we made sure that existing TensorFlow/Keras
models can be made ready for distributed training with minimal effort
(follow the :ref:`quick start guide <quick-start-label>` to learn more).
Tarantella supports distributed training on GPU and pure CPU clusters,
independently of the hardware vendor.

Performance results
-------------------

.. todo::

   [add plots]

