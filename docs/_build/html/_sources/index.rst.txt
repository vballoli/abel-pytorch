.. ABEL PyTorch documentation master file, created by
   sphinx-quickstart on Tue Mar 30 16:21:20 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ABEL PyTorch's documentation!
========================================

PyTorch implementation of `ABEL` LRScheduler based on weight-norm. If you find this work interesting, do consider starring the repository. If you use this in your research, don't forget to cite!

Original paper can be found at arxiv_.

.. _arxiv: https://arxiv.org/pdf/2103.12682v1.pdf

*********
Install
*********

.. code-block:: console

   pip install abel-pytorch

******************
Sample usage
******************

.. code-block:: python

   import torch
   from torch import nn, optim
   from abel import ABEL

   model = resnet18()
   optim = optim.SGD(model.parameters(), 1e-3)
   scheduler = ABEL(optim, 0.9)

   for i, (images, labels) in enumerate(trainloader):
      # forward pass...
      optim.step()
      scheduler.step()


.. toctree::
   :maxdepth: 2
   :caption: API reference

   abel