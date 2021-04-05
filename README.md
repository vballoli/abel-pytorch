# How to decay your Learning Rate (PyTorch)

PyTorch implementation of `ABEL` LRScheduler based on weight-norm. If you find this work interesting, do consider starring the repository. If you use this in your research, don't forget to cite!

[Original paper](https://arxiv.org/pdf/2103.12682v1.pdf): `https://arxiv.org/pdf/2103.12682v1.pdf`

[Docs](https://abel-pytorch.readthedocs.io/en/latest/): `https://abel-pytorch.readthedocs.io/en/latest/`

## Installation

```
pip install abel-pytorch
```

## Usage

```python
import torch
from torch import nn, optim
from abel import ABEL

model = resnet18()
optim = optim.SGD(model.parameters(), 1e-3)
scheduler = ABEL(optim, 0.2)

for i, (images, labels) in enumerate(trainloader):
  # forward pass...
  optim.step()
  scheduler.step()

```

## Cite original paper:
```
@article{lewkowycz2021decay,
  title={How to decay your learning rate},
  author={Lewkowycz, Aitor},
  journal={arXiv preprint arXiv:2103.12682},
  year={2021}
}
```

## Cite this work:
```
@misc{abel2021pytorch,
  author = {Vaibhav Balloli},
  title = {A PyTorch implementation of ABEL},
  year = {2021},
  howpublished = {\url{https://github.com/tourdeml/abel-pytorch}}
}
```
