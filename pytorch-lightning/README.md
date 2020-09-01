## PyTorch Lightning Tutorial

PyTorch Lightning is a lightweight PyTorch wrapper that helps you scale your models and write less boilerplate code. In this Tutorial we learn about this framework and how we can convert our PyTorch code to a Lightning code.

Lightning is nothing more than organized PyTorch code, so once you’ve organized it into a LightningModule, it automates most of the training for you.The beauty of Lightning is that it handles the details of when to validate, when to call .eval(), turning off gradients, detaching graphs, making sure you don’t enable shuffle for val, etc…

In this Tutorial we convert the code from the [PyTorch beginner Course #13](https://github.com/python-engineer/pytorchTutorial) to a Lightning code.

## Watch the Tutorial
  [![Alt text](https://img.youtube.com/vi/Hgg8Xy6IRig/hqdefault.jpg)](https://youtu.be/Hgg8Xy6IRig)
 
## Lightning Repo
[https://github.com/PyTorchLightning/pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning)

## Installation
```
conda install pytorch-lightning -c conda-forge
or
pip install pytorch-lightning
```