# Train Adapter

This module is an adapter between the original GANDALF CLI code (``train`` folder) and the Python library API used in notebooks or Python scripts. See more about the adapter pattern in [https://refactoring.guru/design-patterns/adapter](https://refactoring.guru/design-patterns/adapter).

This implementation is organised as follows:

 - ``gandalf_train_interface.py``: Abstract class that defines all the necessary attributes and methods.
 - ``gandalf_train.py``: Implements the previous abstract class. This class does the same as ``train_cli.py``, but in this case it is necessary to do a step-by-step execution using all the GANDALFTrain methods.
 - ``utils/``: Includes functions adapted from ``train_cli.py``.

UML diagram available in `docs/design/gandalf-arch-actual.pdf`.