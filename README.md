# Generative Adversarial Networks for Disentangling And Learning Framework (GANDALF)
GANDALF is an ad-hoc framework written in Python to define, train, test and visualize disentangling models. It is divided into two modules. The 'train' module, where you can define, train and test different models, and the 'visualization' folder, a bokeh server who display a representation of the model, allowing not just its visualization, but also interactions to understand how it works.

The project is structured as follow:
```
gandalf
├── LICENSE.txt
├── README.md
├── gandalf
│   ├── train
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── data
│   │   │   └── sample
│   │   │       ├── X.npy
│   │   │       ├── axis_labels.npy
│   │   │       ├── ids.npy
│   │   │       └── params.npy
│   │   ├── data_preparation.py
│   │   ├── models.py
│   │   ├── test_cli.py
│   │   ├── train_cli.py
│   │   └── utils.py
│   └── visualization
│       ├── README.md
│       ├── configuration.json
│       └── main.py
└── pyproject.toml
```

## Packaging project
Through the .toml file you can build this project using build to create a wheel that can be installed with pip (name of the .whl file is subject to change). 
All the dependencies used in this project will also be installed:

```
python -m build

pip install .\dist\gandalf-1.0.0-py3-none-any.whl 
```

Then you can use the cli tools installed simply calling them:

```
train_cli --help
test_cli --help

```

The installation will also include the dataset located in gandalf/train/data/sample, so that you can carry out a verification test easily. For instance:
```
train_cli --survey sample --epochs 5 -vv
test_cli <generated_model_id> --tsne_comparison
```

Note that the sample dataset it just for testing purposes.

You also can work with GANDALF without the need of the packaging process. Use requirements.txt to install all the dependencies with pip. You can work with train_cli and test_cli also as scripts. This has been tested using python 3.7.

```
pip install -r requirements.txt
```

```
python gandalf/train/train_cli.py --survey sample --epochs 5 -vv
python gandalf/train/test_cli.py <generated_model_id> --tsne_comparison
```
```


## More detail
To see in more depth how each of them works, consult the readme of each module ([gandalf/train/README.md](gandalf/train/README.md) and [gandalf/visualization/README.md](gandalf/visualization/README.md))
