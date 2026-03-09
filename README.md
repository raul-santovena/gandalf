![GANDALF icon](docs/logo.png)

# Generative Adversarial Networks for Disentangling And Learning Framework (GANDALF)

GANDALF is an ad-hoc framework written in Python to define, train, test and visualize disentangling models. It is divided into three modules. The [train](gandalf/train/) module, where you can define and train different models. The [test](gandalf/test/) module, where you can test the models using different techniques, such as the t-SNE. The [visualization](gandalf/visualization) folder, a bokeh server who display a representation of the model, allowing not just its visualization, but also interactions to understand how it works.


The project is structured as follow (not all files are shown):
```
gandalf/
├── LICENSE.txt
├── README.md
├── gandalf/
│   ├── train
│   │   ├── README.md
│   │   ├── data/
│   │   │   └── sample/
│   │   ├── data_preparation.py
│   │   ├── models.py
│   │   ├── test_cli.py
│   │   ├── train_cli.py
│   │   └── utils.py
│   ├── train_adapter/
│   │   ├── README.md
│   │   ├── gandalf_train_interface.py
│   │   └── gandalf_train.py
│   ├── test/
│   │   ├── cond_params.md
│   │   └── tsne.py
│   ├── utils/
│   │   └── generate_samples.py
│   └── visualization/
│       ├── README.md
│       ├── visualization_config.yaml
│       ├── load_utils.py
│       ├── app.py
│       └── main.py
├── notebooks/
└── pyproject.toml
```

## Packaging project
Through the `.toml` file you can build this project using `build` to create a wheel that can be installed with `pip` (name of the `.whl` file is subject to change).
All the dependencies used in this project will also be installed:

```
python -m build

pip install .\dist\gandalf-4.0.0-py3-none-any.whl
```

Then you can use the cli tools installed simply calling them:

```
train_cli --help
test_cli --help
```

The installation will also include the dataset located in `gandalf/train/data/sample`, so that you can carry out a verification test easily. For instance:
```
train_cli --epochs 5 -vv
test_cli <generated_model_id> --tsne_comparison
```

Note that the sample dataset it just for testing purposes.

You also can work with GANDALF without the need of the packaging process. Use [requirements.txt](requirements.txt) to install all the dependencies with `pip`. You can work with `train_cli` and `test_cli` also as scripts.
```
pip install -r requirements.txt
```

```
python gandalf/train/train_cli.py --epochs 5 -vv
python gandalf/train/test_cli.py <generated_model_id> --tsne_comparison
```

NOTES:
 - This has been tested using python 3.9 and python 3.12 (2025 Supported versions).

## Notebooks

The ``notebooks`` folder contains some Python notebooks for testing GANDALF's functionalities. To work with them it is recommended to use the Dockerfile, which will open a Jupyter environment.

You can read more about them in the corresponding [README](notebooks/README.md).

## Test

Due to TensorFlow's stateful nature, running tests concurrently within a single process using the standard ``pytest test`` command will result in errors (`ValueError: tf.function only supports singleton tf.Variables created on the first call`). Each test file must be executed in an isolated process to ensure a clean state.

A Python script has been created to handle this sequential execution. To run all tests, use the following command:

````bash
python run_tests.py
````

## More detail
To see in more depth how each of them works, consult the README of each module:
- [gandalf/train/README.md](gandalf/train/README.md)
- [gandalf/test/README.md](gandalf/test/README.md)
- [gandalf/visualization/README.md](gandalf/visualization/README.md)
