[![Galaxy Tool Linting and Tests for push and PR](https://github.com/goeckslab/gleam/actions/workflows/pr.yaml/badge.svg?branch=main)](https://github.com/goeckslab/gleam/actions/workflows/pr.yaml/badge.svg)


# GLEAM: Galaxy Learning and Modeling

GLEAM (Galaxy Learning and Modeling) is a suite of machine learning tools for the [Galaxy](https://galaxyproject.org/) platform. Developed by the [Goecks Lab](https://goeckslab.org/), GLEAM empowers researchers to train models, generate predictions, and produce reproducible reports—all from a user-friendly interface without writing code.

## Features
- Machine learning support for diverse data types: tabular, image, text, categorical, and more
- Deep learning via Ludwig and automated ML via PyCaret
- Easy installation in Galaxy via XML wrappers
- Reproducible and scalable workflows
- Auto-generated visual reports

## Available Tools

### 1. TabularLearner

Machine learning for structured tabular datasets using [PyCaret](https://pycaret.org/).

- Train classification and regression models
- Evaluate performance and extract feature importance
- Generate predictions on new datasets
- Create interactive HTML reports

### 2. ImageLearner

Deep learning-based image classification using [Ludwig](https://ludwig.ai/).

- Train on labeled image datasets using pretrained or custom backbones
- Support for train/validation/test split
- Predict on test images and evaluate results
- Generate visual reports with learning curves, confusion matrices, and examples

### 3. Galaxy-Ludwig

General-purpose interface to Ludwig's full machine learning capabilities.

- Train and evaluate models on structured input (tabular, image, text, etc.)
- Expose Ludwig’s flexible configuration system
- Ideal for users needing advanced model customization

A fourth toolset, Galaxy-Tiler, for large image preprocessing, is in development and will be included in future updates.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/goeckslab/gleam.git
   ```

2. Add entries for each tool in your tool_conf.xml of your galaxy instance:
    ```xml
    <tool file="<path-to-your-local-tabularlearner/pycaret_train.xml>" />
    <tool file="<path-to-your-local-imagelearner/image_learner_train.xml>" />
    <tool file="<path-to-your-local-galaxy-ludwig/ludwig_train.xml>" />
    ```


## Contributing
We welcome contributions. To propose new tools, report bugs, or suggest improvements:

1. Fork the repository

2. Create a feature branch

3. Commit and test your changes

4. Submit a pull request


