## SegmentaTion AwaRe cLusterING (starling)

Highly multiplexed imaging technologies such as Imaging Mass Cytometry (IMC) enable the quantification of the expression proteins in tissue sections while retaining spatial information. Data preprocessing pipelines subsequently segment the data to single cells, recording their average expression profile along with spatial characteristics (area, morphology, location etc.). However, segmentation of the resulting images to single cells remains a challenge, with doublets -- an area erroneously segmented as a single-cell that is composed of more than one 'true' single cell -- being frequent in densely packed tissues. This results in cells with implausible protein co-expression combinations, confounding the interpretation of important cellular populations across tissues.

While doublets have been extensively discussed in the context of single-cell RNA-sequencing analysis, there is currently no method to cluster IMC data while accounting for such segmentation errors. Therefore, we introduce SegmentaTion AwaRe cLusterING (STARLING), a probabilistic method tailored for densely packed tissues profiled with IMC that clusters the cells explicitly allowing for doublets resulting from mis-segmentation. To benchmark STARLING against a range of existing clustering methods, we further develop a novel evaluation score that penalizes methods that return clusters with biologically-implausible marker co-expression combinations. Finally, we generate IMC data of the human tonsil -- a densely packed human secondary lymphoid organ -- and demonstrate cellular states captured by STARLING identify known cell types not visible with other methods and important for understanding the dynamics of immune response.

https://github.com/camlab-bioml/starling/blob/main/starling.png

## Installation

_starling_ can be cloned and installed locally using access to the Github repository,

```
git clone https://github.com/camlab-bioml/starling.git && cd starling
```

Note that _starling_ requires at least python version 3.9.

After cloning the repository, the next step is to install the required dependencies. There are two recommended methods:

### 1. Use `requirements.tex` and your own virtual environment:

We use virtualenvwrapper (4.8.4) to create and activated a standalone virtual environment for _starling_:

```
pip install virtualenvwrapper==4.8.4
mkvirtualenv starling
```

For convenience, one can install packages in the tested environment:

```
pip install -r requirements.txt
```

The virtual environment can be activated and deactivated subsequently:

```
workon starling
deactivate
```

### 2. Use Poetry and `pyproject.toml`.

[Poetry](https://python-poetry.org/) is a packaging and dependency management tool can simplify code development and deployment. If you do not have Poetry installed, you can find instructions [here](https://python-poetry.org/docs/).

Once poetry is installed, navigate to the `starling` directory and run `poetry install`. This will download the required packages into a virtual environment and install Starling in development mode. The location and state of the virtual environment may depend on your system. For more details, see [the documentation](https://python-poetry.org/docs/managing-environments/).


A list of minimal required packages needed for _starling_ can be found in setup.py if creating a new virtual environment is not an option.

## Getting started

Launch the interactive tutorial: [jupyter notebook][tutorial]

## License

Distributed under the terms of the [MIT license][license],
_starling_ is free and open source software.

## Authors

Jett (Yuju) Lee & Kieran Campbell
Lunenfeld-Tanenbaum Research Institute & University of Toronto

<!-- github-only -->

[tutorial]: https://github.com/camlab-bioml/starling/blob/main/docs/tutorial/getting-started.ipynb
[license]: https://github.com/camlab-bioml/starling/blob/main/LICENSE
