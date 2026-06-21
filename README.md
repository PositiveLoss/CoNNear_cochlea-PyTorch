# CoNNear: A convolutional neural-network model of human cochlear mechanics and filter tuning for real-time applications

If you use this code, please cite (bibtex given below [1]):    
Baby, D., Van Den Broucke, A. & Verhulst, S. A convolutional neural-network model of human cochlear mechanics and filter tuning for real-time applications. Nat Mach Intell (2021). https://doi.org/10.1038/s42256-020-00286-8

**The supporting paper can be found at [https://www.nature.com/articles/s42256-020-00286-8] with DOI 10.1038/s42256-020-00286-8 (https://arxiv.org/abs/2004.14832).**

> This work was funded with support from the EU Horizon 2020 programme under grant agreement No 678120 (RobSpear).


This repository contains notebooks for running and testing the CoNNear model. The full version `connear_notebook.ipynb` holds both the CoNNear model and the reference transmission line (TL) model (Verhulst et al.), the latter can be used as a validation tool. The `connear_notebook_light.ipynb` only runs the CoNNear model, which significantly reduces computation time. Both notebooks consist of independent blocks corresponding to different sections of the paper.

Besides the notebooks, the repository contains the PyTorch CoNNear implementation (`connear_pytorch.py`), helper utilities (`helper_ops.py`), a speech fragment (`dutch_sentence.wav`), and the reference TL model (`tlmodel`). The original Keras weights are still included as `connear/Gmodel.h5`, and the converted PyTorch weights are available as `connear/Gmodel.pt`.

## Setup

This project uses `pyproject.toml`, `uv.lock`, and `.python-version`.

```bash
uv sync
```

If you do not use `uv`, create a virtual environment and install the dependencies from `pyproject.toml` with your preferred Python packaging tool.

## Running Notebooks

Start Jupyter from the repository root:

```bash
uv run --with notebook jupyter notebook
```

Open either:

- `connear_notebook_light.ipynb` for the PyTorch CoNNear model only.
- `connear_notebook.ipynb` for CoNNear plus the TL reference model.

If you run the full notebook with the TL model for the first time, compile the C helper library:

```bash
cd tlmodel
gcc -shared -fpic -O3 -ffast-math -o tridiag.so cochlea_utils.c
cd ..
```

For Google Colab or another hosted runtime, run the same `gcc` command in a notebook cell before executing TL-model cells.
    
## CoNNear model specifications

The CoNNear model is an 8-layer, tanh, 64-filter-length deep convolutional neural-network model trained on 2310 speech sentences from the TIMIT speech dataset. It predicts basilar membrane displacements for 201 cochlear channels, covering approximately 100 Hz (channel 0) to 12 kHz (channel 200) based on the Greenwood map.
		
During CoNNear simulations, 256 context samples are added on both sides to account for possible information loss when slicing full speech fragments.

The CoNNear model can take variable-length stimuli as input. Because of the convolutional architecture, the sample length should be a multiple of 16.

## PyTorch model

The notebooks now use the PyTorch port:

```python
from connear_pytorch import load_connear

connear = load_connear("connear/Gmodel.pt")
prediction = connear.predict(stim)
```

To regenerate `connear/Gmodel.pt` from the original Keras HDF5 weights, run:

```bash
uv run python convert_keras_to_pytorch.py
```

## System test

The system was tested on a MacBook Pro 2015 (macOS Sierra v10.12.6) with 3.1 GHz Intel Corei7, 16 GB RAM, and on a MacBook Air 2017 (macOS Catalina v10.15.3) with 1.8 GHz Dual-Core Intel Core i5, 8 GB RAM. 

## Runtime notes

The PyTorch-only notebook cells run quickly on a typical laptop. The DPOAE simulation and TL reference model cells are slower because they run many stimulus presentations and analyses; the DPOAE block can take up to about 25 minutes.

----
For questions, please reach out to one of the corresponding authors

* Deepak Baby: deepakbabycet@gmail.com
* Arthur Van Den Broucke: arthur.vandenbroucke@ugent.be
* Sarah Verhulst: s.verhulst@ugent.be

----
[1] Bibtex
```
@Article{Baby2021,
author={Baby, Deepak
and Van Den Broucke, Arthur
and Verhulst, Sarah},
title={A convolutional neural-network model of human cochlear mechanics and filter tuning for real-time applications},
journal={Nature Machine Intelligence},
year={2021},
month={Feb},
day={08},
abstract={Auditory models are commonly used as feature extractors for automatic speech-recognition systems or as front-ends for robotics, machine-hearing and hearing-aid applications. Although auditory models can capture the biophysical and nonlinear properties of human hearing in great detail, these biophysical models are computationally expensive and cannot be used in real-time applications. We present a hybrid approach where convolutional neural networks are combined with computational neuroscience to yield a real-time end-to-end model for human cochlear mechanics, including level-dependent filter tuning (CoNNear). The CoNNear model was trained on acoustic speech material and its performance and applicability were evaluated using (unseen) sound stimuli commonly employed in cochlear mechanics research. The CoNNear model accurately simulates human cochlear frequency selectivity and its dependence on sound intensity, an essential quality for robust speech intelligibility at negative speech-to-background-noise ratios. The CoNNear architecture is based on parallel and differentiable computations and has the power to achieve real-time human performance. These unique CoNNear features will enable the next generation of human-like machine-hearing applications.},
issn={2522-5839},
doi={10.1038/s42256-020-00286-8},
url={https://doi.org/10.1038/s42256-020-00286-8}
}
```
