# Neural Network-based Moments for Probability Density Function Identification
---

## Introduction

This repository contains the code and examples presenting a method for moment functions identification, with neural networks, from samples of a distribution.
These functions are a representation of this distribution, that allows for an estimation of its probability density function.

---

## Installation

The code can be executed using the Conda environment provided:
```bash
# Install the Conda environment
conda env create -f environment.yml
```

>:warning: The 2D examples need to use the distributions from [deep-kexpfam](https://github.com/kevin-w-li/deep-kexpfam).
>The path to the source code must be provided in the first cell of the [2D example notebook](https://github.com/GLevillain/Neural-Network-based-Moments-PDF/blob/main/mennpdf_2D_example.ipynb).
>Moreover, the code will return errors as the code it built for an older version of `Scipy`, and some `import`s need to be modified:
>```python
># Replace 
>from scipy.misc import ...
># with
>from scipy.special import ...
>```
>when errors occur.

---

## License

This work is licensed under a [CC-by-nc Creative Commons Attribution 4.0 Unported License](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en").
For details, see the [LICENSE](https://github.com/GLevillain/Neural-Network-based-Moments-PDF/blob/main/LICENSE.md) file.
---

## Acknowledgments

This research was financed by the French government IDEX-ISITE initiative 16-IDEX-0001 (CAP 20-25) under the program DATA.
This organization is gratefully acknowledged for the support.