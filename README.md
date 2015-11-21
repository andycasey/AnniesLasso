# Annie's Lasso
*The Cannon* part 2: Compressed sensing edition

## Authors
- **Andy Casey** (Cambridge)
- **David W. Hogg** (NYU) (MPIA) (SCDA)
- **Melissa K. Ness** (MPIA)

## License
**Copyright 2015 the authors**.
The code in this repository is released under the open-source **MIT License**.
See the file `LICENSE` for more details.

## Code
[![Build Status](https://img.shields.io/travis/andycasey/AnniesLasso/code.svg)](https://travis-ci.org/andycasey/AnniesLasso)
[![Coverage Status](https://img.shields.io/coveralls/andycasey/AnniesLasso/code.svg)](https://coveralls.io/github/andycasey/AnniesLasso?branch=code)
[![Scrutinizer](https://img.shields.io/scrutinizer/g/andycasey/AnniesLasso.svg?b=code)](https://scrutinizer-ci.com/g/andycasey/AnniesLasso/?branch=code)
[![License](https://img.shields.io/badge/LICENSE-MIT-blue.svg)](https://github.com/andycasey/AnniesLasso/blob/master/LICENSE)

To install:

``
pip install https://github.com/andycasey/AnniesLasso/archive/code.zip
``

## Comments
- If we take *The Cannon* to large numbers of labels (say chemical abundances),
the model complexity grows very fast.
At the same time, we know that most chemicals affect very few wavelengths
in the spectrum; that is, we know that the problem is sparse.
Here we try to use standard methods to discover and enforce sparsity.
