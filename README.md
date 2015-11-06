# Annie's Lasso
*The Cannon* part 2: Compressed sensing edition

## Authors
- Andy Casey (Cambridge)
- David W. Hogg (NYU) (MPIA) (SCDA)
- Melissa K. Ness

## License
**Copyright 2015 the authors**.
The code in this repository is released under the open-source *MIT License*.
See the file `LICENSE` for more details.

## Comments
- If we take *The Cannon* to large numbers of labels (say chemical abundances),
the model complexity grows very fast.
At the same time, we know that most chemicals affect very few wavelengths
in the spectrum; that is, we know that the problem is sparse.
Here we try to use standard methods to discover and enforce sparsity.
