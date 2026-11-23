## Introduction 
A python code that calculates the pair product approximation to a many body action. Currently we support a variety of Helium-Helium interactions via the [heprops](https://github.com/agdelma/heprops/tree/master) library but any desired interaction function can be added as long as it is centrosymmetric.

## Usage
A table can be quickly generated for a box of 6 x 6 x 6 for the Aziz 79 potential for $\tau = 0.025$.
```bash
python3 main.py -t 0.025 --box 6 6 6 --dim 3 -n 4 --n_tau 6 --dr 0.05 --ord 50 --n_d 6 --suffix 2
```
This should write a config_{suffix}.ini file in the directory with all simulation parameters and a folder called Pair_data_{suffix} with all fitted tables. 

The kind of potential function is set by the ```pot_f``` variable in ```main.py```. The box size is set by the box flag --box fla and --tau is set by 0.025. Other parameters need to be tuned for convergence which will be detailed in a forthcoming publication.

The code has been optimised with the help of Claude Opus 5.5 but has been checked for correctness and I take full responsibility. Feel free to reach out if you have any questions.
