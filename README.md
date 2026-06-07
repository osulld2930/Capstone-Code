Magnet Equilibrium Optimiser:
A computational physics project that solves an inverse design problem in magnetostatics: given a set of target positions, find the optimal non-uniform distribution of fixed magnets arranged on a ring such that freely floating magnets come to rest at those target positions.

Physical Problem:
A ring of fixed magnets exerts forces on a set of free-floating magnets in the interior. For a uniform ring, the floating magnets settle at equilibrium positions determined by symmetry. This project asks the reverse question: can we engineer the density of fixed magnets on the ring so that the equilibrium positions match an arbitrary target configuration?
The magnet density on the ring is parameterised as a Fourier series, and the Fourier coefficients are optimised to minimise the RMS distance between the actual equilibrium positions and the targets.

What the Code Does:
Models magnetic dipole-dipole interactions between fixed ring magnets and free-floating magnets
Finds equilibrium positions of the floating magnets by minimising the total potential energy using L-BFGS
Uses the Hungarian algorithm for optimal assignment between computed and target positions
Runs a two-phase global-then-local optimisation (CRS2-LM followed by BOBYQA) over the Fourier coefficients of the ring density
Outputs CSV files with the best density, equilibrium positions, and fixed magnet positions

Dependencies:
Eigen — linear algebra
NLopt — nonlinear optimisation

Compilation:
bashg++ -O2 -std=c++17 magnet_equilibrium_optimizer.cpp -o magnet_opt -I/path/to/eigen -lnlopt -lm

Output:
Running the binary produces four CSV files:

best_summary.csv — optimal Fourier coefficients and final RMS error
best_equilibrium.csv — computed vs target equilibrium positions
best_fixed.csv — positions of the fixed ring magnets
best_density.csv — the optimised angular density function
