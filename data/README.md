# data

Read-only inputs: reference-solution `.mat` files and grid data.
The retained smooth multiscale Darcy experiment uses `darcy_phi.mat` for
the unit-source problem `-div(K_phi grad u) = 1`. The file contains the
saved reference grid and solution arrays used by `darcy_phi`.

The periodic KdV experiment uses `KdV62_reference_100x100.mat`. 
