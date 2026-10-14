function example_poisson_2peak()
%EXAMPLE_POISSON_2PEAK  Small-scale two-peak Poisson example (~1 minute).
%
%   Runs the full SFRBE pipeline (PDAD -> SFR -> RBE -> evaluate) on a coarse
%   grid so you can see the project end-to-end without waiting for the real
%   experiment. Results go to results/example_poisson_2peak/.
%
%   Usage:
%     cd example
%     example_poisson_2peak
    expDir = fileparts(mfilename('fullpath'));
    root   = fileparts(expDir);
    addpath(root);
    startup;                       % add src paths
    addpath(fullfile(root, 'experiments'));
    addpath(fullfile(root, 'experiments', 'poisson_2peak'));

    cfg = config();
    cfg.seed = 4;                % run_experiment requires a scalar seed
    cfg.grid.N = [60, 60];         % small grid
    cfg.model.neuron = 600;
    cfg.model.neuron_add = 200;
    cfg.pdad.n_pde_rtrain = 800;
    cfg.pdad.n_bd_rtrain = 200;
    cfg.pdad.n_neuron_rtrain = 300;
    cfg.rbe.max_family_rounds = 1; % one RBE round
    cfg.output.pp = 60;
    cfg.output.run_name = 'example_poisson_2peak';
    cfg.output.make_figs = false;  % faster; plot later with plotFromResults
    run_experiment(cfg);

    fprintf('\n[example] done. results in results/example_poisson_2peak/\n');
end
