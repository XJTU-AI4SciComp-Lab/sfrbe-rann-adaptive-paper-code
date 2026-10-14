function smoke()
%SMOKE  Small-scale poisson_2peak test (about 1 minute).
%   Same pipeline as main but with a coarse grid and few features. Results go
%   to results/poisson_2peak_smoke/ so the real results are not overwritten.
    expDir = fileparts(mfilename('fullpath'));
    root   = fileparts(fileparts(expDir));
    % Put this checkout first so a MATLAB startup path cannot shadow the
    % implementation with the current SFR/RBE modules.
    addpath(genpath(fullfile(root, 'src')),'-begin');
    addpath(fullfile(root, 'experiments'),'-begin');
    addpath(expDir,'-begin');

    cfg = config();
    cfg.seed = 1;                    % smoke is a single-run diagnostic
    cfg.grid.N = [60, 60];              % small grid
    cfg.model.neuron = 600;
    cfg.model.neuron_add = 200;
    cfg.pdad.n_pde_rtrain = 800;
    cfg.pdad.n_bd_rtrain = 200;
    cfg.pdad.n_neuron_rtrain = 300;
    cfg.sfr.cfg.diag_N = [60, 60];     % small cell-centered diagnostic grid
    cfg.rbe.max_family_rounds = 1;      % one RBE round
    cfg.rbe.neuron_add = cfg.model.neuron_add;
    cfg.rbe.m_block0 = cfg.model.neuron_add;
    cfg.rbe.m_block_max = cfg.model.neuron_add;
    cfg.output.pp = 60;
    cfg.output.make_figs = false;
    cfg.output.save_data = false;
    cfg.output.run_name = 'poisson_2peak_smoke';
    cfg.output.make_figs = false;       % faster; plot later with plotFromResults
    run_experiment(cfg);
end
