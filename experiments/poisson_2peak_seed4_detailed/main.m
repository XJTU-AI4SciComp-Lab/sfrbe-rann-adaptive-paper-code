function main()
%MAIN  Seed-4 Poisson run with detailed stage and residual diagnostics.
    expDir = fileparts(mfilename('fullpath'));
    root = fileparts(fileparts(expDir));
    addpath(genpath(fullfile(root,'src')),'-begin');
    addpath(fullfile(root,'experiments','poisson_2peak'),'-begin');
    addpath(fullfile(root,'experiments'),'-begin');
    addpath(expDir,'-begin');

    cfg = config();
    cfg.seed = 4;
    cfg.paper.table = 'tab02_poisson2peak_seed4_detailed';
    cfg.output.run_name = 'poisson_2peak_seed4_detailed';
    cfg.output.make_figs = false;
    cfg.output.detailed_residuals = true;
    cfg.rbe.max_family_rounds = 8;
    run_experiment(cfg);
end
