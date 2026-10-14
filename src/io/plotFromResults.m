function plotFromResults(cfg, runDir)
%PLOTFROMRESULTS Draw the paper figures from saved result snapshots (no rerun).
%
%   Usage:
%     cfg = config();
%     plotFromResults(cfg);                                   % results/<problem>/
%     plotFromResults(cfg, fullfile(cfg.output.results_root, 'poisson_2peak_seed4'));
%
%   Requires (saved by run_experiment):
%     01_initial_global_RaNN.mat   (U_pred, U_exact, U_error, xp1, xp2)
%     02_after_SFR.mat
%     03_after_RBE.mat
%     02_SFR_anchors.mat           (sfr_data.centers, sfr_data.diag)
    if nargin < 2 || isempty(runDir)
        runDir = fullfile(cfg.output.results_root, cfg.problem);
    end

    f1 = fullfile(runDir, '01_initial_global_RaNN.mat');
    f2 = fullfile(runDir, '02_after_SFR.mat');
    f3 = fullfile(runDir, '03_after_RBE.mat');
    fa = fullfile(runDir, '02_SFR_anchors.mat');
    need = {f1, f2, f3, fa};
    names = {'01_initial_global_RaNN.mat','02_after_SFR.mat', ...
             '03_after_RBE.mat','02_SFR_anchors.mat'};
    for k = 1:numel(need)
        if exist(need{k}, 'file') ~= 2
            error('plotFromResults:MissingSnapshot', ...
                'Missing %s under %s. Run the experiment first.', names{k}, runDir);
        end
    end

    S1 = load(f1);
    S2 = load(f2);
    S3 = load(f3);
    A  = load(fa);

    pd = struct();
    pd.xp1 = S3.xp1;
    pd.xp2 = S3.xp2;
    pd.exact = S3.U_exact(:);
    pd.pred_initial = S1.U_pred(:);
    pd.pred_sfr = S2.U_pred(:);
    pd.pred_final = S3.U_pred(:);
    if isfield(S3, 'mask')
        pd.mask = S3.mask;
    end
    pd.centers = A.sfr_data.centers;
    pd.diagS = A.sfr_data.diag;   % fields: G_SAB, xx1, xx2, domain_labels, ...

    makePaperFigures(cfg, pd);
    fprintf('[plotFromResults] figures written to %s\n', cfg.output.figures_root);
end
