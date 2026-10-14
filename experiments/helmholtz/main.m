function main()
%MAIN  helmholtz experiment entry.
%
%   cfg.seed can be a scalar or a vector; each seed is saved to
%   results/poisson_2peak_sfr_seed<k>/ and a mean/std table is printed.
    expDir = fileparts(mfilename('fullpath'));
    root   = fileparts(fileparts(expDir));
    addpath(genpath(fullfile(root, 'src')));
    addpath(fullfile(root, 'experiments'));
    addpath(expDir);

    cfg = config();
    seeds = cfg.seed(:).';
    n = numel(seeds);
    if n == 0, seeds = 1; n = 1; end

    res = struct('seed', cell(1, n), 'initial', cell(1, n), ...
        'after_sfr', cell(1, n), 'final', cell(1, n));

    for k = 1:n
        s = seeds(k);
        fprintf('\n========== seed %d (%d/%d) ==========\n', s, k, n);
        cfg.seed = s;
        if n > 1
            cfg.output.run_name = sprintf('%s_seed%d', cfg.output.run_tag_base, s);
        else
            cfg.output.run_name = cfg.output.run_tag_base;
        end

        run_experiment(cfg);

        runDir = fullfile(cfg.output.results_root, cfg.output.run_name);
        smFile = fullfile(runDir, '99_summary.mat');
        if exist(smFile, 'file') == 2
            S = load(smFile);
            res(k).seed = s;
            res(k).initial  = S.summary.errors.initial.rel_L2;
            res(k).after_sfr = S.summary.errors.after_sfr.rel_L2;
            res(k).final    = S.summary.errors.final.rel_L2;
        end
    end

    if n >= 2
        fprintf('\n========== multi-seed summary ==========\n');
        fprintf('seed   initial     after_sfr   final\n');
        for k = 1:n
            fprintf('%4d   %.4e   %.4e   %.4e\n', res(k).seed, ...
                res(k).initial, res(k).after_sfr, res(k).final);
        end
        fprintf('mean   %.4e   %.4e   %.4e\n', ...
            mean([res.initial]), mean([res.after_sfr]), mean([res.final]));
        fprintf('std    %.4e   %.4e   %.4e\n', ...
            std([res.initial]), std([res.after_sfr]), std([res.final]));

        outTxt = fullfile(cfg.output.results_root, [cfg.output.run_tag_base, '_sweep.txt']);
        fid = fopen(outTxt, 'w', 'n', 'UTF-8');
        if fid > 0
            fprintf(fid, 'seed   initial     after_sfr   final\n');
            for k = 1:n
                fprintf(fid, '%4d   %.6e   %.6e   %.6e\n', res(k).seed, ...
                    res(k).initial, res(k).after_sfr, res(k).final);
            end
            fprintf(fid, 'mean   %.6e   %.6e   %.6e\n', ...
                mean([res.initial]), mean([res.after_sfr]), mean([res.final]));
            fprintf(fid, 'std    %.6e   %.6e   %.6e\n', ...
                std([res.initial]), std([res.after_sfr]), std([res.final]));
            fclose(fid);
            fprintf('saved: %s\n', outTxt);
        end
    end
end