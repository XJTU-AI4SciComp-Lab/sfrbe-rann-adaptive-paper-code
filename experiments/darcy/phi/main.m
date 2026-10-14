function main()
%MAIN Run the fixed smooth multiscale Darcy experiment for seed 4.
    expDir = fileparts(mfilename('fullpath'));
    root = fileparts(fileparts(fileparts(expDir)));
    addpath(genpath(fullfile(root,'src')));
    addpath(fullfile(root,'experiments'));
    addpath(expDir);
    cfg = config();
    seeds = cfg.seed(:).';
    if isempty(seeds), seeds = 4; end
    res = struct('seed',cell(1,numel(seeds)),'initial',cell(1,numel(seeds)), ...
        'after_sfr',cell(1,numel(seeds)),'final',cell(1,numel(seeds)));
    for k = 1:numel(seeds)
        s = seeds(k); fprintf('\n========== seed %d (%d/%d) ==========\n',s,k,numel(seeds)); cfg.seed=s;
        if numel(seeds)>1, cfg.output.run_name=sprintf('%s_seed%d',cfg.problem,s); end
        run_experiment(cfg);
        if isempty(cfg.output.run_name), runDir=fullfile(cfg.output.results_root,cfg.problem); else, runDir=fullfile(cfg.output.results_root,cfg.output.run_name); end
        smFile=fullfile(runDir,'99_summary.mat');
        if exist(smFile,'file')==2
            S=load(smFile); res(k).seed=s; res(k).initial=S.summary.errors.initial.rel_L2; res(k).after_sfr=S.summary.errors.after_sfr.rel_L2; res(k).final=S.summary.errors.final.rel_L2;
        end
    end
    if numel(seeds)>=2
        outTxt=fullfile(cfg.output.results_root,[cfg.problem,'_sweep.txt']); fid=fopen(outTxt,'w','n','UTF-8');
        if fid>0
            fprintf(fid,'seed   initial     after_sfr   final\n');
            for k=1:numel(seeds), fprintf(fid,'%4d   %.6e   %.6e   %.6e\n',res(k).seed,res(k).initial,res(k).after_sfr,res(k).final); end
            fprintf(fid,'mean   %.6e   %.6e   %.6e\n',mean([res.initial]),mean([res.after_sfr]),mean([res.final]));
            fprintf(fid,'std    %.6e   %.6e   %.6e\n',std([res.initial]),std([res.after_sfr]),std([res.final])); fclose(fid);
        end
    end
end
