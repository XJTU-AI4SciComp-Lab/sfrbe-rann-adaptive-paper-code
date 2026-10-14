function main()
    expDir=fileparts(mfilename('fullpath')); root=fileparts(fileparts(expDir));
    addpath(genpath(fullfile(root,'src'))); addpath(fullfile(root,'src'),'-begin'); addpath(fullfile(root,'src','solver'),'-begin');
    addpath(fullfile(root,'experiments'),'-begin'); addpath(expDir,'-begin');
    cfg=config(); seeds=cfg.seed(:).';
    for k=1:numel(seeds)
        cfg.seed=seeds(k); if numel(seeds)>1, cfg.output.run_name=sprintf('%s_seed%d',cfg.output.run_tag_base,seeds(k)); else, cfg.output.run_name=cfg.output.run_tag_base; end
        run_experiment(cfg);
    end
end
