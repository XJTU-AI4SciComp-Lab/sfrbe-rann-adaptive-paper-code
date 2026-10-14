function saveSummary(dirName, cfg, m_init, m_sfr, m_final, diagS, ctx)
%SAVESUMMARY Save a minimal per-run summary with per-stage errors
%   (Rel./Abs. L2 and H1), paper table mapping, and the RBE selection log.
    summary = struct();
    summary.paper = cfg.paper;
    summary.seed = cfg.seed;
    summary.errors = struct();
    summary.errors.initial  = m_init;
    summary.errors.after_sfr = m_sfr;
    summary.errors.final    = m_final;
    summary.sfr = struct('G_SAB', diagS.G_SAB, ...
        'num_region', getfield_default_local(diagS,'num_region',0), ...
        'SAB_accept', getfield_default_local(ctx,'SAB_accept',0), ...
        'm_S_target', getfield_default_local(diagS,'m_S_target',[]), ...
        'm_B_target', getfield_default_local(diagS,'m_B_target',[]));
    summary.rbe = struct( ...
        'orb_block_count', getfield_default_local(ctx,'orb_block_count',0), ...
        'last_r_opt', getfield_default_local(ctx,'last_r_opt',[]), ...
        'log', getfield_default_local(ctx,'rbe_log',struct([])));
    summary.total_features = cfg.model.neuron + ...
        getfield_default_local(ctx,'orb_block_count',0) * cfg.model.neuron_add;
    summary.cfg = cfg;

    matName = fullfile(dirName, '99_summary.mat');
    save(matName, 'summary');
    fprintf('  [save] %s\n', matName);
end

function val = getfield_default_local(s, name, default_val)
    if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
        val = s.(name);
    else
        val = default_val;
    end
end