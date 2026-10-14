function varargout = SFR_module(action, varargin)
%SFR_MODULE_COMPAT
% One file managing SFR:
%   diagS = SFR_module('diagnose', ctx, sab_cfg)
%   ctx   = SFR_module('refresh',  ctx, diagS, refresh_cfg)

    switch lower(action)
        case 'diagnose'
            varargout{1} = sab_diagnostic_local(varargin{:});

        case 'refresh'
            varargout{1} = sab_refresh_local(varargin{:});

        otherwise
            error('Unknown SFR_module action = %s', action);
    end
end

%% ============================================================
% SFR Step 1: structure diagnosis
%% ============================================================
function diagS = sab_diagnostic_local(ctx,cfg)

    cfg = fill_sab_cfg_local(cfg);

    SAB_eps = cfg.SAB_eps;

    % The diagnostic set is a Cartesian cell partition.  Each cell keeps one
    % center for connectivity/anchor bookkeeping, while the indicators are
    % integrated with a fixed 2-by-2 Gauss--Legendre rule.
    if isfield(cfg,'diag_N') && ~isempty(cfg.diag_N)
        diag_N = cfg.diag_N;
    elseif isfield(cfg,'SAB_pp') && ~isempty(cfg.SAB_pp)
        % Backward compatibility for old experiment files.
        diag_N = [cfg.SAB_pp,cfg.SAB_pp];
    else
        error('SAB diagnostic grid requires cfg.diag_N = [Nx,Ny].');
    end
    if isscalar(diag_N), diag_N = [diag_N,diag_N]; end
    if numel(diag_N) ~= 2 || any(~isfinite(diag_N)) || ...
            any(diag_N < 1) || any(diag_N ~= round(diag_N))
        error('cfg.diag_N must contain one or two positive integers.');
    end
    diag_N = double(diag_N(:).');
    Nx_diag = diag_N(1);
    Ny_diag = diag_N(2);
    hx = (ctx.a2-ctx.a1)/Nx_diag;
    hy = (ctx.a4-ctx.a3)/Ny_diag;

    xx1 = ctx.a1 + ((0:Nx_diag-1)+0.5)*hx;
    xx2 = ctx.a3 + ((0:Ny_diag-1)+0.5)*hy;
    [Xsab,Ysab] = meshgrid(xx1,xx2);
    sab_points = [Xsab(:),Ysab(:)];
    Nsab = size(sab_points,1);

    % Four tensor-product Gauss points per diagnostic cell.  The cell index
    % remains the first dimension of all aggregated arrays below; q indexes
    % the four quadrature points and is never treated as a new cell.
    gauss1 = 1/sqrt(3);
    qxi = [-gauss1; gauss1; -gauss1; gauss1];
    qeta = [-gauss1; -gauss1; gauss1; gauss1];
    quad_points = zeros(4*Nsab,2);
    for qq = 1:4
        idx_q = (qq-1)*Nsab + (1:Nsab);
        quad_points(idx_q,:) = sab_points + ...
            [0.5*hx*qxi(qq), 0.5*hy*qeta(qq)];
    end

    if isfield(ctx,'inside_fn') && ~isempty(ctx.inside_fn)
        quad_valid = logical(gather(ctx.inside_fn(quad_points)));
        quad_valid = reshape(quad_valid(:),4,Nsab);
    else
        quad_valid = true(4,Nsab);
    end
    % A cut cell is valid when at least one Gauss point lies in the domain.
    % Its effective measure is the sum of the valid Gauss weights.
    n_quad_valid = sum(quad_valid,1).';
    valid_mask = n_quad_valid > 0;
    mu = (hx*hy/4)*n_quad_valid;
    Nvalid = nnz(valid_mask);
    if Nvalid == 0
        error('The diagnostic grid has no valid Gauss points inside the domain.');
    end

    term = struct('w',ctx.w,'b',ctx.b);

    op = get_problem_op_local(ctx);
    need_xy = op_coeff_maybe_nonzero_local(op.alpha5);

    if isfield(ctx,'use_orth_basis') && ctx.use_orth_basis == 1
        coef_now = ctx.T_current * ctx.gamma_current;
    else
        coef_now = ctx.weight;
    end

    % Evaluate the current solution and PDE residual at all four Gauss points.
    u_val = BasisFunction('tanh',term,quad_points,0,0) * coef_now;
    ux    = BasisFunction('tanh',term,quad_points,1,1) * coef_now;
    uy    = BasisFunction('tanh',term,quad_points,1,2) * coef_now;
    uxx   = BasisFunction('tanh',term,quad_points,2,1) * coef_now;
    uyy   = BasisFunction('tanh',term,quad_points,2,2) * coef_now;
    if need_xy
        uxy = BasisFunction('tanh',term,quad_points,2,3) * coef_now;
    else
        uxy = zeros(4*Nsab,1);
    end

    f_quad = problem_rhs_local(ctx,quad_points);
    Lu = apply_operator_to_solution_values_local(quad_points,u_val,ux,uxx,uy,uyy,uxy,op);
    R_quad = Lu - f_quad;

    u_val=gather(u_val); ux=gather(ux); uy=gather(uy);
    uxx=gather(uxx); uyy=gather(uyy); R_quad=gather(R_quad); f_quad=gather(f_quad);

    % Aggregate the four quadrature values into one cell-average indicator.
    % The physical cell measure is applied once below in eta_i=mu_i*S_i.
    s_res = zeros(Nsab,1);
    s_grad = zeros(Nsab,1);
    for qq = 1:4
        idx_q = (qq-1)*Nsab + (1:Nsab);
        wq = double(quad_valid(qq,:).')./(n_quad_valid + (n_quad_valid==0));
        s_res = s_res + wq.*abs(R_quad(idx_q));
        s_grad = s_grad + wq.*sqrt(abs(ux(idx_q)).^2 + abs(uy(idx_q)).^2);
    end
    s_res(~valid_mask) = 0;
    s_grad(~valid_mask) = 0;
    s_res_hat = s_res./(max(s_res) + SAB_eps);
    s_grad_hat = s_grad./(max(s_grad) + SAB_eps);
    theta_res = cfg.theta_res;
    theta_grad = cfg.theta_grad;
    S_score = theta_res.*s_res_hat + theta_grad.*s_grad_hat;
    S_score(~valid_mask) = 0;
    S_norm = S_score./(max(S_score) + SAB_eps);
    S_norm(~valid_mask) = 0;
    etaR = s_res.*mu;
    etaG = s_grad.*mu;
    % The cell indicator is the normalized pointwise score multiplied by
    % the diagnostic-cell measure.  Dörfler marking is therefore area-aware
    % without introducing the old absolute/top-ratio truncation.
    eta2 = mu.*S_norm;
    eta2(~valid_mask) = 0;
    eta2_total = sum(eta2);
    S_hat = S_norm;

    % Single global SFR region.  partition_grid is retained only as a
    % backward-compatible field; all SISC configurations use [1,1].
    partition_grid = cfg.partition_grid;
    [partition_id,num_partition,partition_raw_id] = ...
        build_macro_partition_local(sab_points,valid_mask,ctx,partition_grid);

    % One global Dörfler marking set over the complete valid domain.
    marked_mask = false(Nsab,1);
    partition_energy = zeros(num_partition,1);
    partition_marked_energy = zeros(num_partition,1);
    partition_lambda = zeros(num_partition,1);
    partition_dorfler_ok = false(num_partition,1);
    partition_valid_count = zeros(num_partition,1);
    idx_valid = find(valid_mask);
    [eta_valid,ord_valid] = sort(eta2(idx_valid),'descend');
    nmark_global = 0;
    if ~isempty(idx_valid) && sum(eta_valid) > SAB_eps
        nmark_global = find(cumsum(eta_valid) >= ...
            cfg.theta_dorfler*sum(eta_valid),1,'first');
        if isempty(nmark_global), nmark_global = numel(ord_valid); end
        idx_mark_global = idx_valid(ord_valid(1:nmark_global));
        marked_mask(idx_mark_global) = true;
    else
        idx_mark_global = zeros(0,1);
    end
    eta2_marked_total = sum(eta2(marked_mask));
    global_dorfler_lambda = eta2_marked_total/(eta2_total + SAB_eps);
    global_dorfler_ok = global_dorfler_lambda + SAB_eps >= cfg.theta_dorfler;
    for pp = 1:num_partition
        idx_pp = find(valid_mask & partition_id == pp);
        partition_valid_count(pp) = numel(idx_pp);
        if isempty(idx_pp), continue; end
        E_pp = sum(eta2(idx_pp));
        partition_energy(pp) = E_pp;
        idx_mark_pp = idx_mark_global(partition_id(idx_mark_global) == pp);
        partition_marked_energy(pp) = sum(eta2(idx_mark_pp));
        partition_lambda(pp) = partition_marked_energy(pp)/(E_pp + SAB_eps);
        % Compatibility field: it records the single global Dörfler result,
        % not a separate local marking decision.
        partition_dorfler_ok(pp) = global_dorfler_ok;
    end

    % Backward-compatible global coverage diagnostics.

    delta_floor = cfg.delta_floor;
    mean_eta2 = eta2_total/(sum(mu(valid_mask)) + SAB_eps);
    density_score = eta2 + delta_floor*mean_eta2*mu;
    density_score(~valid_mask) = 0;
    p_S = density_score ./ (sum(density_score) + SAB_eps);

    r_res = s_res;
    r_res(~valid_mask) = 0;
    R_total = sum(r_res) + SAB_eps;
    S_hat(~valid_mask) = 0;

    % Diagnostics used by the refresh gates and paper logging.
    H_top = marked_mask;
    Lambda_beta = sum(eta2(marked_mask)) / (eta2_total + SAB_eps);
    if eta2_total <= SAB_eps
        % A vanishing indicator is not a localized support and must not
        % trigger a refresh through the effective-support gate.
        q_res = zeros(Nsab,1);
        n_eff = Nvalid;
        chi_eff = 1;
    else
        q_res = eta2 ./ eta2_total;
        n_eff = 1 / sum(q_res.^2);
        chi_eff = n_eff / Nvalid;
    end

    H_mat = reshape(H_top,Ny_diag,Nx_diag);
    valid_mat = reshape(valid_mask,Ny_diag,Nx_diag);
    kernel8 = [1 1 1; 1 0 1; 1 1 1];

    neighbor_total = conv2(double(valid_mat),kernel8,'same');
    neighbor_high = conv2(double(H_mat),kernel8,'same');

    local_ratio_mat = neighbor_high ./ (neighbor_total + SAB_eps);

    if any(H_mat(:))
        C_loc = mean(local_ratio_mat(H_mat));
    else
        C_loc = 0;
    end

    G_loc = (chi_eff <= cfg.tau_eff);

    % ============================================================
    % Structure mask: the single global Dörfler set.  No area-only
    % connected-component deletion is applied.  In particular, a small but
    % residual-important corner/interface component remains available to the
    % regional SFR allocation.
    % ============================================================
    structure_mask_raw = marked_mask;
    marked_values = S_hat(marked_mask);
    if isempty(marked_values)
        tau_S = inf;
    else
        tau_S = min(marked_values);
    end

    % Label marked components over the complete global domain.  The loop is
    % retained for backward-compatible bookkeeping; with [1,1] it has one
    % pass and cannot introduce an artificial spatial split.
    label_vec_raw = zeros(Nsab,1);
    num_region_raw = 0;
    partition_component_ids = cell(num_partition,1);
    for pp = 1:num_partition
        mask_pp = reshape(structure_mask_raw & (partition_id == pp), ...
            Ny_diag,Nx_diag);
        [lab_pp,nloc_pp] = YY_utils('label_components_8conn',mask_pp);
        ids_pp = zeros(nloc_pp,1);
        lab_vec_pp = lab_pp(:);
        for jj = 1:nloc_pp
            num_region_raw = num_region_raw + 1;
            ids_pp(jj) = num_region_raw;
            label_vec_raw(lab_vec_pp == jj) = num_region_raw;
        end
        partition_component_ids{pp} = ids_pp;
    end

    region_size = zeros(num_region_raw,1);
    region_mass = zeros(num_region_raw,1);
    region_res_mass = zeros(num_region_raw,1);
    region_centroid = zeros(num_region_raw,2);

    total_struct_mass_raw = sum(density_score(structure_mask_raw)) + SAB_eps;
    % Keep component masses in the same cell-measure units as the global
    % Dörfler indicator mass eta_i=mu_i*S_i.  This makes the connected
    % component filter independent of the diagnostic-grid resolution.
    total_res_mass_all = sum(eta2(valid_mask)) + SAB_eps;

    for kk = 1:num_region_raw
        idx_k = find(label_vec_raw == kk);
        region_size(kk) = numel(idx_k);
        region_mass(kk) = sum(density_score(idx_k));
        region_res_mass(kk) = sum(eta2(idx_k));
        region_centroid(kk,:) = mean(sab_points(idx_k,:),1);
    end

    % Retain components by their global area-weighted structure-score mass,
    % not by area alone.  This prevents low-score noise from becoming
    % artificial structures.
    if isfield(cfg,'tau_conn') && ~isempty(cfg.tau_conn)
        tau_component_mass = cfg.tau_conn;
    else
        tau_component_mass = 0.01;
    end
    component_partition = zeros(num_region_raw,1);
    for pp = 1:num_partition
        ids_pp_raw = partition_component_ids{pp};
        for jj = 1:numel(ids_pp_raw)
            idx_jj = find(label_vec_raw == ids_pp_raw(jj),1,'first');
            if ~isempty(idx_jj), component_partition(ids_pp_raw(jj)) = pp; end
        end
    end
    component_res_ratio = region_res_mass / total_res_mass_all;
    keep_region = find(component_res_ratio >= tau_component_mass);
    keep_region = sort(keep_region(:));
    num_region = numel(keep_region);

    old_to_new = zeros(num_region_raw,1);
    old_to_new(keep_region) = (1:num_region).';
    component_id = zeros(Nsab,1);
    marked_raw = label_vec_raw > 0;
    component_id(marked_raw) = old_to_new(label_vec_raw(marked_raw));
    for pp = 1:num_partition
        ids_pp_raw = partition_component_ids{pp};
        partition_component_ids{pp} = old_to_new(ids_pp_raw);
        partition_component_ids{pp} = partition_component_ids{pp}( ...
            partition_component_ids{pp} > 0);
    end

    % There is no artificial macro decomposition. Multiple connected
    % components, if present, are residual-supported components in this
    % single global SFR region.
    multi_structure_enabled = (num_region > 1);
    partition_num_components = zeros(num_partition,1);
    for pp = 1:num_partition
        partition_num_components(pp) = numel(partition_component_ids{pp});
    end

    structure_mask = (component_id > 0);
    background_mask = valid_mask & ~structure_mask;

    domain_labels = zeros(Nsab,1);
    domain_labels(background_mask) = 1;
    domain_labels(structure_mask) = component_id(structure_mask) + 1;

    % ============================================================
    % Regional neuron budgets and regional gates.
    % ============================================================
    alpha_S = sum(density_score(structure_mask)) / ...
              (sum(density_score) + SAB_eps);

    m_partition_tar = allocate_partition_budgets_local( ...
        ctx.neuron,partition_energy,partition_valid_count);
    partition_alpha_S = zeros(num_partition,1);
    partition_alpha_use = zeros(num_partition,1);
    partition_m_S = zeros(num_partition,1);
    partition_m_B = m_partition_tar(:);
    partition_region_weight = cell(num_partition,1);
    partition_m_each = cell(num_partition,1);
    partition_G = false(num_partition,1);
    partition_chi = zeros(num_partition,1);
    partition_cover = false(num_partition,1);
    component_energy = zeros(num_region,1);
    component_chi = inf(num_region,1);
    component_cover = false(num_region,1);
    component_G = false(num_region,1);

    old_centers = gather(ctx.b_basis.');
    component_mat = reshape(component_id,Ny_diag,Nx_diag);
    partition_mat = reshape(partition_id,Ny_diag,Nx_diag);
    ix = floor((old_centers(:,1)-ctx.a1)/hx) + 1;
    iy = floor((old_centers(:,2)-ctx.a3)/hy) + 1;
    inside_box = old_centers(:,1) >= ctx.a1 & old_centers(:,1) <= ctx.a2 & ...
                 old_centers(:,2) >= ctx.a3 & old_centers(:,2) <= ctx.a4;
    ix = min(max(ix,1),Nx_diag);
    iy = min(max(iy,1),Ny_diag);
    center_component = zeros(size(ix));
    center_partition = zeros(size(ix));
    lin = sub2ind([Ny_diag,Nx_diag],iy(inside_box),ix(inside_box));
    center_component(inside_box) = component_mat(lin);
    center_partition(inside_box) = partition_mat(lin);
    center_component = round(center_component(:));
    center_partition = round(center_partition(:));

    alpha_S_tar_each = zeros(num_region,1);
    alpha_hat_S_each = zeros(num_region,1);
    cover_deficit_each = zeros(num_region,1);
    for pp = 1:num_partition
        idx_pp = find(valid_mask & partition_id == pp);
        if isempty(idx_pp), continue; end
        D_pp = sum(density_score(idx_pp)) + SAB_eps;
        S_pp = idx_pp(structure_mask(idx_pp));
        partition_alpha_S(pp) = sum(density_score(S_pp))/D_pp;
        idx_comp_pp = partition_component_ids{pp};
        nloc_pp = numel(idx_comp_pp);
        ncenter_pp = max(nnz(center_partition == pp),1);
        comp_local = zeros(Nsab,1);
        for jj = 1:nloc_pp
            comp_local(component_id == idx_comp_pp(jj)) = jj;
        end
        ctx_alloc = ctx;
        ctx_alloc.neuron = m_partition_tar(pp);
        [partition_alpha_use(pp),partition_m_S(pp),partition_m_B(pp), ...
            partition_region_weight{pp},partition_m_each{pp}] = ...
            allocate_sab_neurons_local( ...
                ctx_alloc,p_S,comp_local,nloc_pp,partition_alpha_S(pp),cfg);

        for jj = 1:nloc_pp
            gid = idx_comp_pp(jj);
            % Target and achieved coverage are measured in the single global
            % SFR region.
            alpha_S_tar_each(gid) = partition_m_each{pp}(jj) / ...
                max(m_partition_tar(pp),1);
            alpha_hat_S_each(gid) = nnz(center_partition == pp & ...
                center_component == gid) / ncenter_pp;
            cover_deficit_each(gid) = max( ...
                alpha_S_tar_each(gid)-alpha_hat_S_each(gid),0) / ...
                (alpha_S_tar_each(gid)+SAB_eps);
        end

        q_pp = eta2(idx_pp)/(partition_energy(pp)+SAB_eps);
        n_eff_pp = 1/(sum(q_pp.^2)+SAB_eps);
        partition_chi(pp) = n_eff_pp/max(numel(idx_pp),1);
        G_loc_pp = partition_chi(pp) <= cfg.tau_eff;
        if nloc_pp > 0
            cover_pp = max(cover_deficit_each(idx_comp_pp)) >= cfg.tau_cover;
        else
            cover_pp = false;
        end
        partition_cover(pp) = cover_pp;
        if isfield(cfg,'use_cover_gate_for_refresh') && ...
                ~cfg.use_cover_gate_for_refresh
            partition_G(pp) = G_loc_pp && (nloc_pp >= 1);
        else
            partition_G(pp) = G_loc_pp && cover_pp && (nloc_pp >= 1);
        end

        % Apply the refresh decision once more at the level of each
        % Each retained connected component Omega_S^(ell) must still carry
        % its own nonnegligible residual mass.
        for jj = 1:nloc_pp
            gid = idx_comp_pp(jj);
            idx_gid = find(component_id == gid & partition_id == pp);
            if isempty(idx_gid), continue; end
            E_gid = sum(eta2(idx_gid));
            component_energy(gid) = E_gid;
            q_gid = eta2(idx_gid)/(E_gid + SAB_eps);
            n_eff_gid = 1/(sum(q_gid.^2) + SAB_eps);
            component_chi(gid) = n_eff_gid/max(numel(idx_gid),1);
            component_cover(gid) = cover_deficit_each(gid) >= cfg.tau_cover;
            % The global coverage gate has established that a refresh is
            % eligible.  At component level the nonnegligible residual-mass
            % test is the independent SFR decision.
            component_G(gid) = partition_G(pp) && ...
                E_gid/(partition_energy(pp)+SAB_eps) >= tau_component_mass;
        end
    end

    % A component that fails its own gate is handled as background; its
    % feature budget is returned to the same global SFR allocation.
    partition_m_each_alloc = partition_m_each;
    partition_m_S_alloc = partition_m_S;
    partition_m_B_alloc = partition_m_B;
    partition_m_each_eff = cell(num_partition,1);
    partition_m_S_eff = zeros(num_partition,1);
    partition_m_B_eff = zeros(num_partition,1);
    for pp = 1:num_partition
        gids = partition_component_ids{pp}(:);
        mloc = partition_m_each{pp}(:);
        keep_loc = component_G(gids);
        mloc(~keep_loc) = 0;
        partition_m_each_eff{pp} = mloc;
        partition_m_S_eff(pp) = sum(mloc);
        partition_m_B_eff(pp) = m_partition_tar(pp) - partition_m_S_eff(pp);
    end
    partition_m_each = partition_m_each_eff;
    partition_m_S = partition_m_S_eff;
    partition_m_B = partition_m_B_eff;
    partition_alpha_use = partition_m_S ./ max(m_partition_tar,1);

    if isempty(cover_deficit_each)
        cover_deficit_max = 0;
    else
        cover_deficit_max = max(cover_deficit_each);
    end
    G_cover = any(partition_cover);
    G_SAB = any(partition_G);
    alpha_use = sum(partition_alpha_use .* m_partition_tar) / max(ctx.neuron,1);
    m_S_tar = sum(partition_m_S);
    m_B_tar = sum(partition_m_B);
    region_weight = vertcat(partition_region_weight{:});
    m_each_tar = vertcat(partition_m_each{:});

    fprintf('\n========== multi-connected structure region detection ==========\n');
    fprintf('raw structure connected component count num_region_raw = %d\n',num_region_raw);
    fprintf('effective structure region count (global score-mass filter) num_region = %d\n',num_region);
    fprintf('global component score-mass floor tau_conn = %.4f\n',tau_component_mass);
    fprintf('global residual/solution-gradient weights = (%.3f, %.3f)\n', ...
        theta_res,theta_grad);
    fprintf('global Dorfler marking: lambda=%.4f, theta=%.4f, ok=%d, marked_cells=%d\n', ...
        global_dorfler_lambda,cfg.theta_dorfler,global_dorfler_ok,nnz(marked_mask));
    fprintf('single global SFR region enabled = 1\n');
    fprintf('global partition bookkeeping count = %d, partition grid = [%d,%d]\n', ...
        num_partition,partition_grid(1),partition_grid(2));
    for pp = 1:num_partition
        fprintf(['global region %d: valid_cells=%d, indicator_mass=%.4e, ', ...
                 'global-marked fraction=%.4f, connected_components=%d, ', ...
                 'chi=%.4f, cover=%d, component_SFR=%d, ', ...
                 'm=%d, m_S=%d, m_B=%d, G_SFR=%d\n'], ...
            pp,partition_valid_count(pp),partition_energy(pp), ...
            partition_lambda(pp),partition_num_components(pp), ...
            partition_chi(pp),partition_cover(pp), ...
            nnz(component_G(partition_component_ids{pp})), ...
            m_partition_tar(pp),partition_m_S(pp), ...
            partition_m_B(pp),partition_G(pp));
    end

    for kk = 1:num_region
        raw_id = keep_region(kk);
        fprintf('Region %d: size=%d, struct_mass_ratio=%.4f, res_mass_ratio=%.4f, centroid=(%.4f, %.4f)\n', ...
            kk,region_size(raw_id),region_mass(raw_id)/total_struct_mass_raw, ...
            region_res_mass(raw_id)/total_res_mass_all, ...
            region_centroid(raw_id,1),region_centroid(raw_id,2));
    end

    fprintf('effective support ratio chi_eff         = %.4f\n',chi_eff);
    fprintf('structure region threshold tau_S             = %.4f\n',tau_S);
    fprintf('structure probability mass alpha_S       = %.4f\n',alpha_S);
    fprintf('target total allocation ratio alpha_use      = %.4f\n',alpha_use);
    fprintf('target structure neurons m_S            = %d\n',m_S_tar);
    fprintf('target background neurons m_B            = %d\n',m_B_tar);

    for kk = 1:num_region
        fprintf(['Region %d coverage: alpha_tar=%.4f, ', ...
                 'alpha_hat=%.4f, relative_deficit=%.4f\n'], ...
            kk,alpha_S_tar_each(kk),alpha_hat_S_each(kk), ...
            cover_deficit_each(kk));
    end

    fprintf('max relative coverage deficit D_cover      = %.4f\n',cover_deficit_max);
    fprintf('Local gate: chi_eff <= tau_eff = %d\n',G_loc);
    fprintf('Cover gate = %d\n',G_cover);
    fprintf('final SFR refresh suggested SFR refresh gate = %d\n',G_SAB);

    if cfg.plot_figures
        S_mask_mat = reshape(structure_mask,Ny_diag,Nx_diag);
        figure;
        contourf(xx1,xx2,reshape(S_hat,Ny_diag,Nx_diag),40,'LineColor','none');
        shading interp; colormap jet; colorbar;
        title('automatic structure indicators');
        xlabel('x'); ylabel('y');

        figure;
        imagesc(xx1,xx2,reshape(double(H_top),Ny_diag,Nx_diag));
        set(gca,'YDir','normal'); colormap(gray); colorbar;
        title('Global residual--solution-gradient Dörfler set');
        xlabel('x'); ylabel('y');

        figure;
        imagesc(xx1,xx2,reshape(domain_labels,Ny_diag,Nx_diag));
        set(gca,'YDir','normal');
        max_label = max(domain_labels);
        cmap_region = jet(max_label);
        cmap_region(1,:) = [0.85 0.85 0.85];
        colormap(cmap_region); colorbar;
        title('SAB multi-connected candidate region partition');
        xlabel('x'); ylabel('y');
    end

    diagS.G_SAB = G_SAB;
    diagS.theta_res = theta_res;
    diagS.theta_grad = theta_grad;
    diagS.etaR = etaR;
    diagS.etaG = etaG;
    diagS.eta2 = eta2;
    diagS.s_res_hat = s_res_hat;
    diagS.s_grad_hat = s_grad_hat;
    diagS.structure_score = S_score;
    diagS.global_dorfler_lambda = global_dorfler_lambda;
    diagS.global_dorfler_ok = global_dorfler_ok;
    diagS.global_marked_count = nnz(marked_mask);
    diagS.p_S = p_S;
    diagS.alpha_S = alpha_S;
    diagS.alpha_use = alpha_use;
    diagS.m_S_target = m_S_tar;
    diagS.m_B_target = m_B_tar;
    diagS.region_weight = region_weight;
    diagS.m_each_target = m_each_tar;
    diagS.alpha_S_tar_each = alpha_S_tar_each;
    diagS.alpha_hat_S_each = alpha_hat_S_each;
    diagS.cover_deficit_each = cover_deficit_each;
    diagS.cover_deficit_max = cover_deficit_max;
    diagS.G_cover = G_cover;
    diagS.structure_mask = structure_mask;
    diagS.background_mask = background_mask;
    diagS.component_id = component_id;
    diagS.num_region = num_region;
    diagS.sab_points = sab_points;
    diagS.quad_points = quad_points;
    diagS.quad_valid = quad_valid;
    diagS.n_quad = 4;
    diagS.quad_rule = 'tensor-product 2x2 Gauss-Legendre';
    diagS.domain_labels = domain_labels;
    diagS.xx1 = xx1;
    diagS.xx2 = xx2;
    diagS.diag_N = diag_N;
    diagS.h = [hx,hy];
    diagS.bounds = [ctx.a1,ctx.a2,ctx.a3,ctx.a4];
    diagS.valid_mask = valid_mask;
    diagS.Nvalid = Nvalid;
    diagS.SAB_eps = SAB_eps;
    diagS.tau_S = tau_S;
    diagS.partition_grid = partition_grid;
    diagS.partition_id = partition_id;
    diagS.partition_raw_id = partition_raw_id;
    diagS.num_partition = num_partition;
    diagS.partition_energy = partition_energy;
    diagS.partition_marked_energy = partition_marked_energy;
    diagS.partition_lambda = partition_lambda;
    diagS.partition_dorfler_ok = partition_dorfler_ok;
    diagS.partition_valid_count = partition_valid_count;
    diagS.partition_target = m_partition_tar;
    diagS.partition_m_S = partition_m_S;
    diagS.partition_m_B = partition_m_B;
    diagS.partition_m_S_alloc = partition_m_S_alloc;
    diagS.partition_m_B_alloc = partition_m_B_alloc;
    diagS.partition_m_each_alloc = partition_m_each_alloc;
    diagS.partition_alpha_S = partition_alpha_S;
    diagS.partition_alpha_use = partition_alpha_use;
    diagS.partition_G = partition_G;
    diagS.partition_chi = partition_chi;
    diagS.partition_cover = partition_cover;
    diagS.partition_component_ids = partition_component_ids;
    diagS.partition_num_components = partition_num_components;
    diagS.multi_structure_enabled = multi_structure_enabled;
    diagS.component_energy = component_energy;
    diagS.component_res_ratio = component_res_ratio;
    diagS.component_chi = component_chi;
    diagS.component_cover = component_cover;
    diagS.component_G = component_G;
    diagS.partition_region_weight = partition_region_weight;
    diagS.partition_m_each = partition_m_each;
end

%% ============================================================
% SFR Step 2/3: data-driven adaptive bias refresh
%% ============================================================
function ctx = sab_refresh_local(ctx,diagS,cfg)

    cfg = fill_sab_ref_cfg_local(cfg,ctx);

    SAB_eps = diagS.SAB_eps;
    p_S = diagS.p_S;
    sab_points = diagS.sab_points;
    component_id = diagS.component_id;
    num_region = diagS.num_region;
    structure_mask = diagS.structure_mask;
    background_mask = diagS.background_mask;
    partition_id = diagS.partition_id;
    num_partition = diagS.num_partition;
    partition_target = diagS.partition_target(:);
    partition_m_S = diagS.partition_m_S(:);
    partition_m_B = diagS.partition_m_B(:);
    partition_G = diagS.partition_G(:);
    partition_component_ids = diagS.partition_component_ids;
    partition_m_each = diagS.partition_m_each;

    fprintf('\n========== SFR global Dörfler--single-domain scale optimization ==========\n');
    fprintf('total neurons neuron = %d, global partitions = %d\n',ctx.neuron,num_partition);
    fprintf('global budget: [%s]\n',num2str(partition_target.','%d '));

    w_keep = ctx.w;
    b_keep = ctx.b;
    b_basis_keep = ctx.b_basis;

    if isfield(ctx,'use_orth_basis') && ctx.use_orth_basis == 1
        old_is_orth = true;
        T_keep = ctx.T_current;
        gamma_keep = ctx.gamma_current;
        weight_keep = [];
    else
        old_is_orth = false;
        weight_keep = ctx.weight;
        T_keep = [];
        gamma_keep = [];
    end
    term_old = struct('w',w_keep,'b',b_keep);

    W_S_all = [];
    b_S_all = [];
    A_S_all = [];
    W_B_all = [];
    b_B_all = [];
    A_B_all = [];

    r_opt_region = zeros(2,num_region);
    MSE_region = zeros(num_region,1);
    r_opt_B = zeros(2,num_partition);
    MSE_B = NaN(num_partition,1);

    fprintf('\n--- optimizing global structure/background scales ---\n');
    for pp = 1:num_partition
        idx_pp = find(diagS.valid_mask & partition_id == pp);
        if isempty(idx_pp) || partition_target(pp) <= 0
            continue;
        end

        % A region whose local gate is false is treated as background for
        % this refresh.  It receives its own regional budget but does not
        % create a structure subproblem.
        if partition_G(pp)
            comp_ids_pp = partition_component_ids{pp};
            m_each_pp = partition_m_each{pp}(:);
        else
            comp_ids_pp = zeros(0,1);
            m_each_pp = zeros(0,1);
        end

        for jj = 1:numel(comp_ids_pp)
            gid = comp_ids_pp(jj);
            rng(1000*pp + jj);

            % A structure component is owned by the single global region.
            % Keep this restriction explicit at refresh time, rather than
            % relying only on the diagnostic labels.
            idx_k_all = find(component_id == gid & partition_id == pp);
            m_k = m_each_pp(jj);
            if isempty(idx_k_all) || m_k <= 0, continue; end

            p_k = gather(p_S(idx_k_all));
            p_k = p_k(:) / (sum(p_k) + SAB_eps);

            nK_train = regional_sample_cap_local( ...
                m_k,length(idx_k_all),cfg,'max_train_points_S');
            perm_k = randperm(length(idx_k_all),nK_train);
            idx_k_train = idx_k_all(perm_k);
            sub_x_k = gather(sab_points(idx_k_train,:));

            if old_is_orth
                mb_k = BasisFunction('tanh',term_old,sub_x_k,0,0) * T_keep * gamma_keep;
            else
                mb_k = BasisFunction('tanh',term_old,sub_x_k,0,0) * weight_keep;
                mb_k = gather(mb_k);
            end

            idx_local_k = YY_utils('sample_from_prob',p_k,m_k);
            idx_k_center = idx_k_all(idx_local_k);

            A_k_raw = gather(sab_points(idx_k_center,:).');
            A_k = jitter_project_centers_local(A_k_raw,ctx,diagS,cfg,gid);
            A_k = enforce_macro_anchor_local(A_k,A_k_raw,pp,ctx,diagS);

            Z_k = 2*rand(2,m_k) - 1;

            [r_opt_k,MSE_k] = optimize_r_adam_data_local( ...
                sub_x_k,mb_k,Z_k,A_k,cfg.lambda_reg,ctx.r,cfg.adam_opts);

            W_k = r_opt_k .* Z_k;
            b_k = -sum(A_k .* W_k,1);

            W_S_all = [W_S_all,W_k];
            b_S_all = [b_S_all,b_k];
            A_S_all = [A_S_all,A_k];

            r_opt_region(:,gid) = r_opt_k;
            MSE_region(gid) = MSE_k;

            fprintf('global structure component %d: m=%d, r=(%.6f, %.6f), MSE=%.6e\n', ...
                gid,m_k,r_opt_k(1),r_opt_k(2),MSE_k);
        end

        if partition_G(pp)
            idx_B_all = find(background_mask & partition_id == pp);
            m_B_pp = partition_m_B(pp);
        else
            idx_B_all = idx_pp;
            m_B_pp = partition_target(pp);
        end

        fprintf('global background: m=%d, available cells=%d\n', ...
            m_B_pp,numel(idx_B_all));

        if m_B_pp > 0 && ~isempty(idx_B_all)
            rng(2000+pp);
            nB_train = regional_sample_cap_local( ...
                m_B_pp,length(idx_B_all),cfg,'max_train_points_B');
            perm_B = randperm(length(idx_B_all),nB_train);
            idx_B_train = idx_B_all(perm_B);
            sub_x_B = gather(sab_points(idx_B_train,:));

            if old_is_orth
                mb_B = BasisFunction('tanh',term_old,sub_x_B,0,0) * T_keep * gamma_keep;
            else
                mb_B = BasisFunction('tanh',term_old,sub_x_B,0,0) * weight_keep;
                mb_B = gather(mb_B);
            end

            p_B_cond = gather(p_S(idx_B_all));
            p_B_cond = p_B_cond(:) / (sum(p_B_cond) + SAB_eps);
            if sum(p_B_cond) <= SAB_eps
                p_B_cond = ones(numel(idx_B_all),1)/numel(idx_B_all);
            end
            idx_local_B = YY_utils('sample_from_prob',p_B_cond,m_B_pp);
            idx_B_center = idx_B_all(idx_local_B);

            A_B_raw = gather(sab_points(idx_B_center,:).');
            A_B = jitter_project_centers_local(A_B_raw,ctx,diagS,cfg,2000+pp);
            A_B = enforce_macro_anchor_local(A_B,A_B_raw,pp,ctx,diagS);

            Z_B = 2*rand(2,m_B_pp) - 1;
            [r_opt_B_pp,MSE_B_pp] = optimize_r_adam_data_local( ...
                sub_x_B,mb_B,Z_B,A_B,cfg.lambda_reg,ctx.r,cfg.adam_opts);

            W_B = r_opt_B_pp .* Z_B;
            b_B = -sum(A_B .* W_B,1);
            W_B_all = [W_B_all,W_B];
            b_B_all = [b_B_all,b_B];
            A_B_all = [A_B_all,A_B];
            r_opt_B(:,pp) = r_opt_B_pp;
            MSE_B(pp) = MSE_B_pp;
        else
            r_opt_B(:,pp) = ctx.r(:);
        end
    end

    w_cand = [W_S_all,W_B_all];
    b_cand = [b_S_all,b_B_all];
    b_basis_cand = [A_S_all,A_B_all];

    term_cand = struct('w',w_cand,'b',b_cand);

     op = get_problem_op_local(ctx);
    A_int = BasisOperator('tanh',term_cand,ctx.x_point,op);

    Phi_bd = BasisFunction('tanh',term_cand,ctx.x_0,0,0);
    B_n = neumann_flux_matrix_local('tanh',term_cand,ctx,ctx.eta_neumann);

    B_cpu = gather([A_int; ctx.eta0*Phi_bd; B_n]);
    y_cpu = gather([ctx.right; ctx.eta0*ctx.y0_exact; zeros(size(B_n,1),1)]);

    [U_r,V_r,s_r,r_keep] = YY_utils('svd_truncate',B_cpu,ctx.orth_cfg);
    T_cand = bsxfun(@rdivide,V_r,s_r.');
    if isfield(cfg,'newton') && ~isempty(cfg.newton)
        % Nonlinear problems: warm start via bestInitSVD (value projection vs
        % Newton-linearized solve, pick the smaller residual), matching the
        % legacy KDV_Newton_module 'best_kdv_init_svd'.
        if old_is_orth
            solEvalOldC = @(ord,dir,X) BasisFunction('tanh',term_old,X,ord,dir) * (T_keep * gamma_keep);
        else
            solEvalOldC = @(ord,dir,X) BasisFunction('tanh',term_old,X,ord,dir) * weight_keep;
        end
        basisEvalRawC = @(ord,dir,X) BasisFunction('tanh',term_cand,X,ord,dir);
        consC = [];
        try
            consC = ProblemFunction('constraints', [], ctx.problem);
        catch
            consC = [];
        end
        etaS = ctx.eta0;
        if isfield(cfg,'eta_val') && ~isempty(cfg.eta_val), etaS = cfg.eta_val; end
        [coef_rawC, initInfoC] = bestInitSVD(ctx.problem, basisEvalRawC, solEvalOldC, consC, ...
            ctx.x_point, etaS, cfg.newton, ctx.GPU, ctx.QR);
        fprintf('SAB warm start: use=%s, rel_value=%.4e, rel_linearized=%.4e\n', ...
            initInfoC.method, initInfoC.rel_value, initInfoC.rel_linearized);
        gamma_cand = s_r .* (V_r' * coef_rawC);
    else
        % Linear problems: original linearized least-squares warm start.
        gamma_cand = U_r' * y_cpu;
    end
    r_cand_W = y_cpu - U_r * gamma_cand;

    % Nonlinear problems: refine the candidate basis with full Newton.
    refined_new = false;
    if isfield(cfg,'newton_refine') && cfg.newton_refine
        consN = get_problem_constraints_local(ctx);
        if ~isempty(consN)
            basisEvalC = @(ord,dir,X) BasisFunction('tanh',term_cand,X,ord,dir) * T_cand;
            [gamma_cand, nwS] = NewtonSolver( ...
                ctx.problem, basisEvalC, ctx.x_point, consN, gamma_cand, cfg.newton);
            fprintf('  [Newton] SFR candidate refine: rel_res=%.4e, iter=%d\n', nwS.rel_res, nwS.iter);
            refined_new = true;
        end
    end

    fprintf('\n========== SVD orthogonal compression of candidate basis space ==========\n');
    fprintf('raw candidate basis size = %d\n',size(B_cpu,2));
    fprintf('retained orthogonal rank r_keep = %d\n',r_keep);
    fprintf('candidate training residual norm = %.6e\n',norm(r_cand_W));

    % Compare old and candidate configurations on exactly the same
    % validation points and boundary samples.
    rng(2025);
    x_val = YY_utils('get_inner_point',ctx.a1,ctx.a2,ctx.a3,ctx.a4,cfg.val_N);
    f_val = problem_rhs_local(ctx,x_val);

    x0_val = boundary_points_for_validation_local(ctx,cfg.b_val);

    g_val = problem_boundary_local(ctx,x0_val);

     op = get_problem_op_local(ctx);

    Lphi_old = BasisOperator('tanh',term_old,x_val,op);
    Phi_bd_old = BasisFunction('tanh',term_old,x0_val,0,0);
    B_n_old = neumann_flux_matrix_local('tanh',term_old,ctx,ctx.eta_neumann);
    
    if old_is_orth
        Lu_old  = Lphi_old * T_keep * gamma_keep;
        ubd_old = Phi_bd_old * T_keep * gamma_keep;
    else
        Lu_old  = Lphi_old * weight_keep;
        ubd_old = Phi_bd_old * weight_keep;
    end
    if old_is_orth
        if_n_old = B_n_old * T_keep * gamma_keep;
    else
        if_n_old = B_n_old * weight_keep;
    end
    R_old = [Lu_old - f_val; cfg.eta_val*(ubd_old - g_val); if_n_old];
    rhs_old = [f_val; cfg.eta_val*g_val; zeros(size(if_n_old,1),1)];
    E_pred_old = norm(R_old) / (norm(rhs_old) + SAB_eps);
    
    
    Lphi_new = BasisOperator('tanh',term_cand,x_val,op);
    Phi_bd_new = BasisFunction('tanh',term_cand,x0_val,0,0);
    B_n_new = neumann_flux_matrix_local('tanh',term_cand,ctx,ctx.eta_neumann);
    
    Lu_new  = Lphi_new * T_cand * gamma_cand;
    ubd_new = Phi_bd_new * T_cand * gamma_cand;
    if_n_new = B_n_new * T_cand * gamma_cand;
    R_new = [Lu_new - f_val; cfg.eta_val*(ubd_new - g_val); if_n_new];
    rhs_new = [f_val; cfg.eta_val*g_val; zeros(size(if_n_new,1),1)];
    E_pred_new = norm(R_new) / (norm(rhs_new) + SAB_eps);

    if refined_new
        % Compare both bases using the full nonlinear residual.
        if old_is_orth
            basisEvalOld = @(ord,dir,X) BasisFunction('tanh',term_old,X,ord,dir) * T_keep;
            coefOld = gamma_keep;
        else
            basisEvalOld = @(ord,dir,X) BasisFunction('tanh',term_old,X,ord,dir);
            coefOld = weight_keep;
        end
        basisEvalNew = @(ord,dir,X) BasisFunction('tanh',term_cand,X,ord,dir) * T_cand;
        E_pred_old = nonlinear_pred_local(ctx.problem, basisEvalOld, coefOld, ...
            x_val, x0_val, f_val, g_val, cfg.eta_val);
        E_pred_new = nonlinear_pred_local(ctx.problem, basisEvalNew, gamma_cand, ...
            x_val, x0_val, f_val, g_val, cfg.eta_val);
    end

    fprintf('\n========== SFR refresh acceptance check: prediction residual ==========\n');
    fprintf('old basis prediction residual E_pred_old = %.6e\n',E_pred_old);
    fprintf('new orthogonal basis prediction residual E_pred_new = %.6e\n',E_pred_new);

    accept_threshold = (1 - cfg.eta_accept) * E_pred_old;
    accept_margin = E_pred_old - E_pred_new;
    accept_refresh = E_pred_new <= accept_threshold;
    fprintf('acceptance threshold = %.6e, empirical margin = %.6e\n', ...
        accept_threshold,accept_margin);

    if accept_refresh
        fprintf('Accept SFR refresh: empirical residual is non-increasing.\n');

        ctx.w = w_cand;
        ctx.b = b_cand;
        ctx.b_basis = b_basis_cand;

        ctx.T_current = T_cand;
        ctx.gamma_current = gamma_cand;

        ctx.Q_current = U_r;
        ctx.r_current_W = r_cand_W;
        ctx.y_current_W = y_cpu;

        ctx.use_orth_basis = 1;
        ctx.SAB_accept = 1;
        ctx.SAB_accept_reason = 'empirical residual non-increasing';
    else
        fprintf('Reject SFR refresh: empirical residual increased, rolled back to old basis.\n');

        ctx.w = w_keep;
        ctx.b = b_keep;
        ctx.b_basis = b_basis_keep;

        if old_is_orth
            ctx.T_current = T_keep;
            ctx.gamma_current = gamma_keep;
            ctx.use_orth_basis = 1;
        else
            ctx.weight = weight_keep;
            if isfield(ctx,'T_current'), ctx.T_current = []; end
            if isfield(ctx,'gamma_current'), ctx.gamma_current = []; end
            ctx.use_orth_basis = 0;
        end

        ctx.SAB_accept = 0;
        ctx.SAB_accept_reason = 'empirical residual increased; rollback';
    end

    % Persist an acceptance certificate for run logs and result files.
    ctx.SAB_E_pred_old = E_pred_old;
    ctx.SAB_E_pred_new = E_pred_new;
    ctx.SAB_accept_threshold = accept_threshold;
    ctx.SAB_accept_margin = accept_margin;
    ctx.SAB_validation_count = size(x_val,1) + size(x0_val,1);
    ctx.SAB_validation_same_samples = true;

    ctx.r_opt_region = r_opt_region;
    ctx.r_opt_B = r_opt_B;
    ctx.MSE_region = MSE_region;
    ctx.MSE_B = MSE_B;

    if cfg.plot_figures
        term_eff = struct('w',ctx.w,'b',ctx.b);

        if isfield(ctx,'use_orth_basis') && ctx.use_orth_basis == 1
            pred = BasisFunction('tanh',term_eff,ctx.pp_points,0,0) * ctx.T_current * ctx.gamma_current;
        else
            pred = BasisFunction('tanh',term_eff,ctx.pp_points,0,0) * ctx.weight;
        end

        % Reference data are used here only for an optional a posteriori plot.
        YY_utils('plot_abs_error',ctx.xp1,ctx.xp2,ctx.pred_out_ex,pred,'Final Absolute Error after SAB');
    end
end

%% ============================================================
% generic nonlinear helpers
%% ============================================================
function cons = get_problem_constraints_local(ctx)
    cons = [];
    try
        cons = ProblemFunction('constraints', [], ctx.problem);
    catch
        cons = [];
    end
end

function E = nonlinear_pred_local(problem, basisEval, coef, x_val, x0_val, f_val, g_val, eta_val)
%NONLINEAR_PRED_LOCAL
% Relative nonlinear prediction residual of a trial solution.
%
% When the problem provides explicit boundary/initial constraints
% (ProblemFunction('constraints')), those are used (periodic differences,
% ux/uxx and initial values) so that time-evolution problems such as KdV
% are not penalized on the final-time line (t = t_max), which is not a
% boundary condition.  Otherwise the legacy rectangle-boundary penalty is
% kept for backward compatibility.
    u    = basisEval(0, 0, x_val) * coef;
    ux   = basisEval(1, 1, x_val) * coef;
    ut   = basisEval(1, 2, x_val) * coef;
    uxx  = basisEval(2, 1, x_val) * coef;
    uxxx = basisEval(3, 1, x_val) * coef;
    sol  = struct('u', u, 'ux', ux, 'ut', ut, 'uxx', uxx, 'uxxx', uxxx);
    F    = ProblemFunction('residual', x_val, problem, sol) - f_val;

    cons = [];
    try
        cons = ProblemFunction('constraints', [], problem);
    catch
        cons = [];
    end

    if isempty(cons)
        % Legacy fallback: rectangle-boundary penalty.
        ubd = basisEval(0, 0, x0_val) * coef;
        R   = [F; eta_val*(ubd - g_val)];
        rhs = [f_val; eta_val*g_val];
        E   = norm(R) / (norm(rhs) + 1e-12);
        return;
    end

    X = cons.X;
    types = cons.types;
    target = cons.target(:);
    eta = eta_val;

    r_c = [];
    tgt_c = [];

    idxV   = strcmpi(types, 'value');
    idxUx  = strcmpi(types, 'ux');
    idxUxx = strcmpi(types, 'uxx');
    if any(idxV)
        Bv = basisEval(0, 0, X(idxV, :));
        r_c = [r_c; Bv*coef - target(idxV)]; %#ok<AGROW>
        tgt_c = [tgt_c; target(idxV)]; %#ok<AGROW>
    end
    if any(idxUx)
        Bu = basisEval(1, 1, X(idxUx, :));
        r_c = [r_c; Bu*coef - target(idxUx)]; %#ok<AGROW>
        tgt_c = [tgt_c; target(idxUx)]; %#ok<AGROW>
    end
    if any(idxUxx)
        Buu = basisEval(2, 1, X(idxUxx, :));
        r_c = [r_c; Buu*coef - target(idxUxx)]; %#ok<AGROW>
        tgt_c = [tgt_c; target(idxUxx)]; %#ok<AGROW>
    end

    % Periodic differences: u(left)-u(right)=0, etc. (target is 0).
    idxPu   = strcmpi(types, 'periodic_u');
    idxPux  = strcmpi(types, 'periodic_ux');
    idxPuxx = strcmpi(types, 'periodic_uxx');
    if any(idxPu)
        Bpu = periodic_diff_pred_local(basisEval, X, idxPu, 0);
        r_c = [r_c; Bpu*coef]; %#ok<AGROW>
        tgt_c = [tgt_c; zeros(size(Bpu,1),1)]; %#ok<AGROW>
    end
    if any(idxPux)
        Bpux = periodic_diff_pred_local(basisEval, X, idxPux, 1);
        r_c = [r_c; Bpux*coef]; %#ok<AGROW>
        tgt_c = [tgt_c; zeros(size(Bpux,1),1)]; %#ok<AGROW>
    end
    if any(idxPuxx)
        Bpuxx = periodic_diff_pred_local(basisEval, X, idxPuxx, 2);
        r_c = [r_c; Bpuxx*coef]; %#ok<AGROW>
        tgt_c = [tgt_c; zeros(size(Bpuxx,1),1)]; %#ok<AGROW>
    end

    R   = [F; eta*r_c];
    rhs = [f_val; eta*tgt_c];
    E   = norm(R) / (norm(rhs) + 1e-12);
end

function B = periodic_diff_pred_local(basisEval, X, idx, ord)
%PERIODIC_DIFF_PRED_LOCAL left-minus-right difference of a periodic constraint.
    seg = find(idx);
    nP = numel(seg) / 2;
    XL = X(seg(1:nP), :);
    XR = X(seg(nP+1:end), :);
    B = basisEval(ord, 1, XL) - basisEval(ord, 1, XR);
end

%% ============================================================
% SFR data-fitting r training helper
%% ============================================================
function [r_opt,MSE] = optimize_r_adam_data_local(x_point,mb,Z,A,lambda,r0,opts)
    eta = opts.eta;
    beta1 = 0.9;
    beta2 = 0.999;
    eps_adam = opts.eps_adam;
    if isfield(opts,'clip_min') && ~isempty(opts.clip_min)
        clip_min = opts.clip_min(:);
    else
        clip_min = -inf(2,1);
    end
    if isfield(opts,'clip_max') && ~isempty(opts.clip_max)
        clip_max = opts.clip_max(:);
    else
        clip_max = inf(2,1);
    end
    maxit = opts.maxit;

    r = r0(:);
    m = zeros(2,1);
    v = zeros(2,1);

    N = size(x_point,1);
    K = size(Z,2);

    fprintf('==== Adam Optimization (2D tanh MSE Loss) ====\n');

    for it = 1:maxit
        W = diag(r) * Z;
        b = -sum(A .* W,1);
        S = x_point * W + b;

        % ============================================================
        % legacy code used Gaussian: U = exp(-(S.^2));
        % switched to tanh so r training matches the tanh basis actually used later.
        % ============================================================
        U = tanh(S);

        G = U.'*U + lambda*eye(K);
        L = chol(G,'lower');
        alpha = L'\(L\(U.'*mb));
        e = U*alpha - mb;

        MSE = mean(e.^2);

        % d tanh(S) / dS = sech^2(S) = 1 - tanh(S)^2
        V = 1 - U.^2;

        t1 = (Z(1,:).*alpha.');
        t2 = (Z(2,:).*alpha.');

        DUa1 = sum(V .* ((x_point(:,1) - A(1,:)) .* t1),2);
        DUa2 = sum(V .* ((x_point(:,2) - A(2,:)) .* t2),2);

        g_r1 = e.' * DUa1;
        g_r2 = e.' * DUa2;
        g_r = [g_r1; g_r2];

        g_r = g_r / (N * MSE + eps);

        m = beta1*m + (1-beta1)*g_r;
        v = beta2*v + (1-beta2)*(g_r.^2);

        m_hat = m ./ (1 - beta1^it);
        v_hat = v ./ (1 - beta2^it);

        r = r - eta * m_hat ./ (sqrt(v_hat) + eps_adam);
        % Scale parameters remain positive; optional finite bounds are only
        % a legacy compatibility hook and are not part of the SISC rule.
        r = max(r,eps);
        r = min(max(r,clip_min),clip_max);

        if norm(g_r) < 1e-6
            break;
        end
    end

    W = diag(r) * Z;
    b = -sum(A .* W,1);
    U = tanh(x_point*W + b);
    G = U.'*U + lambda*eye(K);
    L = chol(G,'lower');
    alpha = L'\(L\(U.'*mb));
    e = U*alpha - mb;

    MSE = mean(e.^2);
    r_opt = r;

    fprintf('\nAdam-optimized r = [%.6f, %.6f],  MSE = %.6e\n', ...
        r_opt(1),r_opt(2),MSE);
end


function tf = op_coeff_maybe_nonzero_local(a)
    if isa(a,'function_handle')
        tf = true;
    elseif isempty(a)
        tf = false;
    elseif isscalar(a)
        tf = (a ~= 0);
    else
        tf = any(a(:) ~= 0);
    end
end

function Lu = apply_operator_to_solution_values_local(X,u,ux,uxx,uy,uyy,uxy,op)
    N = size(X,1);
    a0 = coef_to_N_local(op.alpha0,X,N);
    a1 = coef_to_N_local(op.alpha1,X,N);
    a2 = coef_to_N_local(op.alpha2,X,N);
    a3 = coef_to_N_local(op.alpha3,X,N);
    a4 = coef_to_N_local(op.alpha4,X,N);
    a5 = coef_to_N_local(op.alpha5,X,N);
    Lu = a0.*u + a1.*ux + a2.*uxx + a3.*uy + a4.*uyy + a5.*uxy;
end

function c = coef_to_N_local(a,X,N)
    if isa(a,'function_handle')
        a = a(X);
    end
    if isempty(a)
        c = zeros(N,1);
    elseif isscalar(a)
        c = a * ones(N,1);
    else
        c = a(:);
        if numel(c) ~= N
            error('operator coefficient length should be N=%d, but got %d.',N,numel(c));
        end
    end
end

function problem = get_problem_local(ctx)
%GET_PROBLEM_LOCAL all PDE information must come from ctx.problem.
    if isfield(ctx,'problem') && ~isempty(ctx.problem)
        problem = ProblemFunction('complete',[],ctx.problem);
    else
        error('ctx.problem does not exist.please main set in: ctx.problem = ProblemFunction(''setup'');');
    end
end

function op = get_problem_op_local(ctx)
    problem = get_problem_local(ctx);
    if isfield(problem,'op') && ~isempty(problem.op)
        op = problem.op;
    else
        error('ctx.problem.op does not exist.please ProblemFunction(''setup'') define inoperator.');
    end
end

function f = problem_rhs_local(ctx,X)
    f = ProblemFunction('rhs',X,get_problem_local(ctx));
end

function B = neumann_flux_matrix_local(family,term,ctx,eta)
%NEUMANN_FLUX_MATRIX_LOCAL Weighted natural-boundary flux rows.
    info = [];
    if isfield(ctx,'neumann_info'), info = ctx.neumann_info; end
    if isempty(info) || ~isstruct(info) || ~isfield(info,'X') || isempty(info.X)
        B = zeros(0,size(term.w,2));
        return;
    end
    X = info.X;
    phix = BasisFunction(family,term,X,1,1);
    phiy = BasisFunction(family,term,X,1,2);
    B = eta * info.coefficient(:) .* ...
        (info.normal(:,1).*phix + info.normal(:,2).*phiy);
end

function g = problem_boundary_local(ctx,X)
    g = ProblemFunction('boundary',X,get_problem_local(ctx));
end

function Xb = boundary_points_for_validation_local(ctx,n)
% Use problem-defined Dirichlet points when a problem has mixed boundaries.
    try
        Xb = ProblemFunction('boundary_points',n,get_problem_local(ctx));
        if ~isempty(Xb), return; end
    catch
    end
    x_line = linspace(ctx.a1,ctx.a2,n)';
    y_line = linspace(ctx.a3,ctx.a4,n)';
    Xb = [x_line,ctx.a3*ones(size(x_line));
          x_line,ctx.a4*ones(size(x_line));
          ctx.a1*ones(n,1),y_line;
          ctx.a2*ones(n,1),y_line];
end



function [partition_id,num_partition,partition_raw_id] = ...
    build_macro_partition_local(X,valid_mask,ctx,partition_grid)
%BUILD_MACRO_PARTITION_LOCAL Backward-compatible ownership map.  The SISC
% configurations use a single [1,1] region; invalid cells remain ignored.
    if nargin < 4 || isempty(partition_grid), partition_grid = [1,1]; end
    partition_grid = double(partition_grid(:).');
    if numel(partition_grid) ~= 2 || any(~isfinite(partition_grid)) || ...
            any(partition_grid < 1) || any(partition_grid ~= round(partition_grid))
        error('cfg.partition_grid must contain two positive integers.');
    end
    nxp = partition_grid(1);
    nyp = partition_grid(2);
    tx = (X(:,1)-ctx.a1) / max(ctx.a2-ctx.a1,eps);
    ty = (X(:,2)-ctx.a3) / max(ctx.a4-ctx.a3,eps);
    ix = min(max(floor(nxp*tx)+1,1),nxp);
    iy = min(max(floor(nyp*ty)+1,1),nyp);
    raw = sub2ind([nyp,nxp],iy,ix);
    raw(~valid_mask(:)) = 0;
    raw_nonzero = raw(raw > 0);
    if isempty(raw_nonzero)
        partition_id = zeros(size(raw));
        partition_raw_id = zeros(0,1);
        num_partition = 0;
        return;
    end
    partition_raw_id = unique(raw_nonzero,'sorted');
    num_partition = numel(partition_raw_id);
    partition_id = zeros(size(raw));
    for kk = 1:num_partition
        partition_id(raw == partition_raw_id(kk)) = kk;
    end
end

function m_part = allocate_partition_budgets_local(N,energy,valid_count)
%ALLOCATE_PARTITION_BUDGETS_LOCAL Backward-compatible budget wrapper.  With
%the SISC [1,1] configuration the single global region receives all features.
    energy = double(energy(:));
    valid_count = double(valid_count(:));
    active = valid_count > 0;
    P = numel(energy);
    m_part = zeros(P,1);
    ids = find(active);
    if isempty(ids) || N <= 0, return; end
    N = floor(double(N));
    if N < numel(ids)
        error('Total neuron count N=%d is smaller than the %d nonempty macro-regions.',N,numel(ids));
    end
    Pn = numel(ids);
    base = floor(N/Pn);
    remn = N - Pn*base;
    m_part(ids) = base;

    % The remainder is assigned deterministically.  Energy is consulted
    % later by the global structure/background allocation.
    [~,ord] = sort(ids,'ascend');
    if remn > 0
        m_part(ids(ord(1:remn))) = m_part(ids(ord(1:remn))) + 1;
    end
end

function [alpha_use,m_S,m_B,region_weight,m_each] = ...
    allocate_sab_neurons_local(ctx,p_S,component_id,num_region,alpha_S,cfg)
%ALLOCATE_SAB_NEURONS_LOCAL
% Compute exactly the same target global allocation for the diagnostic
% coverage gate and the subsequent SFR refresh.

    if num_region < 1
        alpha_use = 0;
        m_S = 0;
        m_B = ctx.neuron;
        region_weight = zeros(0,1);
        m_each = zeros(0,1);
        return;
    end

    % No artificial alpha_min/alpha_max clipping: the residual-energy
    % partition determines the structure mass.  Integer feasibility is
    % handled globally; when there are more marked components than available
    % structure neurons, the largest residual masses receive the slots and
    % the remaining components are left at zero.
    alpha_use = min(max(alpha_S,0),1);
    m_S = min(max(round(ctx.neuron * alpha_use),0),ctx.neuron);
    m_B = ctx.neuron - m_S;

    region_weight = zeros(num_region,1);
    for kk = 1:num_region
        region_weight(kk) = sum(p_S(component_id == kk));
    end

    region_weight = gather(region_weight(:));
    region_weight = region_weight / (sum(region_weight) + eps);

    if m_S <= 0
        m_each = zeros(num_region,1);
        return;
    elseif m_S < num_region
        m_each = zeros(num_region,1);
        [~,order_small] = sort(region_weight,'descend');
        m_each(order_small(1:m_S)) = 1;
        return;
    end

    m_each = round(m_S * region_weight);
    m_min_each = max(1,min(10,floor(m_S/max(num_region,1))));

    m_each = max(m_each,m_min_each);

    diff_m = m_S - sum(m_each);

    if diff_m > 0
        % Add remaining neurons to the regions with the largest target mass.
        [~,order_add] = sort(region_weight,'descend');
        jj = 1;
        while diff_m > 0
            kk = order_add(jj);
            m_each(kk) = m_each(kk) + 1;
            diff_m = diff_m - 1;
            jj = jj + 1;
            if jj > num_region, jj = 1; end
        end

    elseif diff_m < 0
        % Remove excess neurons without violating the regional minimum.
        remain = -diff_m;
        [~,order_remove] = sort(m_each,'descend');

        while remain > 0
            changed = false;
            for jj = 1:num_region
                kk = order_remove(jj);
                if m_each(kk) > m_min_each
                    m_each(kk) = m_each(kk) - 1;
                    remain = remain - 1;
                    changed = true;
                    if remain == 0, break; end
                end
            end

            if ~changed
                error('Unable to satisfy the SFR regional allocation.');
            end
        end
    end
end

function n_train = regional_sample_cap_local(m_features,n_available,cfg,field_name)
%REGIONAL_SAMPLE_CAP_LOCAL independent N log(1+N) cap for each region.
% The structure components and the background use separate budgets; there
% is deliberately no constraint of the form N_S+N_B=N_AD.
    if n_available <= 0
        n_train = 0;
        return;
    end
    gamma_ad = getfield_default_sab_local(cfg,'gamma_AD',1.0);
    m_features = max(1,double(m_features));
    cap_auto = ceil(gamma_ad*m_features*log(1+m_features));
    cap = cap_auto;
    if isfield(cfg,field_name) && ~isempty(cfg.(field_name)) && ...
            isnumeric(cfg.(field_name)) && isfinite(cfg.(field_name)) && ...
            cfg.(field_name) > 0
        cap = min(cap,double(cfg.(field_name)));
    end
    n_train = max(1,min(double(n_available),cap));
end

function cfg = fill_sab_cfg_local(cfg)
%FILL_SAB_CFG_LOCAL defaults for the global residual--solution-gradient SFR diagnostic.
    if ~isfield(cfg,'SAB_eps') || isempty(cfg.SAB_eps), cfg.SAB_eps = 1e-12; end
    if ~isfield(cfg,'partition_grid') || isempty(cfg.partition_grid)
        cfg.partition_grid = [1,1];
    end
    if ~isfield(cfg,'theta_dorfler') || isempty(cfg.theta_dorfler)
        cfg.theta_dorfler = 0.50;
    end
    if ~isfield(cfg,'theta_res') || isempty(cfg.theta_res), cfg.theta_res = 0.5; end
    if ~isfield(cfg,'theta_grad') || isempty(cfg.theta_grad), cfg.theta_grad = 0.5; end
    if ~isscalar(cfg.theta_res) || ~isscalar(cfg.theta_grad) || ...
            ~isfinite(cfg.theta_res) || ~isfinite(cfg.theta_grad) || ...
            cfg.theta_res < 0 || cfg.theta_grad < 0 || ...
            cfg.theta_res + cfg.theta_grad <= 0
        error('cfg.theta_res and cfg.theta_grad must be nonnegative with positive sum.');
    end
    if ~isfield(cfg,'delta_floor') || isempty(cfg.delta_floor)
        cfg.delta_floor = 1e-3;
    end
    if ~isfield(cfg,'tau_eff') || isempty(cfg.tau_eff), cfg.tau_eff = 0.35; end
    if ~isfield(cfg,'tau_conn') || isempty(cfg.tau_conn), cfg.tau_conn = 0.01; end
    if ~isfield(cfg,'tau_cover') || isempty(cfg.tau_cover), cfg.tau_cover = 0.35; end
    if ~isfield(cfg,'use_cover_gate_for_refresh') || isempty(cfg.use_cover_gate_for_refresh)
        cfg.use_cover_gate_for_refresh = true;
    end
    if ~isfield(cfg,'plot_figures') || isempty(cfg.plot_figures), cfg.plot_figures = false; end
end

function cfg = fill_sab_ref_cfg_local(cfg,ctx)
%FILL_SAB_REF_CFG_LOCAL defaults for independent regional scale training.
    if ~isfield(cfg,'gamma_AD') || isempty(cfg.gamma_AD), cfg.gamma_AD = 1.0; end
    if ~isfield(cfg,'eta_accept') || isempty(cfg.eta_accept), cfg.eta_accept = 0.0; end
    if ~isscalar(cfg.eta_accept) || ~isfinite(cfg.eta_accept) || ...
            cfg.eta_accept < 0 || cfg.eta_accept >= 1
        error('cfg.eta_accept must satisfy 0 <= eta_accept < 1.');
    end
    if ~isfield(cfg,'val_N') || isempty(cfg.val_N), cfg.val_N = [100,100]; end
    if ~isfield(cfg,'b_val') || isempty(cfg.b_val), cfg.b_val = 600; end
    if ~isfield(cfg,'max_train_points_S'), cfg.max_train_points_S = []; end
    if ~isfield(cfg,'max_train_points_B'), cfg.max_train_points_B = []; end
    if ~isfield(cfg,'lambda_reg') || isempty(cfg.lambda_reg), cfg.lambda_reg = 1e-6; end
    if ~isfield(cfg,'adam_opts') || isempty(cfg.adam_opts)
        cfg.adam_opts = struct('eta',0.7,'maxit',30,'eps_adam',1e-8);
    end
    if ~isfield(cfg,'r0') || isempty(cfg.r0), cfg.r0 = ctx.r; end
end

function v = getfield_default_sab_local(s,name,default_val)
    if isstruct(s) && isfield(s,name) && ~isempty(s.(name))
        v = s.(name);
    else
        v = default_val;
    end
end

function A = enforce_macro_anchor_local(A,A0,pp,ctx,diagS)
%ENFORCE_MACRO_ANCHOR_LOCAL Keep every refreshed anchor in its owner macro.
% Jittering a diagnostic-cell center is already local to that cell, but the
% explicit macro projection below makes the ownership invariant exact at
% floating-point boundaries and documents the regional SFR rule.
    if isempty(A), return; end
    pg = double(diagS.partition_grid(:).');
    nxp = pg(1); nyp = pg(2);
    raw_id = diagS.partition_raw_id(pp);
    [iy,ix] = ind2sub([nyp,nxp],raw_id);

    xlo = ctx.a1 + (ix-1)*(ctx.a2-ctx.a1)/nxp;
    xhi = ctx.a1 + ix*(ctx.a2-ctx.a1)/nxp;
    ylo = ctx.a3 + (iy-1)*(ctx.a4-ctx.a3)/nyp;
    yhi = ctx.a3 + iy*(ctx.a4-ctx.a3)/nyp;

    A(1,:) = min(max(A(1,:),xlo),xhi);
    A(2,:) = min(max(A(2,:),ylo),yhi);

    % If a clamp lands on an invalid part of a nonrectangular macro-cell,
    % revert only that anchor to its already-valid diagnostic center.
    if isfield(ctx,'inside_fn') && ~isempty(ctx.inside_fn)
        good = logical(gather(ctx.inside_fn(A.'))).';
        if any(~good)
            A(:,~good) = A0(:,~good);
        end
    end
end


function A = jitter_project_centers_local(A0,ctx,diagS,cfg,stream_offset)
    %JITTER_PROJECT_CENTERS_LOCAL
    % After a diagnostic cell is selected by weight, sample its anchor
    % componentwise uniformly inside that cell.
    %
    % A0 is d-by-m, each column is a selected diagnostic point.
    % xi_{j,k} ~ U(-kappa_jit*h_k/2, kappa_jit*h_k/2).
    
        A = A0;
        if nargin < 5, stream_offset = 0; end
    
        % Backward-compatible switch. All paper experiments enable this.
        if ~isfield(cfg,'use_center_jitter') || ~cfg.use_center_jitter
            return;
        end
    
        if isfield(cfg,'kappa_jit')
            kappa_jit = cfg.kappa_jit;
        elseif isfield(cfg,'center_jitter_factor')
            kappa_jit = cfg.center_jitter_factor;
        else
            kappa_jit = 1.0;
        end
        if ~isscalar(kappa_jit) || ~isfinite(kappa_jit) || ...
                kappa_jit < 0 || kappa_jit > 1
            error('cfg.kappa_jit must lie in [0,1]. Use 1 for the full diagnostic cell.');
        end
    
        d = size(A0,1);
    
        if d ~= 2
            error('jitter_project_centers_local currently supports 2D problems only.');
        end
    
        if isfield(diagS,'xx1') && numel(diagS.xx1) >= 2
            h1 = diagS.xx1(2) - diagS.xx1(1);
        else
            h1 = diagS.h(1);
        end
    
        if isfield(diagS,'xx2') && numel(diagS.xx2) >= 2
            h2 = diagS.xx2(2) - diagS.xx2(1);
        else
            h2 = diagS.h(2);
        end
    
        h = [h1; h2];
    
        % Optionally use an independent stream so anchor offsets do not
        % shift the random-direction sequence used by the feature basis.
        use_independent_stream = isfield(cfg,'decouple_jitter_rng') && ...
            cfg.decouple_jitter_rng;
        if use_independent_stream
            seed_base = 0;
            if isfield(cfg,'jitter_seed_base') && ~isempty(cfg.jitter_seed_base)
                seed_base = cfg.jitter_seed_base;
            end
            jitter_seed = mod(round(double(seed_base))*10007 + ...
                round(double(stream_offset))*1009 + 7919,2^32-1);
            jitter_stream = RandStream('mt19937ar','Seed',jitter_seed);
            U = rand(jitter_stream,size(A0));
        else
            jitter_stream = [];
            U = rand(size(A0));
        end

        % For kappa_jit=1, this is uniform over the selected diagnostic cell.
        Xi = bsxfun(@times,U-0.5,kappa_jit*h);
        A = A0 + Xi;

        % Roundoff-safe rectangular bounds.
        lower = [ctx.a1; ctx.a3];
        upper = [ctx.a2; ctx.a4];
        A = bsxfun(@max,A,lower);
        A = bsxfun(@min,A,upper);

        % On a non-rectangular domain, reject only out-of-domain draws and
        % resample them in their originally selected cells. This samples
        % uniformly from the valid part of each selected diagnostic cell.
        if isfield(ctx,'inside_fn') && ~isempty(ctx.inside_fn)
            is_inside = logical(gather(ctx.inside_fn(A.'))).';
            for attempt = 1:50
                bad = find(~is_inside);
                if isempty(bad), break; end
                if use_independent_stream
                    U_bad = rand(jitter_stream,d,numel(bad));
                else
                    U_bad = rand(d,numel(bad));
                end
                Xi_bad = bsxfun(@times,U_bad-0.5,kappa_jit*h);
                A_bad = A0(:,bad) + Xi_bad;
                A_bad = bsxfun(@max,A_bad,lower);
                A_bad = bsxfun(@min,A_bad,upper);
                good_bad = logical(gather(ctx.inside_fn(A_bad.'))).';
                if any(good_bad)
                    accepted = bad(good_bad);
                    A(:,accepted) = A_bad(:,good_bad);
                    is_inside(accepted) = true;
                end
            end
            A(:,~is_inside) = A0(:,~is_inside); % safe measure-zero fallback
        end
    end
