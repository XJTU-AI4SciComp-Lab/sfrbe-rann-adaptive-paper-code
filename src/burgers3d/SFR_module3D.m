function varargout = SFR_module3D(action,varargin)
%SFR_MODULE_COMPAT SFR structure-based adaptive bias refresh for Burgers3D space-time.
%
% This is a compact 3D version. It diagnoses residual-gradient
% space-time structure on a cell-centered (x,y,t) diagnostic grid and, if triggered,
% rebuilds the basis
% as one structure block plus one background block. Both r-training steps
% use ADRaNN_train_r_analytic3D through Burgers2D_Newton_module3D('train_r_value').

    switch lower(action)
        case 'diagnose'
            varargout{1} = sab_diagnose_local(varargin{:});
        case 'refresh'
            varargout{1} = sab_refresh_local(varargin{:});
        otherwise
            error('Unknown SFR_module3D action = %s',action);
    end
end

%% ============================================================
% Diagnose
%% ============================================================
function diagS = sab_diagnose_local(ctx,cfg)
    if nargin < 2 || isempty(cfg), cfg = struct(); end
    eps0 = getfield_default_local(cfg,'SAB_eps',1e-12);

    diag_N = getfield_default_local(cfg,'diag_N',[ctx.grid.Nx,ctx.grid.Ny,ctx.grid.Nt]);
    if isscalar(diag_N), diag_N = repmat(diag_N,1,3); end
    if numel(diag_N) ~= 3 || any(~isfinite(diag_N)) || ...
            any(diag_N < 1) || any(diag_N ~= round(diag_N))
        error('cfg.diag_N must contain one or three positive integers.');
    end
    diag_N = double(diag_N(:).');

    bounds = [ctx.grid.x_grid(1),ctx.grid.x_grid(end), ...
              ctx.grid.y_grid(1),ctx.grid.y_grid(end), ...
              ctx.grid.t_grid(1),ctx.grid.t_grid(end)];
    h = [(bounds(2)-bounds(1))/diag_N(1), ...
         (bounds(4)-bounds(3))/diag_N(2), ...
         (bounds(6)-bounds(5))/diag_N(3)];
    x_diag = bounds(1) + ((0:diag_N(1)-1)+0.5)*h(1);
    y_diag = bounds(3) + ((0:diag_N(2)-1)+0.5)*h(2);
    t_diag = bounds(5) + ((0:diag_N(3)-1)+0.5)*h(3);
    [XXd,YYd,TTd] = ndgrid(x_diag,y_diag,t_diag);
    XT_diag = [XXd(:),YYd(:),TTd(:)];

    % Analytic derivatives on diagnostic cell centers.  The solver's PDE,
    % boundary, initial, and output grids remain unchanged.
    u   = Burgers2D_Newton_module3D('eval',ctx.terms,ctx.coef,XT_diag,0,0);
    ux  = Burgers2D_Newton_module3D('eval',ctx.terms,ctx.coef,XT_diag,1,1);
    uy  = Burgers2D_Newton_module3D('eval',ctx.terms,ctx.coef,XT_diag,1,2);
    ut  = Burgers2D_Newton_module3D('eval',ctx.terms,ctx.coef,XT_diag,1,3);
    uxx = Burgers2D_Newton_module3D('eval',ctx.terms,ctx.coef,XT_diag,2,1);
    uyy = Burgers2D_Newton_module3D('eval',ctx.terms,ctx.coef,XT_diag,2,2);
    residual = ut + u.*(ux+uy) - ctx.problem.eps_burgers*(uxx+uyy);

    U_grid = reshape(gather(u),diag_N);
    R_grid = reshape(gather(residual),diag_N);
    UX_grid = reshape(gather(ux),diag_N);
    UY_grid = reshape(gather(uy),diag_N);

    valid_mask = true(diag_N);
    exclude_boundary_layers = getfield_default_local(cfg,'exclude_boundary_layers',0);
    if exclude_boundary_layers > 0
        L = exclude_boundary_layers;
        if 2*L >= min(diag_N)
            error('exclude_boundary_layers is too large for cfg.diag_N.');
        end
        valid_mask(1:L,:,:) = false;
        valid_mask(end-L+1:end,:,:) = false;
        valid_mask(:,1:L,:) = false;
        valid_mask(:,end-L+1:end,:) = false;
        valid_mask(:,:,1:L) = false;
    end
    % Unified residual--gradient score on the cell-centered diagnostic grid.
    % The same score drives Dörfler marking and within-cell sampling.
    mu = prod(h);
    s_res = abs(R_grid);
    s_grad = sqrt(abs(UX_grid).^2 + abs(UY_grid).^2);
    s_res(~valid_mask) = 0;
    s_grad(~valid_mask) = 0;
    s_res_hat = s_res./(max(s_res(:)) + eps0);
    s_grad_hat = s_grad./(max(s_grad(:)) + eps0);
    theta_res = getfield_default_local(cfg,'theta_res',0.5);
    theta_grad = getfield_default_local(cfg,'theta_grad',0.5);
    S_score = theta_res.*s_res_hat + theta_grad.*s_grad_hat;
    S_score(~valid_mask) = 0;
    S_norm = S_score./(max(S_score(:)) + eps0);
    S_norm(~valid_mask) = 0;
    eta2 = mu.*S_norm;
    eta2(~valid_mask) = 0;
    eta2_total = sum(eta2(:));
    density_score = eta2;
    delta_floor = getfield_default_local(cfg,'delta_floor',1e-3);
    mean_eta2 = eta2_total/(nnz(valid_mask)*mu + eps0);
    density_score = density_score + delta_floor*mean_eta2*mu;
    density_score(~valid_mask) = 0;

    maskS = false(diag_N);
    if eta2_total > eps0
        vals = eta2(valid_mask);
        [vals,ord] = sort(vals,'descend'); %#ok<ASGLU>
        theta = getfield_default_local(cfg,'theta_dorfler',0.50);
        nmark = find(cumsum(vals) >= theta*eta2_total,1,'first');
        if isempty(nmark), nmark = numel(ord); end
        valid_linear = find(valid_mask);
        maskS(valid_linear(ord(1:nmark))) = true;
    end

    min_region_size = getfield_default_local(cfg,'min_region_size',80);
    num_region = double(nnz(maskS) >= min_region_size);
    if num_region == 0 && nnz(maskS) > 0
        % Keep a nonempty Dörfler set on coarse grids rather than deleting
        % the entire structure candidate by a cell-count safeguard.
        num_region = 1;
    end
    mass_total = sum(density_score(valid_mask)) + eps0;
    mass_struct = sum(density_score(maskS));
    main_region_mass_ratio = mass_struct / mass_total;

    E_diag = mean(abs(R_grid(valid_mask)));
    Lambda = sum(eta2(maskS))/(eta2_total + eps0);
    q_score = eta2/(eta2_total + eps0);
    n_eff = 1/(sum(q_score(:).^2) + eps0);
    chi_eff = n_eff/nnz(valid_mask);
    C_loc_tilde = Lambda;

    tau_eff = getfield_default_local(cfg,'tau_eff',0.35);
    G_SAB = (num_region > 0) && (chi_eff <= tau_eff);

    diagS = struct();
    diagS.R_grid = R_grid;
    diagS.U_grid = U_grid;
    diagS.s_res = s_res;
    diagS.s_grad = s_grad;
    diagS.S_norm = S_norm;
    diagS.theta_res = theta_res;
    diagS.theta_grad = theta_grad;
    diagS.density_score = density_score;
    diagS.maskS = maskS;
    diagS.maskB = valid_mask & ~maskS;
    diagS.structure_mask = maskS;
    diagS.background_mask = diagS.maskB;
    diagS.component_id = double(maskS);
    p_diag = zeros(size(density_score));
    p_diag(valid_mask) = density_score(valid_mask);
    diagS.p_S = p_diag(:)/(sum(p_diag(:))+eps0);
    diagS.XT_diag = XT_diag;
    diagS.x_diag = x_diag;
    diagS.y_diag = y_diag;
    diagS.t_diag = t_diag;
    diagS.diag_N = diag_N;
    diagS.h = h;
    diagS.bounds = bounds;
    diagS.valid_mask = valid_mask;
    diagS.num_region = num_region;
    diagS.num_region_raw = num_region;
    diagS.main_region_mass_ratio = main_region_mass_ratio;
    diagS.E_diag = E_diag;
    diagS.Lambda = Lambda;
    diagS.chi_eff = chi_eff;
    diagS.C_loc_tilde = C_loc_tilde;
    diagS.G_SAB = double(G_SAB);

    fprintf('\n========== SFR 3D diagnosis ==========\n');
    fprintf('score weights: residual=%.3f, gradient=%.3f\n',theta_res,theta_grad);
    fprintf('num_region=%d, mass_ratio=%.4f, chi_eff=%.4f, G_SAB=%d\n', ...
        num_region,main_region_mass_ratio,chi_eff,diagS.G_SAB);
end

%% ============================================================
% Refresh
%% ============================================================
function ctx = sab_refresh_local(ctx,diagS,cfg)
    if nargin < 3 || isempty(cfg), cfg = struct(); end
    if ~isfield(diagS,'G_SAB') || ~diagS.G_SAB
        fprintf('SAB refresh skipped: G_SAB = 0.\n');
        ctx.SAB_accept = 0;
        return;
    end

    cfg = fill_refresh_cfg_local(cfg,ctx);
    ctx_old = ctx;

    [~,~,parts_old] = Burgers2D_Newton_module3D('assemble',ctx,cfg.newton_cfg);
    rel_old = parts_old.rel_res;
    U_grid = diagS.U_grid;
    y_all = U_grid(:);
    XT_diag = diagS.XT_diag;

    density_score = diagS.density_score;
    maskS = diagS.maskS;
    maskB = diagS.maskB;

    pS = zeros(numel(density_score),1);
    ds = density_score(:);
    pS(maskS(:)) = ds(maskS(:));
    if sum(pS) <= 0
        pS(maskS(:)) = 1;
    end
    pS = pS/(sum(pS)+eps);

    pB = zeros(numel(density_score),1);
    pB(maskB(:)) = ds(maskB(:));
    if sum(pB) <= 0
        pB(maskB(:)) = 1;
    end
    pB = pB/(sum(pB)+eps);

    alpha_use = cfg.alpha_use;
    mS = max(1,round(alpha_use*ctx.neuron));
    mB = max(1,ctx.neuron - mS);

    terms_new = {};
    centers_all = [];
    r_list = [];

    fprintf('\n========== SFR refresh: structure/background 3D basis ==========\n');

    % Structure block.
    idxC = YY_utils3D('sample_from_prob',pS,mS);
    centersS = XT_diag(idxC,:).';
    centersS = sample_within_cells_local(centersS,diagS,cfg);

    nTrainS = regional_sample_cap_3d_local( ...
        mS,numel(pS),cfg,'max_train_points_S');
    idxT = YY_utils3D('sample_from_prob',pS,nTrainS);
    XS = XT_diag(idxT,:);
    yS = y_all(idxT);

    rawS = Burgers2D_Newton_module3D('make_raw_family',cfg.family,mS,centersS,cfg);
    [rS,mseS,termS] = Burgers2D_Newton_module3D('train_r_value', ...
        XS,yS,rawS,cfg.r0,cfg.adam_opts,cfg.family,cfg.lambda_reg);
    fprintf('Structure block: m=%d, r=(%s ), MSE=%.6e\n',mS,sprintf(' %.6f',rS),mseS);

    terms_new{end+1} = termS;
    centers_all = [centers_all,centersS]; %#ok<AGROW>
    r_list = [r_list,rS(:)]; %#ok<AGROW>

    % Background block.
    idxC = YY_utils3D('sample_from_prob',pB,mB);
    centersB = XT_diag(idxC,:).';
    centersB = sample_within_cells_local(centersB,diagS,cfg);

    nTrainB = regional_sample_cap_3d_local( ...
        mB,numel(pB),cfg,'max_train_points_B');
    idxT = YY_utils3D('sample_from_prob',pB,nTrainB);
    XB = XT_diag(idxT,:);
    yB = y_all(idxT);

    rawB = Burgers2D_Newton_module3D('make_raw_family',cfg.family,mB,centersB,cfg);
    [rB,mseB,termB] = Burgers2D_Newton_module3D('train_r_value', ...
        XB,yB,rawB,cfg.r0,cfg.adam_opts,cfg.family,cfg.lambda_reg);
    fprintf('Background block: m=%d, r=(%s ), MSE=%.6e\n',mB,sprintf(' %.6f',rB),mseB);

    terms_new{end+1} = termB;
    centers_all = [centers_all,centersB];
    r_list = [r_list,rB(:)];

    % PDE-aware initialization: compare value projection and linearized Burgers2D.
    proj_cfg = cfg.newton_cfg;
    proj_cfg.solve_lambda = 0;

    [coef_new,init_info] = Burgers2D_Newton_module3D('best_burgers2d_init_svd', ...
        ctx,terms_new,ctx.grid.XT_grid,y_all,proj_cfg);
    fprintf('SAB init compare: value rel_res=%.6e, linearized rel_res=%.6e, use=%s\n', ...
        init_info.rel_value,init_info.rel_linearized,init_info.method);

    ctx_try = ctx;
    ctx_try.terms = terms_new;
    ctx_try.coef = coef_new;
    ctx_try.weight = coef_new;
    % SFR replaces the entire basis, so neither the fixed-operator SVD nor
    % the frozen-Jref state from the old basis is compatible with ctx_try.
    % Keeping either state would make the next solve report reused_svd while
    % multiplying by a factorization of different columns.
    if isfield(ctx_try,'fixed_operator_svd_state')
        ctx_try.fixed_operator_svd_state = [];
    end
    if isfield(ctx_try,'fixed_operator_basis_cache')
        ctx_try.fixed_operator_basis_cache = [];
    end
    if isfield(ctx_try,'jref')
        ctx_try.jref = [];
    end
    ctx_try.current_centers = centers_all;
    ctx_try.last_r_opt = mean(r_list,2);
    ctx_try.r = ctx_try.last_r_opt(:);
    ctx_try.r_opt_region = r_list;

    ctx_try = Burgers2D_Newton_module3D('relax_full',ctx_try,cfg.newton_cfg);
    [~,~,parts_new] = Burgers2D_Newton_module3D('assemble',ctx_try,cfg.newton_cfg);
    rel_new = parts_new.rel_res;

    eta_accept = getfield_default_local(cfg,'eta_accept',0.0);
    fprintf('\n========== SFR refresh acceptance check ==========\n');
    fprintf('Old basis rel_old = %.6e\n',rel_old);
    fprintf('New basis rel_new = %.6e\n',rel_new);

    if rel_new <= (1-eta_accept)*rel_old
        fprintf('Accept SFR refresh: nonlinear residual decreased.\n');
        ctx = ctx_try;
        ctx.SAB_accept = 1;
    else
        fprintf('Reject SFR refresh: nonlinear residual did not decrease; restore the old basis.\n');
        ctx = ctx_old;
        ctx.SAB_accept = 0;
    end

    ctx.sab_refresh_info = struct();
    ctx.sab_refresh_info.alpha_use = alpha_use;
    ctx.sab_refresh_info.mS = mS;
    ctx.sab_refresh_info.mB = mB;
    ctx.sab_refresh_info.r_list = r_list;
end

function cfg = fill_refresh_cfg_local(cfg,ctx)
    if ~isfield(cfg,'family'), cfg.family = 'tanh'; end
    if ~isfield(cfg,'r0'), cfg.r0 = ctx.r(:); end
    if ~isfield(cfg,'alpha_use'), cfg.alpha_use = 0.60; end
    if ~isfield(cfg,'lambda_reg'), cfg.lambda_reg = 1e-6; end
    if ~isfield(cfg,'gamma_AD') || isempty(cfg.gamma_AD), cfg.gamma_AD = 1.0; end
    if ~isfield(cfg,'max_train_points_S'), cfg.max_train_points_S = []; end
    if ~isfield(cfg,'max_train_points_B'), cfg.max_train_points_B = []; end
    if ~isfield(cfg,'eta_accept'), cfg.eta_accept = 0.0; end
    if ~isfield(cfg,'sample_within_cell'), cfg.sample_within_cell = true; end
    if ~isfield(cfg,'newton_cfg'), cfg.newton_cfg = struct(); end

    if ~isfield(cfg,'adam_opts'), cfg.adam_opts = struct(); end
    if ~isfield(cfg.adam_opts,'eta'), cfg.adam_opts.eta = 0.35; end
    if ~isfield(cfg.adam_opts,'maxit'), cfg.adam_opts.maxit = 30; end
    if ~isfield(cfg.adam_opts,'clip_min'), cfg.adam_opts.clip_min = ones(numel(cfg.r0),1); end
    if ~isfield(cfg.adam_opts,'clip_max'), cfg.adam_opts.clip_max = 300*ones(numel(cfg.r0),1); end
    if ~isfield(cfg.adam_opts,'verbose'), cfg.adam_opts.verbose = true; end
    if ~isfield(cfg.adam_opts,'use_rel_grad'), cfg.adam_opts.use_rel_grad = true; end
end

function n_train = regional_sample_cap_3d_local(m_features,n_available,cfg,field_name)
    if n_available <= 0
        n_train = 0;
        return;
    end
    m_features = max(1,double(m_features));
    cap = ceil(cfg.gamma_AD*m_features*log(1+m_features));
    if isfield(cfg,field_name) && ~isempty(cfg.(field_name)) && ...
            isnumeric(cfg.(field_name)) && isfinite(cfg.(field_name)) && ...
            cfg.(field_name) > 0
        cap = min(cap,double(cfg.(field_name)));
    end
    n_train = max(1,min(double(n_available),cap));
end

function centers = sample_within_cells_local(cell_centers,diagS,cfg)
    % Select-cell sampling is performed by the caller. This second stage
    % draws one anchor uniformly inside each selected 3D diagnostic cell.
    centers = cell_centers;
    if ~cfg.sample_within_cell || isempty(centers)
        return;
    end
    h = diagS.h(:);
    bounds = diagS.bounds(:);
    if numel(h) ~= 3 || numel(bounds) ~= 6
        error('3D within-cell sampling requires diagS.h and diagS.bounds.');
    end
    centers = centers + bsxfun(@times,rand(size(centers))-0.5,h);
    lower = bounds([1,3,5]);
    upper = bounds([2,4,6]);
    centers = bsxfun(@max,centers,lower);
    centers = bsxfun(@min,centers,upper);
end

function val = getfield_default_local(s,name,default_val)
    if isstruct(s) && isfield(s,name) && ~isempty(s.(name))
        val = s.(name);
    else
        val = default_val;
    end
end
