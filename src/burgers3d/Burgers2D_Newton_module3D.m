function varargout = Burgers2D_Newton_module3D(action,varargin)
%BURGERS2D_NEWTON_MODULE Nonlinear residual, Newton, frozen-Jref SVD tools
% for 2D viscous Burgers in 3D space-time:
%
%   R(u)=u_t + u*(u_x + u_y) - eps*(u_xx + u_yy).
%
% Main interfaces:
%   coef = Burgers2D_Newton_module3D('fit_values',terms,X,y,lambda)
%   ctx  = Burgers2D_Newton_module3D('relax_full',ctx,cfg)
%   ctx  = Burgers2D_Newton_module3D('freeze_jacobian',ctx,cfg)
%   ctx  = Burgers2D_Newton_module3D('append_jref_block',ctx,term,cfg)
%   [R,J,parts] = Burgers2D_Newton_module3D('assemble',ctx,cfg)
%   val  = Burgers2D_Newton_module3D('eval',terms,coef,X,ord,dir)

    switch lower(action)
        case {'fit','fit_values'}
            varargout{1} = fit_coeff_to_values_local(varargin{:});

        case {'fit_values_svd','project_values_svd','svd_fit_values'}
            varargout{1} = fit_coeff_to_values_svd_local(varargin{:});

        case {'fit_linearized_burgers2d_svd','linearized_burgers2d_fit'}
            varargout{1} = fit_linearized_burgers2d_svd_local(varargin{:});

        case {'best_burgers2d_init_svd','best_linearized_or_value_init','best_kdv_init_svd'}
            [varargout{1},varargout{2}] = best_burgers2d_initial_coef_local(varargin{:});

        case {'incremental_init_block','incremental_block_init'}
            [varargout{1},varargout{2}] = incremental_block_initial_coef_local(varargin{:});

        case {'jref_augmented_init','incremental_jref_init'}
            [varargout{1},varargout{2}] = jref_augmented_initial_coef_local(varargin{:});

        case {'relax','newton','relax_full','full_newton'}
            varargout{1} = newton_relax_full_ctx_local(varargin{:});

        case {'relax_inexact','inexact_newton','frozen_newton'}
            varargout{1} = newton_relax_inexact_ctx_local(varargin{:});

        case {'assemble','assemble_newton'}
            [varargout{1},varargout{2},varargout{3}] = assemble_newton_system_ctx_local(varargin{:});

        case {'eval','eval_solution'}
            varargout{1} = eval_solution_local(varargin{:});

        case {'eval_terms'}
            varargout{1} = eval_terms_local(varargin{:});

        case {'grid_solution'}
            varargout{1} = grid_solution_local(varargin{:});

        case {'residual_grid'}
            varargout{1} = residual_grid_local(varargin{:});

        case {'num_cols','term_num_cols'}
            varargout{1} = term_num_cols_local(varargin{:});

        case {'ridge_solve'}
            varargout{1} = ridge_solve_local(varargin{:});

        case {'make_raw_family'}
            varargout{1} = make_raw_family_local(varargin{:});

        case {'build_term_from_r'}
            varargout{1} = build_term_from_r_local(varargin{:});

        case {'train_r_value'}
            [varargout{1},varargout{2},varargout{3}] = train_r_value_local(varargin{:});

        case {'train_r_residual'}
            [varargout{1},varargout{2},varargout{3}] = train_r_residual_local(varargin{:});

        case {'jref_matrix_for_term'}
            varargout{1} = jref_matrix_for_term_local(varargin{:});

        case {'current_jacobian_for_term'}
            varargout{1} = current_jacobian_for_term_local(varargin{:});

        case {'activate_block_ls','block_local_ls'}
            varargout{1} = activate_block_ls_local(varargin{:});

        case {'freeze_jacobian','freeze_jref'}
            varargout{1} = freeze_jacobian_local(varargin{:});

        case {'append_jref_block'}
            varargout{1} = append_jref_block_local(varargin{:});

        otherwise
            error('Unknown Burgers2D_Newton_module3D action = %s',action);
    end
end

%% ============================================================
% basic solvers
%% ============================================================
function coef = fit_coeff_to_values_local(terms,X,y,lambda)
% Training/readout fit with ridge regularization.
    Phi = eval_terms_local(terms,X,0,0);
    coef = ridge_solve_local(Phi,y(:),lambda);
end

function coef = fit_coeff_to_values_svd_local(terms,X,y,cfg)
% Projection/initialization solve with truncated SVD and solve_lambda=0.
    if nargin < 4 || isempty(cfg), cfg = struct(); end
    cfg = fill_newton_cfg_local(cfg);
    cfg.linear_solver = 'svd';
    cfg.solve_lambda = getfield_default_local(cfg,'solve_lambda',0);
    Phi = eval_terms_local(terms,X,0,0);
    [A_solve,y_solve] = compress_tall_system_local(Phi,y(:));
    clear Phi
    coef = linear_solve_svd_or_normal_local(A_solve,y_solve,cfg);
end

function beta = ridge_solve_local(A,y,lambda)
    if nargin < 3 || isempty(lambda), lambda = 1e-6; end
    A = gather(A);
    y = gather(y(:));
    beta = (A.'*A + lambda*eye(size(A,2))) \ (A.'*y);
end

%% ============================================================
% Burgers2D-aware warm-start
%% ============================================================
function coef = fit_linearized_burgers2d_svd_local(ctx_old,terms_new,cfg)
% Solve the Burgers2D equation linearized around old state u_old:
%
%   v_t + v*(ux_old+uy_old) + u_old*(v_x+v_y)
%       - eps*(v_xx+v_yy)
%     = u_old*(ux_old+uy_old),
%
% with Dirichlet boundary and initial constraints.

    cfg = fill_newton_cfg_local(cfg);
    cfg.linear_solver = 'svd';
    cfg.solve_lambda = 0;

    problem = ctx_old.problem;
    grid = ctx_old.grid;
    epsv = problem.eps_burgers;

    X = grid.XT_int;

    u_old  = eval_solution_local(ctx_old.terms,ctx_old.coef,X,0,0);
    ux_old = eval_solution_local(ctx_old.terms,ctx_old.coef,X,1,1);
    uy_old = eval_solution_local(ctx_old.terms,ctx_old.coef,X,1,2);

    yint = u_old.*(ux_old+uy_old);

    Phi = eval_terms_local(terms_new,X,0,0);
    Aint = (ux_old+uy_old).*Phi;
    clear Phi
    Phix = eval_terms_local(terms_new,X,1,1);
    Aint = Aint + u_old.*Phix;
    clear Phix
    Phiy = eval_terms_local(terms_new,X,1,2);
    Aint = Aint + u_old.*Phiy;
    clear Phiy
    Phit = eval_terms_local(terms_new,X,1,3);
    Aint = Aint + Phit;
    clear Phit
    Phixx = eval_terms_local(terms_new,X,2,1);
    Aint = Aint - epsv*Phixx;
    clear Phixx
    Phiyy = eval_terms_local(terms_new,X,2,2);
    Aint = Aint - epsv*Phiyy;
    clear Phiyy

    Abd = eval_terms_local(terms_new,grid.X_bd,0,0);
    Aic = eval_terms_local(terms_new,grid.X_ic,0,0);

    A = [Aint;
         cfg.eta_bc*Abd;
         cfg.eta_ic*Aic];

    y = [yint;
         cfg.eta_bc*grid.g_bd;
         cfg.eta_ic*grid.u_ic];

    clear Aint Abd Aic yint u_old ux_old uy_old
    [A_solve,y_solve] = compress_tall_system_local(A,y);
    clear A y
    coef = linear_solve_svd_or_normal_local(A_solve,y_solve,cfg);
end

function [coef_best,info] = best_burgers2d_initial_coef_local(ctx_old,terms_new,X_value,y_value,cfg)
% Choose value projection or Burgers2D-linearized initialization.

    cfg = fill_newton_cfg_local(cfg);
    cfg.linear_solver = 'svd';
    cfg.solve_lambda = 0;

    coef_val = fit_coeff_to_values_svd_local(terms_new,X_value,y_value,cfg);
    coef_lin = fit_linearized_burgers2d_svd_local(ctx_old,terms_new,cfg);

    ctx_val = ctx_old;
    ctx_val.terms = terms_new;
    ctx_val.coef = coef_val;
    ctx_val.weight = coef_val;

    ctx_lin = ctx_old;
    ctx_lin.terms = terms_new;
    ctx_lin.coef = coef_lin;
    ctx_lin.weight = coef_lin;

    [~,~,parts_val] = assemble_newton_system_ctx_local(ctx_val,cfg);
    [~,~,parts_lin] = assemble_newton_system_ctx_local(ctx_lin,cfg);

    info = struct();
    info.rel_value = parts_val.rel_res;
    info.rel_linearized = parts_lin.rel_res;

    if parts_lin.rel_res <= parts_val.rel_res
        coef_best = coef_lin;
        info.method = 'linearized_burgers2d';
        info.rel_best = parts_lin.rel_res;
    else
        coef_best = coef_val;
        info.method = 'value_projection';
        info.rel_best = parts_val.rel_res;
    end
end

function [coef_aug,info] = incremental_block_initial_coef_local(ctx_old,term_new,cfg)
%INCREMENTAL_BLOCK_INITIAL_COEF Initialize only the newly appended block.
% The existing coefficients are retained.  A small SVD solve is applied to
% the new Jacobian block against the current nonlinear residual; no SVD of
% the complete old-plus-new basis is formed.

    cfg = fill_newton_cfg_local(cfg);
    [R_old,~,parts_old] = assemble_newton_system_ctx_local(ctx_old,cfg);
    C_new = current_jacobian_for_term_local(ctx_old,term_new,cfg);
    a_new = linear_solve_svd_or_normal_local(C_new,-R_old,cfg);

    coef_aug = [ctx_old.coef(:); a_new(:)];

    ctx_try = ctx_old;
    ctx_try.terms = ctx_old.terms;
    ctx_try.terms{end+1} = term_new;
    ctx_try.coef = coef_aug;
    ctx_try.weight = coef_aug;
    [~,~,parts_new] = assemble_newton_system_ctx_local(ctx_try,cfg);

    info = struct();
    info.method = 'incremental_linearized_block';
    info.rel_before = parts_old.rel_res;
    info.rel_after = parts_new.rel_res;
    info.rel_value = NaN;
    info.rel_linearized = parts_new.rel_res;
    info.block_size = numel(a_new);
end

function [coef_aug,info] = jref_augmented_initial_coef_local(ctx_old,term_new,R,cfg)
%JREF_AUGMENTED_INITIAL_COEF Incremental frozen-Jacobian projection.
%
% The RBE capture score is computed from the component of the new block
% orthogonal to the frozen old Jacobian.  Solving only C_new*a_new against
% the residual does not realize that score because the old coefficients are
% not allowed to change.  This routine applies the small augmented SVD
% update [J_ref,C_new] and solves for both the old correction and the new
% block.  No SVD of the complete augmented tall matrix is formed.

    cfg = fill_newton_cfg_local(cfg);
    if ~isfield(cfg,'orth_cfg')
        cfg.orth_cfg = ctx_old.orth_cfg;
    end
    R = gather(R(:));
    if ~isfield(ctx_old,'jref') || ~isfield(ctx_old.jref,'svd_state') || ...
            isempty(ctx_old.jref.svd_state)
        error('jref_augmented_init requires ctx.jref.svd_state from freeze_jacobian.');
    end

    C_new = gather(jref_matrix_for_term_local(ctx_old,term_new,cfg));
    state_old = ctx_old.jref.svd_state;
    [state_aug,~,~,~] = append_jref_svd_state_local(C_new,state_old,cfg.orth_cfg);

    delta = apply_jref_svd_state_local(state_aug,-R);
    n_old = numel(ctx_old.coef);
    n_new = size(C_new,2);
    if numel(delta) ~= n_old+n_new
        error('Incremental Jref SVD size mismatch: delta=%d, expected=%d.', ...
            numel(delta),n_old+n_new);
    end

    coef_aug = [ctx_old.coef(:); zeros(n_new,1)] + delta(:);

    ctx_try = ctx_old;
    ctx_try.terms = ctx_old.terms;
    ctx_try.terms{end+1} = term_new;
    ctx_try.coef = coef_aug;
    ctx_try.weight = coef_aug;
    [~,~,parts_old] = assemble_newton_system_ctx_local(ctx_old,cfg);
    [~,~,parts_new] = assemble_newton_system_ctx_local(ctx_try,cfg);

    info = struct();
    info.method = 'incremental_frozen_jref_svd';
    info.rel_before = parts_old.rel_res;
    info.rel_after = parts_new.rel_res;
    info.rel_value = NaN;
    info.rel_linearized = parts_new.rel_res;
    info.block_size = n_new;
    info.svd_rank = numel(state_aug.s);
    info.state = state_aug;
end

%% ============================================================
% full Newton
%% ============================================================
function ctx = newton_relax_full_ctx_local(ctx,cfg)
    cfg = fill_newton_cfg_local(cfg);

    % The Burgers pipeline can use the same stabilized fixed-operator path
    % as the other nonlinear experiments.  The legacy Newton implementation
    % remains the default when this flag is absent/false.
    if isfield(cfg,'fixed_operator') && isstruct(cfg.fixed_operator) && ...
            isfield(cfg.fixed_operator,'enabled') && cfg.fixed_operator.enabled
        ctx = fixed_operator_relax_full_ctx_local(ctx,cfg);
        return;
    end

    [R,J,parts] = assemble_newton_system_ctx_local(ctx,cfg);
    hist_rel = zeros(cfg.num_newton+1,1);
    hist_rel(1) = parts.rel_res;
    stall_count = 0;

    if cfg.verbose
        fprintf('Full Newton start: rel_res = %.6e\n',parts.rel_res);
    end

    for it = 1:cfg.num_newton
        if parts.rel_res < cfg.tau_newton
            if cfg.verbose
                fprintf('Full Newton stop: rel_res < tau_newton = %.2e\n',cfg.tau_newton);
            end
            break;
        end

        rel_old = parts.rel_res;
        [J_solve,R_solve] = compress_tall_system_local(J,R);
        clear J R
        dcoef = -linear_solve_svd_or_normal_local(J_solve,R_solve,cfg);
        clear J_solve R_solve

        alpha = 1.0;
        accept = false;

        if cfg.line_search
            for ls = 1:cfg.max_line_search
                ctx_try = ctx;
                ctx_try.coef = ctx.coef + alpha*dcoef;
                [~,~,parts_try] = assemble_newton_system_ctx_local(ctx_try,cfg);

                if parts_try.rel_res <= rel_old
                    ctx.coef = ctx_try.coef;
                    parts = parts_try;
                    accept = true;
                    break;
                end
                alpha = 0.5*alpha;
            end
        else
            ctx.coef = ctx.coef + dcoef;
            [~,~,parts] = assemble_newton_system_ctx_local(ctx,cfg);
            accept = true;
        end

        if ~accept
            if cfg.verbose
                fprintf('Full Newton %d: line search failed, stop. rel_res = %.6e\n',it,parts.rel_res);
            end
            break;
        end

        [R,J,parts] = assemble_newton_system_ctx_local(ctx,cfg);
        hist_rel(it+1) = parts.rel_res;

        rel_improve = (rel_old - parts.rel_res)/(rel_old + eps);

        if cfg.verbose
            fprintf('Full Newton %d: alpha=%.3f, rel_res=%.6e, improve=%.4e\n', ...
                it,alpha,parts.rel_res,rel_improve);
        end

        if it >= cfg.min_newton_iter
            if rel_improve < cfg.tol_newton_improve
                stall_count = stall_count + 1;
            else
                stall_count = 0;
            end

            if stall_count >= cfg.stall_patience
                if cfg.verbose
                    fprintf('Full Newton stop: residual stagnates for %d steps, last improve=%.4e < %.4e.\n', ...
                        cfg.stall_patience,rel_improve,cfg.tol_newton_improve);
                end
                break;
            end
        end
    end

    ctx.newton_info = parts;
    ctx.newton_info.hist_rel = hist_rel(hist_rel>0);
    ctx.weight = ctx.coef;
end

%% ============================================================
% stabilized fixed-operator Picard
%% ============================================================
function ctx = fixed_operator_relax_full_ctx_local(ctx,cfg)
    fc = cfg.fixed_operator;
    if ~isfield(fc,'sigma'), fc.sigma = 1; end
    if ~isfield(fc,'omega'), fc.omega = 0.7; end
    if ~isfield(fc,'omega_min'), fc.omega_min = 1e-3; end
    if ~isfield(fc,'maxit'), fc.maxit = 25; end
    if ~isfield(fc,'max_line_search'), fc.max_line_search = 10; end
    if ~isfield(fc,'accept_increase'), fc.accept_increase = 0; end
    if ~isfield(fc,'tol'), fc.tol = 1e-7; end
    if ~isfield(fc,'verbose'), fc.verbose = cfg.verbose; end

    X = ctx.grid.XT_int;
    epsv = ctx.problem.eps_burgers;
    state_old = getfield_default_local(ctx,'fixed_operator_svd_state',[]);
    ncols_full = numel(ctx.coef);

    % Keep the fixed-operator response matrices with the context.  After an
    % accepted RBE block only the new columns are evaluated; the old columns
    % are reused verbatim.  A DDAD/SFR basis rebuild explicitly clears this
    % cache below, so this is only a computational cache, not a change in the
    % trial space or in the solver tolerances.
    basis_cache_old = getfield_default_local(ctx,'fixed_operator_basis_cache',[]);
    use_appended_basis_cache = false;
    if isstruct(basis_cache_old) && isfield(basis_cache_old,'term_count') && ...
            isfield(basis_cache_old,'ncols') && ...
            basis_cache_old.term_count == numel(ctx.terms)-1 && ...
            basis_cache_old.ncols < ncols_full && ...
            isfield(basis_cache_old,'P0_blocks') && ...
            isfield(basis_cache_old,'Bd_blocks') && ...
            isfield(basis_cache_old,'Ic_blocks') && ...
            size(basis_cache_old.P0_blocks{1},1) == size(X,1) && ...
            size(basis_cache_old.Bd_blocks{1},1) == size(ctx.grid.X_bd,1) && ...
            size(basis_cache_old.Ic_blocks{1},1) == size(ctx.grid.X_ic,1)
        term_new = ctx.terms{end};
        m_new = term_num_cols_local(term_new);
        if basis_cache_old.ncols + m_new == ncols_full
            Phi_new = eval_terms_local({term_new},X,0,0);
            Phit_new = eval_terms_local({term_new},X,1,3);
            Phixx_new = eval_terms_local({term_new},X,2,1);
            Phiyy_new = eval_terms_local({term_new},X,2,2);
            Phix_new = eval_terms_local({term_new},X,1,1);
            Phiy_new = eval_terms_local({term_new},X,1,2);
            Abd_new = eval_terms_local({term_new},ctx.grid.X_bd,0,0);
            Aic_new = eval_terms_local({term_new},ctx.grid.X_ic,0,0);
            basis_cache = basis_cache_old;
            basis_cache.P0_blocks{end+1} = Phi_new;
            basis_cache.Pt_blocks{end+1} = Phit_new;
            basis_cache.Pxx_blocks{end+1} = Phixx_new;
            basis_cache.Pyy_blocks{end+1} = Phiyy_new;
            basis_cache.Px_blocks{end+1} = Phix_new;
            basis_cache.Py_blocks{end+1} = Phiy_new;
            basis_cache.Bd_blocks{end+1} = Abd_new;
            basis_cache.Ic_blocks{end+1} = Aic_new;
            basis_cache.term_count = numel(ctx.terms);
            basis_cache.ncols = ncols_full;
            use_appended_basis_cache = true;
        end
    end
    if ~use_appended_basis_cache
        % Initial solves and SFR/DDAD rebuilds evaluate the complete basis once.
        Phi_full = eval_terms_local(ctx.terms,X,0,0);
        Phit_full = eval_terms_local(ctx.terms,X,1,3);
        Phixx_full = eval_terms_local(ctx.terms,X,2,1);
        Phiyy_full = eval_terms_local(ctx.terms,X,2,2);
        Phix_full = eval_terms_local(ctx.terms,X,1,1);
        Phiy_full = eval_terms_local(ctx.terms,X,1,2);
        Abd_full = eval_terms_local(ctx.terms,ctx.grid.X_bd,0,0);
        Aic_full = eval_terms_local(ctx.terms,ctx.grid.X_ic,0,0);
        basis_cache = struct('P0_blocks',{{Phi_full}},'Pt_blocks',{{Phit_full}}, ...
            'Pxx_blocks',{{Phixx_full}},'Pyy_blocks',{{Phiyy_full}}, ...
            'Px_blocks',{{Phix_full}},'Py_blocks',{{Phiy_full}}, ...
            'Bd_blocks',{{Abd_full}},'Ic_blocks',{{Aic_full}}, ...
            'term_count',numel(ctx.terms), ...
            'ncols',ncols_full);
    end
    can_append_block = fixed_svd_state_compatible_local(state_old, ...
        size(X,1)+size(ctx.grid.X_bd,1)+size(ctx.grid.X_ic,1),ncols_full) && ...
        isfield(state_old,'term_count') && ...
        numel(ctx.terms) == state_old.term_count + 1;

    if can_append_block
        % RBE appends exactly one term.  Assemble only that term's fixed
        % operator and boundary/initial rows; old columns stay in U/S/V.
        term_new = ctx.terms{end};
        Phi = eval_terms_local({term_new},X,0,0);
        Phit = eval_terms_local({term_new},X,1,3);
        Phixx = eval_terms_local({term_new},X,2,1);
        Phiyy = eval_terms_local({term_new},X,2,2);
        Aint_new = Phit - epsv*(Phixx + Phiyy);
        Abd_new = eval_terms_local({term_new},ctx.grid.X_bd,0,0);
        Aic_new = eval_terms_local({term_new},ctx.grid.X_ic,0,0);
        Anew_fix = [Aint_new + fc.sigma*Phi; cfg.eta_bc*Abd_new; cfg.eta_ic*Aic_new];
        [Ufix,sfix,Vfix,svd_state,svd_mode] = append_fixed_svd_state_local(Anew_fix,state_old);
        clear Phit Phixx Phiyy Abd_new Aic_new Aint_new Phi Anew_fix
    else
        Aint = Phit_full - epsv*(Phixx_full + Phiyy_full);
        Afix = [Aint + fc.sigma*Phi_full; cfg.eta_bc*Abd_full; cfg.eta_ic*Aic_full];
        clear Aint

        % The initial space (or a rebuilt basis) is factorized once.  Later
        % RBE blocks enter through append_fixed_svd_state_local above.
        [Ufix,sfix,Vfix,svd_state,svd_mode] = update_fixed_svd_state_local( ...
            gather(Afix),state_old);
        clear Afix
    end
    svd_state.term_count = numel(ctx.terms);

    [parts] = burgers_residual_parts_local(ctx,cfg,basis_cache);
    rel0 = max(parts.rel_res,eps);
    hist = nan(fc.maxit+1,3); hist(1,:) = [parts.rel_res,fc.omega,0];
    omega = fc.omega;

    if fc.verbose
        fprintf('FixedOp Burgers start: rel_res = %.6e, sigma = %.3e, factor=%s\n', ...
            parts.rel_res,fc.sigma,svd_mode);
    end

    for it = 1:fc.maxit
        u = apply_basis_blocks_local(basis_cache.P0_blocks,ctx.coef);
        ux = apply_basis_blocks_local(basis_cache.Px_blocks,ctx.coef);
        uy = apply_basis_blocks_local(basis_cache.Py_blocks,ctx.coef);
        Nold = u.*(ux+uy);
        rhs = [-Nold + fc.sigma*u; cfg.eta_bc*ctx.grid.g_bd; cfg.eta_ic*ctx.grid.u_ic];
        % Only the right-hand side changes between Picard steps.
        coef_trial = apply_fixed_svd_state_local(Ufix,sfix,Vfix,gather(rhs));

        rel_old = parts.rel_res;
        accepted = false; omega_try = omega;
        for ls = 1:fc.max_line_search
            ctx_try = ctx;
            ctx_try.coef = (1-omega_try)*ctx.coef + omega_try*coef_trial;
            parts_try = burgers_residual_parts_local(ctx_try,cfg,basis_cache);
            if parts_try.rel_res <= rel_old*(1+fc.accept_increase)
                accepted = true; break;
            end
            omega_try = 0.5*omega_try;
            if omega_try < fc.omega_min, break; end
        end
        if ~accepted
            if fc.verbose, fprintf('FixedOp Burgers stop: no improving step at it=%d.\n',it); end
            break;
        end

        ctx = ctx_try; parts = parts_try;
        rel_change = (rel_old-parts.rel_res)/(rel_old+eps);
        omega = min(fc.omega,max(omega_try,fc.omega_min));
        hist(it+1,:) = [parts.rel_res,omega_try,rel_change];
        if fc.verbose
            fprintf('FixedOp Burgers %d: rel_res=%.6e, omega=%.3g, improve=%.4e\n', ...
                it,parts.rel_res,omega_try,rel_change);
        end
        if parts.rel_res <= fc.tol || parts.rel_res/(rel0 + eps) <= fc.tol
            break;
        end
    end

    ctx.newton_info = parts;
    ctx.newton_info.hist_rel = hist(all(isfinite(hist),2),1);
    ctx.fixed_operator_svd_state = svd_state;
    ctx.fixed_operator_basis_cache = basis_cache;
    ctx.fixed_operator_info = struct('sigma',fc.sigma,'omega',omega, ...
        'iter',sum(isfinite(hist(:,1)))-1,'history',hist(all(isfinite(hist),2),:), ...
        'factorization',svd_mode,'residual_only_line_search',true);
    ctx.weight = ctx.coef;
end

function [U,s,V,state,mode] = update_fixed_svd_state_local(A,state_old)
%UPDATE_FIXED_SVD_STATE_LOCAL Full initialization followed by incremental SVD.
    nrow = size(A,1);
    ncol = size(A,2);
    valid = isstruct(state_old) && isfield(state_old,'U') && ...
        isfield(state_old,'s') && isfield(state_old,'V') && ...
        isfield(state_old,'nrows') && isfield(state_old,'ncols') && ...
        state_old.nrows == nrow && state_old.ncols <= ncol && ...
        ~isempty(state_old.U) && ~isempty(state_old.s) && ~isempty(state_old.V);
    if ~valid
        [U,S,V] = svd(A,'econ');
        s = diag(S);
        state = struct('U',U,'s',s,'V',V,'nrows',size(U,1), ...
            'ncols',size(V,2),'mode','full_svd');
        mode = 'full_svd';
        return;
    end
    nold = state_old.ncols;
    if nold == ncol
        U = state_old.U; s = state_old.s; V = state_old.V;
        state = state_old;
        mode = 'reused_svd';
        return;
    end
    [U,s,V,state,mode] = append_fixed_svd_state_local(A(:,nold+1:ncol),state_old);
end

function tf = fixed_svd_state_compatible_local(state,nrow,ncol)
    tf = isstruct(state) && isfield(state,'U') && isfield(state,'s') && ...
        isfield(state,'V') && isfield(state,'nrows') && isfield(state,'ncols') && ...
        state.nrows == nrow && state.ncols < ncol && ...
        ~isempty(state.U) && ~isempty(state.s) && ~isempty(state.V);
end

function [U,s,V,state,mode] = append_fixed_svd_state_local(Anew,state_old)
%APPEND_FIXED_SVD_STATE_LOCAL Update the SVD using only a new block.
    Uold = state_old.U;
    sold = state_old.s(:);
    Vold = state_old.V;
    Anew = gather(Anew);
    P = Uold' * Anew;
    Aperp = Anew - Uold * P;
    P2 = Uold' * Aperp;
    Aperp = Aperp - Uold * P2;
    P = P + P2;
    [Uperp,Sperp,Vperp] = svd(Aperp,'econ');
    sperp = diag(Sperp);
    k = size(Anew,2);
    r = numel(sold);
    K = [diag(sold), P; zeros(k,r), diag(sperp)*Vperp'];
    [Uk,Sk,Vk] = svd(K,'econ');
    U = [Uold,Uperp] * Uk;
    s = diag(Sk);
    V = blkdiag(Vold,eye(k)) * Vk;
    state = struct('U',U,'s',s,'V',V,'nrows',size(U,1), ...
        'ncols',size(V,2),'mode','incremental_block_svd');
    mode = 'incremental_block_svd';
end

function coef = apply_fixed_svd_state_local(U,s,V,rhs)
    z = U' * rhs;
    tol = max(size(U)) * eps(max([s(:);1]));
    invs = zeros(size(s));
    keep = abs(s) > tol;
    invs(keep) = 1 ./ s(keep);
    coef = V * (invs .* z);
end

function parts = burgers_residual_parts_local(ctx,cfg,basis_cache)
%BURGERS_RESIDUAL_PARTS_LOCAL Residual-only evaluation for fixed Picard.
% Unlike the Newton assembly, this routine does not build the Jacobian.  It
% is used by the fixed-operator line search because the left-hand operator
% has already been factorized and only the nonlinear residual is needed.
    cfg = fill_newton_cfg_local(cfg);
    grid = ctx.grid;
    terms = ctx.terms;
    coef = ctx.coef;
    epsv = ctx.problem.eps_burgers;
    X = grid.XT_int;
    if nargin < 3 || isempty(basis_cache)
        basis_cache = struct();
        basis_cache.P0 = eval_terms_local(terms,X,0,0);
        basis_cache.Px = eval_terms_local(terms,X,1,1);
        basis_cache.Py = eval_terms_local(terms,X,1,2);
        basis_cache.Pt = eval_terms_local(terms,X,1,3);
        basis_cache.Pxx = eval_terms_local(terms,X,2,1);
        basis_cache.Pyy = eval_terms_local(terms,X,2,2);
        basis_cache.Bd = eval_terms_local(terms,grid.X_bd,0,0);
        basis_cache.Ic = eval_terms_local(terms,grid.X_ic,0,0);
    end

    if isfield(basis_cache,'P0_blocks')
        u   = apply_basis_blocks_local(basis_cache.P0_blocks,coef);
        ux  = apply_basis_blocks_local(basis_cache.Px_blocks,coef);
        uy  = apply_basis_blocks_local(basis_cache.Py_blocks,coef);
        ut  = apply_basis_blocks_local(basis_cache.Pt_blocks,coef);
        uxx = apply_basis_blocks_local(basis_cache.Pxx_blocks,coef);
        uyy = apply_basis_blocks_local(basis_cache.Pyy_blocks,coef);
        ubd = apply_basis_blocks_local(basis_cache.Bd_blocks,coef);
        uic = apply_basis_blocks_local(basis_cache.Ic_blocks,coef);
    else
        u   = basis_cache.P0*coef;
        ux  = basis_cache.Px*coef;
        uy  = basis_cache.Py*coef;
        ut  = basis_cache.Pt*coef;
        uxx = basis_cache.Pxx*coef;
        uyy = basis_cache.Pyy*coef;
        ubd = basis_cache.Bd*coef;
        uic = basis_cache.Ic*coef;
    end

    F = ut + u.*(ux+uy) - epsv*(uxx+uyy);
    Rbd = ubd - grid.g_bd;
    Ric = uic - grid.u_ic;

    scale_vec = [zeros(size(F));
                 cfg.eta_bc*grid.g_bd;
                 cfg.eta_ic*grid.u_ic];
    R = [F; cfg.eta_bc*Rbd; cfg.eta_ic*Ric];
    rhs_norm = norm(scale_vec) + sqrt(length(R)) + eps;

    parts = struct();
    parts.F_int = F;
    parts.u_int = u;
    parts.ux_int = ux;
    parts.uy_int = uy;
    parts.rel_res = norm(R)/(rhs_norm + eps);
    parts.res_norm = norm(R);
    parts.pde_res_norm = norm(F);
    parts.ic_res_norm = norm(Ric);
    parts.bc_res_norm = norm(Rbd);
end

function cfg = fill_newton_cfg_local(cfg)
    if nargin < 1 || isempty(cfg), cfg = struct(); end
    if ~isfield(cfg,'num_newton'), cfg.num_newton = 6; end
    if ~isfield(cfg,'tau_newton'), cfg.tau_newton = 1e-8; end
    if ~isfield(cfg,'lambda'), cfg.lambda = 1e-6; end
    if ~isfield(cfg,'solve_lambda'), cfg.solve_lambda = 0; end
    if ~isfield(cfg,'line_search'), cfg.line_search = true; end
    if ~isfield(cfg,'max_line_search'), cfg.max_line_search = 8; end
    if ~isfield(cfg,'eta_bc'), cfg.eta_bc = 100; end
    if ~isfield(cfg,'eta_ic'), cfg.eta_ic = 100; end
    if ~isfield(cfg,'verbose'), cfg.verbose = true; end
    if ~isfield(cfg,'linear_solver'), cfg.linear_solver = 'svd'; end
    if ~isfield(cfg,'svd_mode'), cfg.svd_mode = 'rel_tol'; end
    if ~isfield(cfg,'svd_rel_tol'), cfg.svd_rel_tol = 1e-10; end
    if ~isfield(cfg,'svd_energy_ratio'), cfg.svd_energy_ratio = 0.99999999; end
    if ~isfield(cfg,'svd_fixed_rank'), cfg.svd_fixed_rank = 1200; end
    if ~isfield(cfg,'tol_newton_improve'), cfg.tol_newton_improve = 1e-4; end
    if ~isfield(cfg,'stall_patience'), cfg.stall_patience = 2; end
    if ~isfield(cfg,'min_newton_iter'), cfg.min_newton_iter = 2; end
end

%% ============================================================
% residual and Jacobian assembly
%% ============================================================
function [R,J,parts] = assemble_newton_system_ctx_local(ctx,cfg)
    cfg = fill_newton_cfg_local(cfg);
    problem = ctx.problem;
    grid = ctx.grid;
    terms = ctx.terms;
    coef = ctx.coef;

    epsv = problem.eps_burgers;
    X = grid.XT_int;

    u    = eval_solution_local(terms,coef,X,0,0);
    ux   = eval_solution_local(terms,coef,X,1,1);
    uy   = eval_solution_local(terms,coef,X,1,2);
    ut   = eval_solution_local(terms,coef,X,1,3);
    uxx  = eval_solution_local(terms,coef,X,2,1);
    uyy  = eval_solution_local(terms,coef,X,2,2);

    F = ut + u.*(ux+uy) - epsv*(uxx+uyy);
    clear ut uxx uyy

    % Assemble the same Jacobian one derivative block at a time.  Keeping
    % all six dense basis matrices simultaneously can exhaust memory.
    Phi = eval_terms_local(terms,X,0,0);
    Jint = (ux+uy).*Phi;
    clear Phi
    Phix = eval_terms_local(terms,X,1,1);
    Jint = Jint + u.*Phix;
    clear Phix
    Phiy = eval_terms_local(terms,X,1,2);
    Jint = Jint + u.*Phiy;
    clear Phiy
    Phit = eval_terms_local(terms,X,1,3);
    Jint = Jint + Phit;
    clear Phit
    Phixx = eval_terms_local(terms,X,2,1);
    Jint = Jint - epsv*Phixx;
    clear Phixx
    Phiyy = eval_terms_local(terms,X,2,2);
    Jint = Jint - epsv*Phiyy;
    clear Phiyy

    ubd = eval_solution_local(terms,coef,grid.X_bd,0,0);
    Rbd = ubd - grid.g_bd;
    Jbd = eval_terms_local(terms,grid.X_bd,0,0);

    uic = eval_solution_local(terms,coef,grid.X_ic,0,0);
    Ric = uic - grid.u_ic;
    Jic = eval_terms_local(terms,grid.X_ic,0,0);

    R = [F;
         cfg.eta_bc*Rbd;
         cfg.eta_ic*Ric];

    J = [Jint;
         cfg.eta_bc*Jbd;
         cfg.eta_ic*Jic];
    clear Jint Jbd Jic

    scale_vec = [zeros(size(F));
                 cfg.eta_bc*grid.g_bd;
                 cfg.eta_ic*grid.u_ic];
    rhs_norm = norm(scale_vec) + sqrt(length(R)) + eps;

    parts = struct();
    parts.F_int = F;
    parts.u_int = u;
    parts.ux_int = ux;
    parts.uy_int = uy;
    parts.rel_res = norm(R)/(rhs_norm + eps);
    parts.res_norm = norm(R);
    parts.pde_res_norm = norm(F);
    parts.ic_res_norm = norm(Ric);
    parts.bc_res_norm = norm(Rbd);
end

function U_grid = grid_solution_local(ctx)
    val = eval_solution_local(ctx.terms,ctx.coef,ctx.grid.XT_grid,0,0);
    if isfield(ctx.grid,'Nx') && isfield(ctx.grid,'Ny') && isfield(ctx.grid,'Nt')
        U_grid = reshape(val,ctx.grid.Nx,ctx.grid.Ny,ctx.grid.Nt);
    else
        U_grid = val;
    end
end

function R_grid = residual_grid_local(ctx)
    X = ctx.grid.XT_grid;
    terms = ctx.terms;
    coef = ctx.coef;
    epsv = ctx.problem.eps_burgers;

    u    = eval_solution_local(terms,coef,X,0,0);
    ux   = eval_solution_local(terms,coef,X,1,1);
    uy   = eval_solution_local(terms,coef,X,1,2);
    ut   = eval_solution_local(terms,coef,X,1,3);
    uxx  = eval_solution_local(terms,coef,X,2,1);
    uyy  = eval_solution_local(terms,coef,X,2,2);

    val = ut + u.*(ux+uy) - epsv*(uxx+uyy);

    if isfield(ctx.grid,'Nx') && isfield(ctx.grid,'Ny') && isfield(ctx.grid,'Nt')
        R_grid = reshape(val,ctx.grid.Nx,ctx.grid.Ny,ctx.grid.Nt);
    else
        R_grid = val;
    end
end

%% ============================================================
% block local / frozen J_ref tools
%% ============================================================
function ctx = activate_block_ls_local(ctx,term,cfg)
    cfg = fill_newton_cfg_local(cfg);
    if ~isfield(cfg,'solve_lambda'), cfg.solve_lambda = 0; end
    if ~isfield(cfg,'local_ls_scale'), cfg.local_ls_scale = 1.0; end

    m_new = term_num_cols_local(term);
    M = numel(ctx.coef);
    idx_new = (M-m_new+1):M;

    [R,~,parts_old] = assemble_newton_system_ctx_local(ctx,cfg);
    rel_old = parts_old.rel_res;

    C = current_jacobian_for_term_local(ctx,term,cfg);
    [C_solve,R_solve] = compress_tall_system_local(C,-R);
    clear C R
    a = linear_solve_svd_or_normal_local(C_solve,R_solve,cfg);
    clear C_solve R_solve
    a = cfg.local_ls_scale*a(:);

    alpha = 1.0;
    accept = false;
    parts = parts_old;

    if cfg.line_search
        for ls = 1:cfg.max_line_search
            ctx_try = ctx;
            ctx_try.coef(idx_new) = ctx.coef(idx_new) + alpha*a;
            [~,~,parts_try] = assemble_newton_system_ctx_local(ctx_try,cfg);
            if parts_try.rel_res <= rel_old
                ctx.coef = ctx_try.coef;
                parts = parts_try;
                accept = true;
                break;
            end
            alpha = 0.5*alpha;
        end
    else
        ctx.coef(idx_new) = ctx.coef(idx_new) + a;
        [~,~,parts] = assemble_newton_system_ctx_local(ctx,cfg);
        accept = true;
    end

    rel_new = parts.rel_res;
    improve = (rel_old - rel_new)/(rel_old + eps);
    if cfg.verbose
        if accept
            fprintf('Block-local LS: alpha=%.3f, rel_res %.6e -> %.6e, improve=%.4e\n', ...
                alpha,rel_old,rel_new,improve);
        else
            fprintf('Block-local LS: line search failed, rel_res remains %.6e\n',rel_old);
        end
    end

    ctx.newton_info = parts;
    ctx.newton_info.hist_rel = [rel_old; rel_new];
    ctx.weight = ctx.coef;
    ctx.block_ls_info = struct('accept',accept,'alpha',alpha, ...
        'rel_old',rel_old,'rel_new',rel_new,'improve',improve);
end

function A = current_jacobian_for_term_local(ctx,term,cfg)
    cfg = fill_newton_cfg_local(cfg);
    X = ctx.grid.XT_int;
    epsv = ctx.problem.eps_burgers;

    u  = eval_solution_local(ctx.terms,ctx.coef,X,0,0);
    ux = eval_solution_local(ctx.terms,ctx.coef,X,1,1);
    uy = eval_solution_local(ctx.terms,ctx.coef,X,1,2);

    Phi = eval_terms_local({term},X,0,0);
    Aint = (ux+uy).*Phi;
    clear Phi
    Phix = eval_terms_local({term},X,1,1);
    Aint = Aint + u.*Phix;
    clear Phix
    Phiy = eval_terms_local({term},X,1,2);
    Aint = Aint + u.*Phiy;
    clear Phiy
    Phit = eval_terms_local({term},X,1,3);
    Aint = Aint + Phit;
    clear Phit
    Phixx = eval_terms_local({term},X,2,1);
    Aint = Aint - epsv*Phixx;
    clear Phixx
    Phiyy = eval_terms_local({term},X,2,2);
    Aint = Aint - epsv*Phiyy;
    clear Phiyy
    Abd = eval_terms_local({term},ctx.grid.X_bd,0,0);
    Aic = eval_terms_local({term},ctx.grid.X_ic,0,0);

    A = [Aint;
         cfg.eta_bc*Abd;
         cfg.eta_ic*Aic];
end

function ctx = freeze_jacobian_local(ctx,cfg)
    cfg = fill_newton_cfg_local(cfg);
    if ~isfield(cfg,'orth_cfg'), cfg.orth_cfg = ctx.orth_cfg; end

    Aref = jref_matrix_for_terms_local(ctx,ctx.terms,cfg);
    nArefCols = size(Aref,2);
    [Q,V,s,rank_keep] = svd_truncate_local(Aref,cfg.orth_cfg);

    U_grid = grid_solution_local(ctx);
    X = ctx.grid.XT_int;
    u_ref  = eval_solution_local(ctx.terms,ctx.coef,X,0,0);
    ux_ref = eval_solution_local(ctx.terms,ctx.coef,X,1,1);
    uy_ref = eval_solution_local(ctx.terms,ctx.coef,X,1,2);

    ctx.jref = struct();
    % Keep the frozen orthogonal basis by blocks in jref.parts.  A second
    % concatenated copy grows expensive after several accepted families.
    ctx.jref.Q = [];
    ctx.jref.parts = {};
    ctx.jref.parts{1} = struct('cols',1:nArefCols,'Q',Q,'V',V,'s',s);
    % Keep a compact incremental SVD state for coefficient-space frozen-Jref
    % projections.  ncols is the number of raw coefficient columns; V may
    % have fewer columns when the orthogonalization is rank-truncated.
    ctx.jref.svd_state = struct('U',Q,'s',s,'V',V, ...
        'nrows',size(Aref,1),'ncols',nArefCols,'mode','full_svd');
    ctx.jref.rank_keep = rank_keep;
    ctx.jref.U_ref_grid = U_grid;
    ctx.jref.u_ref_int = u_ref;
    ctx.jref.ux_ref_int = ux_ref;
    ctx.jref.uy_ref_int = uy_ref;
    ctx.jref.num_blocks_since_refresh = 0;
    ctx.jref.cfg = cfg;

    clear Aref Q V s
    [~,~,parts] = assemble_newton_system_ctx_local(ctx,cfg);
    fprintf('\n========== Freeze J_ref and big SVD ==========\n');
    fprintf('current columns = %d, rank_keep = %d, rel_res = %.6e\n',nArefCols,rank_keep,parts.rel_res);
end

function ctx = append_jref_block_local(ctx,term,cfg)
    cfg = fill_newton_cfg_local(cfg);
    if ~isfield(cfg,'orth_cfg'), cfg.orth_cfg = ctx.orth_cfg; end
    if ~isfield(ctx,'jref') || isempty(ctx.jref)
        ctx = freeze_jacobian_local(ctx,cfg);
        ctx.jref_append.valid = true;
        return;
    end

    C = jref_matrix_for_term_local(ctx,term,cfg);
    if ~isfield(ctx.jref,'svd_state') || isempty(ctx.jref.svd_state)
        % This should only occur for an old context created before the
        % incremental frozen-Jref state was introduced.
        ctx = freeze_jacobian_local(ctx,cfg);
    end
    [state_new,Qnew,Vnew,snew] = append_jref_svd_state_local( ...
        C,ctx.jref.svd_state,cfg.orth_cfg);
    rank_keep = numel(snew);

    valid = rank_keep >= 1 && ~isempty(snew);
    ctx.jref_append = struct('valid',valid,'rank_keep',rank_keep);
    if ~valid, return; end

    oldM = numel(ctx.coef) - term_num_cols_local(term);
    newCols = oldM + (1:term_num_cols_local(term));

    ctx.jref.parts{end+1} = struct('cols',newCols,'Q',Qnew,'V',Vnew,'s',snew);
    ctx.jref.svd_state = state_new;
    clear C Vnew snew
    ctx.jref.rank_keep = ctx.jref.rank_keep + rank_keep;
    ctx.jref.num_blocks_since_refresh = ctx.jref.num_blocks_since_refresh + 1;
end

function A = jref_matrix_for_terms_local(ctx,terms,cfg)
    A = eval_jref_matrix_local(ctx,terms,cfg);
end

function A = jref_matrix_for_term_local(ctx,term,cfg)
    A = eval_jref_matrix_local(ctx,{term},cfg);
end

function A = eval_jref_matrix_local(ctx,terms,cfg)
    cfg = fill_newton_cfg_local(cfg);
    X = ctx.grid.XT_int;
    epsv = ctx.problem.eps_burgers;

    if isfield(ctx,'jref') && isfield(ctx.jref,'u_ref_int') && ~isempty(ctx.jref.u_ref_int)
        u_ref  = ctx.jref.u_ref_int;
        ux_ref = ctx.jref.ux_ref_int;
        uy_ref = ctx.jref.uy_ref_int;
    else
        u_ref  = eval_solution_local(ctx.terms,ctx.coef,X,0,0);
        ux_ref = eval_solution_local(ctx.terms,ctx.coef,X,1,1);
        uy_ref = eval_solution_local(ctx.terms,ctx.coef,X,1,2);
    end

    Phi = eval_terms_local(terms,X,0,0);
    Aint = (ux_ref+uy_ref).*Phi;
    clear Phi
    Phix = eval_terms_local(terms,X,1,1);
    Aint = Aint + u_ref.*Phix;
    clear Phix
    Phiy = eval_terms_local(terms,X,1,2);
    Aint = Aint + u_ref.*Phiy;
    clear Phiy
    Phit = eval_terms_local(terms,X,1,3);
    Aint = Aint + Phit;
    clear Phit
    Phixx = eval_terms_local(terms,X,2,1);
    Aint = Aint - epsv*Phixx;
    clear Phixx
    Phiyy = eval_terms_local(terms,X,2,2);
    Aint = Aint - epsv*Phiyy;
    clear Phiyy

    Abd = eval_terms_local(terms,ctx.grid.X_bd,0,0);
    Aic = eval_terms_local(terms,ctx.grid.X_ic,0,0);

    A = [Aint;
         cfg.eta_bc*Abd;
         cfg.eta_ic*Aic];
end

%% ============================================================
% frozen inexact Newton
%% ============================================================
function ctx = newton_relax_inexact_ctx_local(ctx,cfg)
    cfg = fill_newton_cfg_local(cfg);
    if ~isfield(ctx,'jref') || isempty(ctx.jref)
        ctx = freeze_jacobian_local(ctx,cfg);
    end

    [R,~,parts] = assemble_newton_system_ctx_local(ctx,cfg);
    hist_rel = zeros(cfg.num_newton+1,1);
    hist_rel(1) = parts.rel_res;
    stall_count = 0;

    if cfg.verbose
        fprintf('Inexact Newton start: rel_res = %.6e\n',parts.rel_res);
    end

    for it = 1:cfg.num_newton
        if parts.rel_res < cfg.tau_newton
            if cfg.verbose, fprintf('Inexact Newton stop: rel_res < tau_newton.\n'); end
            break;
        end

        rel_old = parts.rel_res;
        dcoef = -frozen_pinv_apply_local(ctx,R);

        alpha = 1.0;
        accept = false;

        if cfg.line_search
            for ls = 1:cfg.max_line_search
                ctx_try = ctx;
                ctx_try.coef = ctx.coef + alpha*dcoef;
                [~,~,parts_try] = assemble_newton_system_ctx_local(ctx_try,cfg);
                if parts_try.rel_res <= rel_old
                    ctx.coef = ctx_try.coef;
                    parts = parts_try;
                    accept = true;
                    break;
                end
                alpha = 0.5*alpha;
            end
        else
            ctx.coef = ctx.coef + dcoef;
            [~,~,parts] = assemble_newton_system_ctx_local(ctx,cfg);
            accept = true;
        end

        if ~accept
            if cfg.verbose, fprintf('Inexact Newton %d: line search failed.\n',it); end
            break;
        end

        [R,~,parts] = assemble_newton_system_ctx_local(ctx,cfg);
        hist_rel(it+1) = parts.rel_res;
        rel_improve = (rel_old - parts.rel_res)/(rel_old + eps);

        if cfg.verbose
            fprintf('Inexact Newton %d: alpha=%.3f, rel_res=%.6e, improve=%.4e\n', ...
                it,alpha,parts.rel_res,rel_improve);
        end

        if it >= cfg.min_newton_iter
            if rel_improve < cfg.tol_newton_improve
                stall_count = stall_count + 1;
            else
                stall_count = 0;
            end
            if stall_count >= cfg.stall_patience
                break;
            end
        end
    end

    ctx.newton_info = parts;
    ctx.newton_info.hist_rel = hist_rel(hist_rel>0);
    ctx.weight = ctx.coef;
end

function dcoef = frozen_pinv_apply_local(ctx,R)
    if isfield(ctx,'jref') && isfield(ctx.jref,'svd_state') && ...
            ~isempty(ctx.jref.svd_state)
        dcoef = apply_jref_svd_state_local(ctx.jref.svd_state,R(:));
        return;
    end
    dcoef = zeros(numel(ctx.coef),1);
    for kk = 1:numel(ctx.jref.parts)
        part = ctx.jref.parts{kk};
        cols = part.cols;
        qtr = part.Q.'*R(:);
        local = part.V * ((1./(part.s+eps)).*qtr);
        dcoef(cols) = dcoef(cols) + local;
    end
end

function Cperp = project_off_jref_parts_local(ctx,C)
%PROJECT_OFF_JREF_PARTS_LOCAL residualize against frozen Q blocks.
    Cperp = C;
    row_chunk = 4096;
    if isfield(ctx,'jref') && isfield(ctx.jref,'svd_state') && ...
            ~isempty(ctx.jref.svd_state)
        Q = ctx.jref.svd_state.U;
        H = Q.'*C;
        for row0 = 1:row_chunk:size(C,1)
            rows = row0:min(row0+row_chunk-1,size(C,1));
            Cperp(rows,:) = Cperp(rows,:) - Q(rows,:)*H;
        end
        return;
    end
    if ~isfield(ctx,'jref') || ~isfield(ctx.jref,'parts')
        return;
    end
    for kk = 1:numel(ctx.jref.parts)
        Qk = ctx.jref.parts{kk}.Q;
        Hk = Qk.'*C;
        for row0 = 1:row_chunk:size(C,1)
            rows = row0:min(row0+row_chunk-1,size(C,1));
            Cperp(rows,:) = Cperp(rows,:) - Qk(rows,:)*Hk;
        end
        clear Qk Hk
    end
end

function [Q,V,s,r_keep] = svd_truncate_local(A,orth_cfg)
    if nargin < 2 || isempty(orth_cfg), orth_cfg = struct(); end
    if ~isfield(orth_cfg,'mode'), orth_cfg.mode = 'rel_tol'; end
    if ~isfield(orth_cfg,'rel_tol'), orth_cfg.rel_tol = 1e-10; end
    if ~isfield(orth_cfg,'fixed_rank'), orth_cfg.fixed_rank = 1000; end
    if ~isfield(orth_cfg,'energy_ratio'), orth_cfg.energy_ratio = 0.99999999; end

    A = gather(A);
    nArows = size(A,1);
    nAcols = size(A,2);
    [U,S,Vfull] = svd(A,'econ');
    sfull = diag(S);
    clear A S

    if isempty(sfull) || sfull(1) == 0
        Q = zeros(nArows,0);
        V = zeros(nAcols,0);
        s = [];
        r_keep = 0;
        return;
    end

    switch lower(orth_cfg.mode)
        case 'fixed_rank'
            r_keep = min(orth_cfg.fixed_rank,length(sfull));
        case {'energy','energy_ratio'}
            e = cumsum(sfull.^2)/(sum(sfull.^2)+eps);
            r_keep = find(e >= orth_cfg.energy_ratio,1,'first');
            if isempty(r_keep), r_keep = length(sfull); end
        otherwise
            r_keep = sum(sfull >= orth_cfg.rel_tol*sfull(1));
    end

    r_keep = max(0,r_keep);
    if r_keep == size(U,2)
        Q = U;
        V = Vfull;
    else
        Q = U(:,1:r_keep);
        clear U
        V = Vfull(:,1:r_keep);
    end
    s = sfull(1:r_keep);
end

function [state,Qkeep,Vkeep,skeep] = append_jref_svd_state_local(Cnew,state_old,orth_cfg)
%APPEND_JREF_SVD_STATE_LOCAL Incrementally append a frozen-Jref block.
%
% Given A_old = U_old*S_old*V_old' and a new raw block Cnew, form the small
% core matrix for [A_old,Cnew].  Only the SVD of the orthogonal complement
% of Cnew and the resulting small core are computed; the complete tall
% augmented matrix is never factorized.

    if nargin < 3 || isempty(orth_cfg), orth_cfg = struct(); end
    if ~isfield(orth_cfg,'mode'), orth_cfg.mode = 'rel_tol'; end
    if ~isfield(orth_cfg,'rel_tol'), orth_cfg.rel_tol = 1e-10; end
    if ~isfield(orth_cfg,'fixed_rank'), orth_cfg.fixed_rank = 1000; end
    if ~isfield(orth_cfg,'energy_ratio'), orth_cfg.energy_ratio = 0.99999999; end

    Cnew = gather(Cnew);
    Uold = gather(state_old.U);
    sold = gather(state_old.s(:));
    Vold = gather(state_old.V);
    nold = state_old.ncols;

    if size(Vold,1) ~= nold || size(Uold,2) ~= numel(sold)
        error('Invalid frozen-Jref SVD state dimensions.');
    end
    if size(Uold,1) ~= size(Cnew,1)
        error('Frozen-Jref append row mismatch: old=%d, new=%d.', ...
            size(Uold,1),size(Cnew,1));
    end

    % Orthogonal residual of the new block relative to the old Jref space.
    P = Uold' * Cnew;
    Cperp = Cnew - Uold * P;
    [Uperp,Sperp,Vperp] = svd(Cperp,'econ');
    sperp = diag(Sperp);
    clear Cperp Sperp

    k = size(Cnew,2);
    rold = numel(sold);
    rperp = numel(sperp);

    % The lower part is k-by-k; zero rows represent discarded left-null
    % directions when an unusually short row system is supplied.
    B = zeros(k,k,'like',Cnew);
    if rperp > 0
        B(1:rperp,:) = diag(sperp) * Vperp';
    end
    K = [diag(sold), P; zeros(k,rold,'like',Cnew), B];
    [Uk,Sk,Vk] = svd(K,'econ');

    Uaug = [Uold,Uperp] * Uk;
    Vaug = blkdiag(Vold,eye(k,'like',Cnew)) * Vk;
    saug = diag(Sk);

    state = struct('U',Uaug,'s',saug,'V',Vaug, ...
        'nrows',size(Uaug,1),'ncols',nold+k, ...
        'mode','incremental_block_svd');

    % Return the truncated orthogonal-complement factors used by the
    % capture score and by the block diagnostics.  The state itself keeps
    % the full small-core rank so coefficient initialization remains exact
    % up to the solve tolerance.
    if isempty(sperp) || sperp(1) == 0
        rkeep = 0;
    else
        switch lower(orth_cfg.mode)
            case 'fixed_rank'
                rkeep = min(orth_cfg.fixed_rank,numel(sperp));
            case {'energy','energy_ratio'}
                e = cumsum(sperp.^2)/(sum(sperp.^2)+eps);
                rkeep = find(e >= orth_cfg.energy_ratio,1,'first');
                if isempty(rkeep), rkeep = numel(sperp); end
            otherwise
                rkeep = sum(sperp >= orth_cfg.rel_tol*sperp(1));
        end
        rkeep = max(0,rkeep);
    end
    Qkeep = Uperp(:,1:rkeep);
    Vkeep = Vperp(:,1:rkeep);
    skeep = sperp(1:rkeep);
end

function coef = apply_jref_svd_state_local(state,rhs)
%APPLY_JREF_SVD_STATE_LOCAL Apply the least-squares pseudoinverse of a
% compact incremental frozen-Jref SVD state.
    U = state.U;
    s = state.s(:);
    V = state.V;
    z = U' * gather(rhs(:));
    tol = max(size(U)) * eps(max([s;1]));
    invs = zeros(size(s));
    keep = abs(s) > tol;
    invs(keep) = 1 ./ s(keep);
    coef = V * (invs .* z);
end

function x = linear_solve_svd_or_normal_local(A,b,cfg)
    if nargin < 3 || isempty(cfg), cfg = struct(); end
    if ~isfield(cfg,'linear_solver'), cfg.linear_solver = 'svd'; end
    if ~isfield(cfg,'solve_lambda'), cfg.solve_lambda = 0; end
    if ~isfield(cfg,'svd_mode'), cfg.svd_mode = 'rel_tol'; end
    if ~isfield(cfg,'svd_rel_tol'), cfg.svd_rel_tol = 1e-10; end
    if ~isfield(cfg,'svd_energy_ratio'), cfg.svd_energy_ratio = 0.99999999; end
    if ~isfield(cfg,'svd_fixed_rank'), cfg.svd_fixed_rank = 1000; end

    A = gather(A);
    b = gather(b(:));
    nAcols = size(A,2);

    % For large tall systems, first perform a row-blocked QR compression
    % A = Q*R and then take the SVD of the small R.  This is the same
    % truncated least-squares solve, but it never forms the full tall U.
    use_tsqr = (8*numel(A) >= 512*1024^2) && (size(A,1) > size(A,2));
    if use_tsqr
        [R_small,b_small] = tsqr_compress_rhs_local(A,b);
        clear A b
        [U,S,V] = svd(R_small,'econ');
        rhs_for_u = b_small;
        clear R_small b_small
    else
        [U,S,V] = svd(A,'econ');
        rhs_for_u = b;
        clear A b
    end
    s = diag(S);
    clear S
    if isempty(s)
        x = zeros(nAcols,1);
        return;
    end

    switch lower(cfg.svd_mode)
        case 'fixed_rank'
            r_keep = min(cfg.svd_fixed_rank,length(s));
        case {'energy','energy_ratio'}
            e = cumsum(s.^2)/(sum(s.^2)+eps);
            r_keep = find(e >= cfg.svd_energy_ratio,1,'first');
            if isempty(r_keep), r_keep = length(s); end
        otherwise
            r_keep = sum(s >= cfg.svd_rel_tol*s(1));
    end

    r_keep = max(1,r_keep);
    s = s(1:r_keep);
    filt = s ./ (s.^2 + cfg.solve_lambda);

    % Form U'*b before truncation so MATLAB does not duplicate the very
    % large N-by-r U array merely to select its leading columns.
    ub = U.'*rhs_for_u;
    clear U rhs_for_u
    coeff = filt .* ub(1:r_keep);
    clear ub filt
    if r_keep == size(V,2)
        x = V * coeff;
    else
        x = V(:,1:r_keep) * coeff;
    end
end

function [R_acc,y_acc] = tsqr_compress_rhs_local(A,b)
%TSQR_COMPRESS_RHS_LOCAL stable row-blocked QR compression of [A,b].
    nrows = size(A,1);
    ncols = size(A,2);
    block_rows = max(ncols,min(5000,nrows));
    R_acc = [];
    y_acc = [];

    for row0 = 1:block_rows:nrows
        rows = row0:min(row0+block_rows-1,nrows);
        A_block = A(rows,:);
        b_block = b(rows);
        [Q_block,R_block] = qr(A_block,0);
        y_block = Q_block.'*b_block;
        clear A_block b_block Q_block

        if isempty(R_acc)
            R_acc = R_block;
            y_acc = y_block;
        else
            R_stack = [R_acc;R_block];
            y_stack = [y_acc;y_block];
            clear R_acc R_block y_acc y_block
            [Q_merge,R_acc] = qr(R_stack,0);
            y_acc = Q_merge.'*y_stack;
            clear R_stack y_stack Q_merge
        end
    end
end

function [A_solve,b_solve] = compress_tall_system_local(A,b)
%COMPRESS_TALL_SYSTEM_LOCAL QR-compress large tall least-squares systems.
    if (8*numel(A) >= 512*1024^2) && (size(A,1) > size(A,2))
        [A_solve,b_solve] = tsqr_compress_rhs_local(A,b(:));
    else
        A_solve = A;
        b_solve = b(:);
    end
end

%% ============================================================
% raw family and r training
%% ============================================================
function raw = make_raw_family_local(family,m,centers,cfg)
    raw = struct();
    raw.family = family;
    raw.m = m;
    raw.centers = centers;

    d = size(centers,1);
    switch lower(family)
        case {'tanh','ridge_gauss','gauss','sin'}
            raw.w_basis = 2*rand(d,m)-1;

        case {'radial_gauss','mexican_hat'}
            raw.w_basis = [];

        otherwise
            error('Unknown family = %s',family);
    end
end

function term = build_term_from_r_local(family,raw,r)
    r = r(:);
    centers = raw.centers;

    switch lower(family)
        case 'tanh'
            W = r.*raw.w_basis;
            term.type = 'tanh';
            term.w = W;
            term.b = -sum(centers.*W,1);

        case {'ridge_gauss','gauss'}
            W = r.*raw.w_basis;
            term.type = 'gauss';
            term.w = W;
            term.b = -sum(centers.*W,1);

        case 'sin'
            W = r.*raw.w_basis;
            term.type = 'sin';
            term.w = W;
            term.b = -sum(centers.*W,1);

        case 'radial_gauss'
            term.type = 'radial_gauss';
            term.A = centers;
            term.r = r;

        case 'mexican_hat'
            term.type = 'mexican_hat';
            term.A = centers;
            term.r = r;

        otherwise
            error('Unknown family = %s',family);
    end
end

function [r_opt,MSE_opt,term_opt] = train_r_value_local(X,target,raw,r0,opts,family,lambda)
    [r_opt,MSE_opt,~,term_opt] = ADRaNN_train_r_analytic3D( ...
        'value',X,target,raw,r0,opts,family,lambda,struct());
end

function [r_opt,MSE_opt,term_opt] = train_r_residual_local(X,target,u_ref,ux_ref,uy_ref,raw,r0,opts,problem,family,lambda)
% Analytic-gradient residual/Newton-linearized r-training for Burgers2D:
%   J_Burgers2D(u_ref) Phi(r) alpha ~= target.
    extra = struct();
    extra.problem = problem;
    extra.u_ref = u_ref;
    extra.ux_ref = ux_ref;
    extra.uy_ref = uy_ref;
    [r_opt,MSE_opt,~,term_opt] = ADRaNN_train_r_analytic3D( ...
        'burgers2d_residual',X,target,raw,r0,opts,family,lambda,extra);
end

%% ============================================================
% evaluation
%% ============================================================
function y = apply_basis_blocks_local(blocks,coef)
%APPLY_BASIS_BLOCKS_LOCAL Multiply blockwise cached response matrices.
    coef = coef(:);
    y = zeros(size(blocks{1},1),1);
    col0 = 0;
    for k = 1:numel(blocks)
        nk = size(blocks{k},2);
        y = y + blocks{k}*coef(col0+(1:nk));
        col0 = col0 + nk;
    end
end

function Phi = eval_terms_local(terms,X,ord,dir)
    Mtotal = 0;
    for k = 1:numel(terms)
        Mtotal = Mtotal + term_num_cols_local(terms{k});
    end

    Phi = zeros(size(X,1),Mtotal);
    col = 0;

    for k = 1:numel(terms)
        term = terms{k};
        m = term_num_cols_local(term);

        switch lower(term.type)
            case 'gauss'
                B = BasisFunction3D('ridge_gauss',term,X,ord,dir);
            otherwise
                B = BasisFunction3D(term.type,term,X,ord,dir);
        end

        Phi(:,col+(1:m)) = B;
        col = col + m;
    end
end

function val = eval_solution_local(terms,coef,X,ord,dir)
    Phi = eval_terms_local(terms,X,ord,dir);
    val = Phi*coef(:);
end

function m = term_num_cols_local(term)
    switch lower(term.type)
        case {'tanh','gauss','sin'}
            m = size(term.w,2);
        case {'radial_gauss','mexican_hat'}
            m = size(term.A,2);
        otherwise
            error('Unknown term.type = %s',term.type);
    end
end

function val = getfield_default_local(s,name,default_val)
    if isstruct(s) && isfield(s,name) && ~isempty(s.(name))
        val = s.(name);
    else
        val = default_val;
    end
end
