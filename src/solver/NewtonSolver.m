function [coef, info] = NewtonSolver(problem, basisEval, X_int, cons, coef0, cfg)
%NEWTONSOLVER Damped Gauss-Newton for nonlinear PDE least-squares problems.
%
%   problem  : ProblemFunction problem with actions 'residual' and 'jacobian'
%   basisEval: @(ord,dir,X) -> N x m matrix of basis (or RBE composite) values
%              (ord=0 value, 1 first, 2 second, 3 third; dir=1 x, dir=2 t)
%   X_int    : interior collocation points (N_int x 2)
%   cons     : struct with X, types, target (from ProblemFunction('constraints'))
%   coef0    : initial coefficients
%   cfg      : num_newton, tau_newton, line_search, max_line_search,
%              tol_newton_improve, stall_patience, min_newton_iter,
%              linear_solver, svd_mode, svd_rel_tol, svd_fixed_rank,
%              solve_lambda, eta_bc, verbose
%
%   Solve the weighted system
%       [J; eta_bc*Bv; eta_bc*Bu; eta_bc*Buu] d = -[F; eta_bc*rv; ...]
%   by a truncated-SVD least-squares solve (same scheme as the legacy
%   KDV_Newton_module 'relax_full' solver), with a backtracking line search
%   that accepts non-increasing residuals and a stagnation-based early stop.
%   'maxit' is accepted as a backward-compatible alias for 'num_newton'.

    if nargin < 6 || isempty(cfg), cfg = struct(); end
    cfg = fill_cfg_local(cfg);

    coef = coef0(:);
    Xc = cons.X;
    types = cons.types;
    target = cons.target(:);
    idxV  = strcmpi(types, 'value');
    idxUx = strcmpi(types, 'ux');
    idxUxx = strcmpi(types, 'uxx');

    w = cfg.eta_bc;

    % Boundary/initial basis matrices are independent of coef: build once.
    Bv  = basisEval(0, 0, Xc);
    Bu  = basisEval(1, 1, Xc);
    Buu = basisEval(2, 1, Xc);

    % Periodic constraints (left - right = 0) if present.
    Bper_u = []; Bper_ux = []; Bper_uxx = [];
    idxPu = strcmpi(types, 'periodic_u');
    idxPux = strcmpi(types, 'periodic_ux');
    idxPuxx = strcmpi(types, 'periodic_uxx');
    if any(idxPu),   Bper_u   = periodic_diff_local(basisEval, Xc, idxPu, 0); end
    if any(idxPux),  Bper_ux  = periodic_diff_local(basisEval, Xc, idxPux, 1); end
    if any(idxPuxx), Bper_uxx = periodic_diff_local(basisEval, Xc, idxPuxx, 2); end

    [A, R, F, scale] = assemble_weighted_local(problem, basisEval, X_int, coef, ...
        Bv, Bu, Buu, target, idxV, idxUx, idxUxx, Bper_u, Bper_ux, Bper_uxx, w);
    rel = rel_res_local(R, scale);
    hist_rel = zeros(cfg.num_newton + 1, 1);
    hist_rel(1) = rel;

    stall_count = 0;
    it = 0;
    for it = 1:cfg.num_newton
        if rel < cfg.tau_newton
            if cfg.verbose
                fprintf('  [Newton] stop: rel_res < tau_newton = %.2e\n', cfg.tau_newton);
            end
            break;
        end

        rel_old = rel;
        d = -linear_solve_local(A, R, cfg);

        alpha = 1.0;
        accept = false;
        if cfg.line_search
            for ls = 1:cfg.max_line_search
                coefT = coef + alpha*d;
                [~, RT, ~, scaleT] = assemble_weighted_local(problem, basisEval, X_int, coefT, ...
                    Bv, Bu, Buu, target, idxV, idxUx, idxUxx, Bper_u, Bper_ux, Bper_uxx, w);
                relT = rel_res_local(RT, scaleT);
                if relT <= rel_old
                    coef = coefT;
                    rel = relT;
                    accept = true;
                    break;
                end
                alpha = 0.5*alpha;
            end
        else
            coef = coef + d;
            accept = true;
        end

        if ~accept
            if cfg.verbose
                fprintf('  [Newton] line search failed; stopping at it=%d\n', it);
            end
            break;
        end

        % Reassemble at the accepted point for the next iteration.
        [A, R, F, scale] = assemble_weighted_local(problem, basisEval, X_int, coef, ...
            Bv, Bu, Buu, target, idxV, idxUx, idxUxx, Bper_u, Bper_ux, Bper_uxx, w);
        rel = rel_res_local(R, scale);
        hist_rel(it+1) = rel;

        rel_improve = (rel_old - rel) / (rel_old + eps);
        if cfg.verbose
            fprintf('  [Newton] it %2d: alpha=%.3f rel_res=%.4e improve=%.4e\n', ...
                it, alpha, rel, rel_improve);
        end

        if it >= cfg.min_newton_iter
            if rel_improve < cfg.tol_newton_improve
                stall_count = stall_count + 1;
            else
                stall_count = 0;
            end
            if stall_count >= cfg.stall_patience
                if cfg.verbose
                    fprintf('  [Newton] stop: residual stagnated for %d steps (improve=%.4e < %.4e)\n', ...
                        cfg.stall_patience, rel_improve, cfg.tol_newton_improve);
                end
                break;
            end
        end
    end

    info.rel_res = rel;
    info.iter = it;
    info.hist_rel = hist_rel(hist_rel > 0);
    info.res_norm = norm(R);
    info.pde_res_norm = norm(F);
    info.bc_res = sqrt(norm(Bv*coef - target)^2 + norm(Bu*coef - target)^2 + ...
        norm(Buu*coef - target)^2);
    info.cost = norm(R)^2;
end

%% ============================================================
function [A, R, F, scale] = assemble_weighted_local(problem, basisEval, X_int, coef, ...
        Bv, Bu, Buu, target, idxV, idxUx, idxUxx, Bper_u, Bper_ux, Bper_uxx, w)
    Phi    = basisEval(0, 0, X_int);
    Phix   = basisEval(1, 1, X_int);
    Phit   = basisEval(1, 2, X_int);
    Phixx  = basisEval(2, 1, X_int);
    Phixxx = basisEval(3, 1, X_int);

    u    = Phi    * coef;
    ux   = Phix   * coef;
    ut   = Phit   * coef;
    uxx  = Phixx  * coef;
    uxxx = Phixxx * coef;
    sol = struct('u', u, 'ux', ux, 'ut', ut, 'uxx', uxx, 'uxxx', uxxx);

    F  = ProblemFunction('residual', X_int, problem, sol);
    Jc = ProblemFunction('jacobian', X_int, problem, sol);

    J = Jc.ut .* Phit + Jc.ux .* Phix + Jc.uxxx .* Phixxx;
    if isfield(Jc, 'u')
        J = J + Jc.u .* Phi;
    end
    if isfield(Jc, 'uxx')
        J = J + Jc.uxx .* Phixx;
    end

    rv  = Bv*coef  - target;
    ru  = Bu*coef  - target;
    ruu = Buu*coef - target;

    A = [J; w*Bv(idxV, :); w*Bu(idxUx, :); w*Buu(idxUxx, :)];
    R = [F; w*rv(idxV); w*ru(idxUx); w*ruu(idxUxx)];
    scale = [zeros(size(F, 1), 1); w*target(idxV); w*target(idxUx); w*target(idxUxx)];

    if ~isempty(Bper_u)
        A = [A; w*Bper_u];
        R = [R; w*(Bper_u*coef)];
        scale = [scale; zeros(size(Bper_u, 1), 1)];
    end
    if ~isempty(Bper_ux)
        A = [A; w*Bper_ux];
        R = [R; w*(Bper_ux*coef)];
        scale = [scale; zeros(size(Bper_ux, 1), 1)];
    end
    if ~isempty(Bper_uxx)
        A = [A; w*Bper_uxx];
        R = [R; w*(Bper_uxx*coef)];
        scale = [scale; zeros(size(Bper_uxx, 1), 1)];
    end
end

%% ============================================================
function rel = rel_res_local(R, scale)
    rel = norm(R) / (norm(scale) + sqrt(length(R)) + eps);
end

%% ============================================================
function x = linear_solve_local(A, b, cfg)
    A = gather(A);
    b = gather(b(:));

    switch lower(cfg.linear_solver)
        case 'svd'
            [U, S, V] = svd(A, 'econ');
            s = diag(S);
            if isempty(s)
                x = zeros(size(A, 2), 1);
                return;
            end

            switch lower(cfg.svd_mode)
                case {'none','full'}
                    % No truncation (paper): keep every positive singular
                    % value; only exact-zero directions are dropped.
                    r_keep = sum(s > 0);
                case 'fixed_rank'
                    r_keep = min(cfg.svd_fixed_rank, length(s));
                case {'energy', 'energy_ratio'}
                    e = cumsum(s.^2) / (sum(s.^2) + eps);
                    r_keep = find(e >= cfg.svd_energy_ratio, 1, 'first');
                    if isempty(r_keep), r_keep = length(s); end
                otherwise
                    r_keep = sum(s >= cfg.svd_rel_tol*s(1));
            end
            r_keep = max(1, r_keep);
            U = U(:, 1:r_keep);
            V = V(:, 1:r_keep);
            s = s(1:r_keep);

            filt = s ./ (s.^2 + cfg.solve_lambda);
            x = V * (filt .* (U.'*b));

        otherwise % 'normal' fallback (normal equations via SVD)
            [U, S, V] = svd(A, 'econ');
            s = diag(S);
            if isempty(s)
                x = zeros(size(A, 2), 1);
                return;
            end
            switch lower(cfg.svd_mode)
                case {'none','full'}
                    r_keep = sum(s > 0);
                otherwise
                    r_keep = sum(s >= cfg.svd_rel_tol*s(1));
            end
            r_keep = max(1, r_keep);
            U = U(:, 1:r_keep);
            V = V(:, 1:r_keep);
            s = s(1:r_keep);
            filt = s ./ (s.^2 + cfg.solve_lambda);
            x = V * (filt .* (U.'*b));
    end
end

%% ============================================================
function cfg = fill_cfg_local(cfg)
    if isfield(cfg, 'num_newton')
        cfg.num_newton = cfg.num_newton;
    elseif isfield(cfg, 'maxit')
        cfg.num_newton = cfg.maxit;   % backward-compatible alias
    else
        cfg.num_newton = 5;
    end
    if ~isfield(cfg, 'tau_newton'), cfg.tau_newton = 1e-5; end
    if ~isfield(cfg, 'line_search'), cfg.line_search = true; end
    if ~isfield(cfg, 'max_line_search'), cfg.max_line_search = 8; end
    if ~isfield(cfg, 'eta_bc'), cfg.eta_bc = 100; end
    if ~isfield(cfg, 'tol_newton_improve'), cfg.tol_newton_improve = 1e-4; end
    if ~isfield(cfg, 'stall_patience'), cfg.stall_patience = 2; end
    if ~isfield(cfg, 'min_newton_iter'), cfg.min_newton_iter = 2; end
    if ~isfield(cfg, 'linear_solver'), cfg.linear_solver = 'svd'; end
    if ~isfield(cfg, 'svd_mode'), cfg.svd_mode = 'rel_tol'; end
    if ~isfield(cfg, 'svd_rel_tol'), cfg.svd_rel_tol = 1e-10; end
    if ~isfield(cfg, 'svd_energy_ratio'), cfg.svd_energy_ratio = 0.99999999; end
    if ~isfield(cfg, 'svd_fixed_rank'), cfg.svd_fixed_rank = 1000; end
    if ~isfield(cfg, 'solve_lambda'), cfg.solve_lambda = 0; end
    if ~isfield(cfg, 'verbose'), cfg.verbose = false; end
end

%% ============================================================
function B = periodic_diff_local(basisEval, Xc, idx, ord)
%PERIODIC_DIFF_LOCAL left-right difference basis for a periodic constraint.
    seg = find(idx);
    nP = numel(seg) / 2;
    XL = Xc(seg(1:nP), :);
    XR = Xc(seg(nP+1:end), :);
    B = basisEval(ord, 1, XL) - basisEval(ord, 1, XR);
end
