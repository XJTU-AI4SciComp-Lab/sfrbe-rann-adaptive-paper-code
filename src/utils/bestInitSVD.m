function [coef, info] = bestInitSVD(problem, basisEval, solEval, cons, X, eta, svd_cfg, GPU, QR)
%BESTINITSVD Warm start for an enriched/rebuilt trial space.
%   Computes two initializations for the new basis and returns the one with
%   the smaller weighted nonlinear residual (matching the legacy
%   KDV_Newton_module 'best_kdv_init_svd'):
%     1. value projection:  Phi_new c ~= u_old
%     2. Newton-linearized solve:  J(u_old) Phi_new c ~= J(u_old) u_old - F(u_old)
%        with the boundary/initial constraints.
%   Inputs:
%     problem  : ProblemFunction problem
%     basisEval: @(ord,dir,X) -> N x m values/derivatives of the NEW basis
%     solEval  : @(ord,dir,X) -> N x 1 current solution (ord-th derivative, dir)
%     cons     : ProblemFunction('constraints') struct (or [])
%     X        : interior collocation points (N x 2)
%     eta      : boundary/initial weight
%     svd_cfg  : cfg with .svd_mode, .svd_rel_tol, .svd_fixed_rank, .solve_lambda
%     GPU, QR  : solver flags for the value projection
%   Outputs:
%     coef : chosen coefficient vector (m x 1)
%     info : struct with .method, .rel_value, .rel_linearized

    if nargin < 9 || isempty(QR), QR = 1; end
    if nargin < 8 || isempty(GPU), GPU = 0; end
    if nargin < 7 || isempty(svd_cfg), svd_cfg = struct(); end

    X = gather(X);
    % Subsample deterministically for large grids (warm start only).
    nX = size(X, 1);
    if nX > 20000
        stepX = ceil(nX / 20000);
        X = X(1:stepX:end, :);
    end

    % ----- value projection -----
    Phi_val = gather(basisEval(0, 0, X));
    u_old0  = gather(solEval(0, 0, X));
    coef_val = YY_utils('solve_af', Phi_val, u_old0, GPU, QR);

    % ----- Newton-linearized solve -----
    sol = struct();
    sol.u    = gather(solEval(0, 0, X));
    sol.ux   = gather(solEval(1, 1, X));
    sol.ut   = gather(solEval(1, 2, X));
    sol.uxx  = gather(solEval(2, 1, X));
    sol.uxxx = gather(solEval(3, 1, X));
    Jc = ProblemFunction('jacobian', X, problem, sol);
    F  = ProblemFunction('residual', X, problem, sol);

    Phit   = gather(basisEval(1, 2, X));
    Phix   = gather(basisEval(1, 1, X));
    Phixxx = gather(basisEval(3, 1, X));
    Aint = Jc.ut .* Phit + Jc.ux .* Phix + Jc.uxxx .* Phixxx;
    if isfield(Jc, 'u')
        Aint = Aint + Jc.u .* Phi_val;
    end
    if isfield(Jc, 'uxx')
        Aint = Aint + Jc.uxx .* gather(basisEval(2, 1, X));
    end

    Ju_old = Jc.ut .* sol.ut + Jc.ux .* sol.ux + Jc.uxxx .* sol.uxxx;
    if isfield(Jc, 'u')
        Ju_old = Ju_old + Jc.u .* sol.u;
    end
    if isfield(Jc, 'uxx')
        Ju_old = Ju_old + Jc.uxx .* sol.uxx;
    end
    yint = Ju_old - F;

    A = Aint;
    y = yint;
    if ~isempty(cons)
        Xc = cons.X;
        types = cons.types;
        target = cons.target(:);
        idxV   = strcmpi(types, 'value');
        idxUx  = strcmpi(types, 'ux');
        idxUxx = strcmpi(types, 'uxx');
        if any(idxV)
            A = [A; eta * gather(basisEval(0, 0, Xc(idxV, :)))]; %#ok<AGROW>
            y = [y; eta * target(idxV)]; %#ok<AGROW>
        end
        if any(idxUx)
            A = [A; eta * gather(basisEval(1, 1, Xc(idxUx, :)))]; %#ok<AGROW>
            y = [y; eta * target(idxUx)]; %#ok<AGROW>
        end
        if any(idxUxx)
            A = [A; eta * gather(basisEval(2, 1, Xc(idxUxx, :)))]; %#ok<AGROW>
            y = [y; eta * target(idxUxx)]; %#ok<AGROW>
        end
        idxPu   = strcmpi(types, 'periodic_u');
        idxPux  = strcmpi(types, 'periodic_ux');
        idxPuxx = strcmpi(types, 'periodic_uxx');
        if any(idxPu)
            Bp = periodic_diff_pred(basisEval, Xc, idxPu, 0);
            A = [A; eta * gather(Bp)]; %#ok<AGROW>
            y = [y; zeros(size(Bp, 1), 1)]; %#ok<AGROW>
        end
        if any(idxPux)
            Bp = periodic_diff_pred(basisEval, Xc, idxPux, 1);
            A = [A; eta * gather(Bp)]; %#ok<AGROW>
            y = [y; zeros(size(Bp, 1), 1)]; %#ok<AGROW>
        end
        if any(idxPuxx)
            Bp = periodic_diff_pred(basisEval, Xc, idxPuxx, 2);
            A = [A; eta * gather(Bp)]; %#ok<AGROW>
            y = [y; zeros(size(Bp, 1), 1)]; %#ok<AGROW>
        end
    end

    coef_lin = linear_solve_svd_pred(A, y, svd_cfg);

    % ----- pick the one with the smaller weighted nonlinear residual -----
    rel_val = nonlinear_residual_rel_pred(problem, basisEval, coef_val, cons, X, eta);
    rel_lin = nonlinear_residual_rel_pred(problem, basisEval, coef_lin, cons, X, eta);

    info = struct();
    info.method = 'value_projection';
    info.rel_value = rel_val;
    info.rel_linearized = rel_lin;
    if rel_lin <= rel_val
        coef = coef_lin;
        info.method = 'linearized';
        info.rel_best = rel_lin;
    else
        coef = coef_val;
        info.rel_best = rel_val;
    end
end

function rel = nonlinear_residual_rel_pred(problem, basisEval, coef, cons, X, eta)
    u    = gather(basisEval(0, 0, X)) * coef;
    ux   = gather(basisEval(1, 1, X)) * coef;
    ut   = gather(basisEval(1, 2, X)) * coef;
    uxx  = gather(basisEval(2, 1, X)) * coef;
    uxxx = gather(basisEval(3, 1, X)) * coef;
    sol  = struct('u', u, 'ux', ux, 'ut', ut, 'uxx', uxx, 'uxxx', uxxx);
    F    = ProblemFunction('residual', X, problem, sol);
    R    = F;
    rhs  = zeros(size(F));
    if ~isempty(cons)
        Xc = cons.X;
        types = cons.types;
        target = cons.target(:);
        idxV   = strcmpi(types, 'value');
        idxUx  = strcmpi(types, 'ux');
        idxUxx = strcmpi(types, 'uxx');
        if any(idxV)
            rv = gather(basisEval(0, 0, Xc(idxV, :))) * coef - target(idxV);
            R = [R; eta*rv]; %#ok<AGROW>
            rhs = [rhs; eta*target(idxV)]; %#ok<AGROW>
        end
        if any(idxUx)
            ru = gather(basisEval(1, 1, Xc(idxUx, :))) * coef - target(idxUx);
            R = [R; eta*ru]; %#ok<AGROW>
            rhs = [rhs; eta*target(idxUx)]; %#ok<AGROW>
        end
        if any(idxUxx)
            ruu = gather(basisEval(2, 1, Xc(idxUxx, :))) * coef - target(idxUxx);
            R = [R; eta*ruu]; %#ok<AGROW>
            rhs = [rhs; eta*target(idxUxx)]; %#ok<AGROW>
        end
        idxPu   = strcmpi(types, 'periodic_u');
        idxPux  = strcmpi(types, 'periodic_ux');
        idxPuxx = strcmpi(types, 'periodic_uxx');
        if any(idxPu)
            Bp = gather(periodic_diff_pred(basisEval, Xc, idxPu, 0));
            R = [R; eta*(Bp*coef)]; %#ok<AGROW>
            rhs = [rhs; zeros(size(Bp,1),1)]; %#ok<AGROW>
        end
        if any(idxPux)
            Bp = gather(periodic_diff_pred(basisEval, Xc, idxPux, 1));
            R = [R; eta*(Bp*coef)]; %#ok<AGROW>
            rhs = [rhs; zeros(size(Bp,1),1)]; %#ok<AGROW>
        end
        if any(idxPuxx)
            Bp = gather(periodic_diff_pred(basisEval, Xc, idxPuxx, 2));
            R = [R; eta*(Bp*coef)]; %#ok<AGROW>
            rhs = [rhs; zeros(size(Bp,1),1)]; %#ok<AGROW>
        end
    end
    rel = norm(R) / (norm(rhs) + sqrt(length(R)) + eps);
end

function B = periodic_diff_pred(basisEval, X, idx, ord)
    seg = find(idx);
    nP = numel(seg) / 2;
    XL = X(seg(1:nP), :);
    XR = X(seg(nP+1:end), :);
    B = basisEval(ord, 1, XL) - basisEval(ord, 1, XR);
end

function x = linear_solve_svd_pred(A, b, cfg)
    [U, S, V] = svd(A, 'econ');
    s = diag(S);
    if isempty(s)
        x = zeros(size(A, 2), 1);
        return;
    end
    mode = getfield_default_pred(cfg, 'svd_mode', 'rel_tol');
    switch lower(mode)
        case {'none','full'}
            % No truncation (paper): keep every positive singular value.
            r_keep = sum(s > 0);
        case 'fixed_rank'
            r_keep = min(getfield_default_pred(cfg, 'svd_fixed_rank', 1000), length(s));
        case {'energy', 'energy_ratio'}
            e = cumsum(s.^2) / (sum(s.^2) + eps);
            r_keep = find(e >= getfield_default_pred(cfg, 'svd_energy_ratio', 0.99999999), 1, 'first');
            if isempty(r_keep), r_keep = length(s); end
        otherwise
            r_keep = sum(s >= getfield_default_pred(cfg, 'svd_rel_tol', 1e-10) * s(1));
    end
    r_keep = max(1, r_keep);
    U = U(:, 1:r_keep);
    V = V(:, 1:r_keep);
    s = s(1:r_keep);
    filt = s ./ (s.^2 + getfield_default_pred(cfg, 'solve_lambda', 0));
    x = V * (filt .* (U.' * b));
end

function v = getfield_default_pred(s, name, default_val)
    if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
        v = s.(name);
    else
        v = default_val;
    end
end
