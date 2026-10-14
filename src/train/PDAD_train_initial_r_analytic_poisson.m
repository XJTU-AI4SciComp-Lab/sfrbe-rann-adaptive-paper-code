function [r_opt, info] = PDAD_train_initial_r_analytic_poisson(X_int, rhs_int, X_bd, rhs_bd, w_basis, b_basis, r_init, op, opts)
%PDAD_TRAIN_INITIAL_R_ANALYTIC_POISSON
% Analytic-gradient PDAD training for the initial two-scale parameter r=[r1;r2]
% in 2D Poisson-type RaNN problems.
%
% It optimizes only r. The hidden directions w_basis and centers b_basis are
% fixed. After r is trained, the main code should rebuild
%     w = r .* w_basis;  b = -sum(b_basis .* w,1)
% and then solve the original least-squares system exactly as before.
%
% Supported initial feature families:
%   'tanh'        : phi=tanh(xi)
%   'gauss'       : phi=exp(-xi^2), same as ridge_gauss
%   'ridge_gauss' : phi=exp(-xi^2)
%
% PDE operator convention:
%   L phi = alpha0*phi + alpha1*phi_x + alpha2*phi_xx
%         + alpha3*phi_y + alpha4*phi_yy + alpha5*phi_xy
%
% Initial PDAD scale training for the SFR/RBE Poisson-Gauss pipeline.

    if nargin < 9 || isempty(opts)
        opts = struct();
    end

    opts = set_default_opts_local(opts);

    w_basis = orient_2_by_m_local(w_basis, 'w_basis');
    b_basis = orient_2_by_m_local(b_basis, 'b_basis');

    r = r_init(:);
    if numel(r) ~= 2
        error('r_init must be a 2-vector.');
    end

    rhs_int = rhs_int(:);
    rhs_bd  = rhs_bd(:);

    matrix_b = [rhs_int; opts.eta_bc_train * rhs_bd];

    m_adam = zeros(2,1);
    v_adam = zeros(2,1);

    best_loss = inf;
    best_r = r;
    best_alpha = [];

    hist.r = zeros(2, opts.maxit);
    hist.loss = nan(1, opts.maxit);
    hist.grad_raw_norm = nan(1, opts.maxit);
    hist.grad_used_norm = nan(1, opts.maxit);

    if opts.verbose
        fprintf('\n========== Initial PDAD r-training, analytic gradient ==========' );
        fprintf('\nfamily = %s\n', opts.family);
        fprintf('r_init = [%g %g]\n', r(1), r(2));
        fprintf('PDE train points = %d, BC train points = %d\n', size(X_int,1), size(X_bd,1));
        fprintf('eta_bc_train = %.6e, lambda_ridge = %.6e\n', opts.eta_bc_train, opts.lambda_ridge);
    end

    for it = 1:opts.maxit
        [loss, grad, alpha] = objective_and_grad_local( ...
            X_int, rhs_int, X_bd, rhs_bd, matrix_b, ...
            w_basis, b_basis, r, op, opts);

        grad = real(grad(:));
        grad(~isfinite(grad)) = 0;

        grad_raw_norm = norm(grad);
        grad_used = grad;
        if opts.normalize_grad && grad_raw_norm > opts.grad_eps
            grad_used = grad / grad_raw_norm;
        end

        grad_used_norm = norm(grad_used);

        if loss < best_loss && all(isfinite(r))
            best_loss = loss;
            best_r = r;
            best_alpha = alpha;
        end

        hist.r(:,it) = r;
        hist.loss(it) = loss;
        hist.grad_raw_norm(it) = grad_raw_norm;
        hist.grad_used_norm(it) = grad_used_norm;

        if opts.verbose && (it == 1 || mod(it, opts.print_every) == 0 || it == opts.maxit)
            fprintf('iter %4d/%4d: loss=%.6e, r=[ %.6g %.6g], |grad_raw|=%.3e, |grad_used|=%.3e\n', ...
                it, opts.maxit, loss, r(1), r(2), grad_raw_norm, grad_used_norm);
        end

        if grad_used_norm < opts.grad_tol
            if opts.verbose
                fprintf('PDAD stops: used gradient norm below %.3e.\n', opts.grad_tol);
            end
            break;
        end

        adam_opt.t = it;
        adam_opt.learning_rate = opts.learning_rate;
        adam_opt.beta1 = opts.beta1;
        adam_opt.beta2 = opts.beta2;
        adam_opt.epsilon = opts.eps_adam;
        adam_opt.lower_bound = opts.clip_min(:);
        adam_opt.upper_bound = opts.clip_max(:);

        [r, m_adam, v_adam] = adam_update_local(r, grad_used, m_adam, v_adam, adam_opt);
    end

    r_opt = best_r(:);

    info = struct();
    info.family = opts.family;
    info.r_init = r_init(:);
    info.r_opt = r_opt;
    info.best_loss = best_loss;
    info.best_alpha = best_alpha;
    info.opts = opts;
    info.hist = hist;

    if opts.verbose
        fprintf('PDAD finished: r_opt=[ %.6g %.6g], best_loss=%.6e\n', r_opt(1), r_opt(2), best_loss);
    end
end

%% ============================================================
%  Objective and analytic gradient
%% ============================================================
function [MSE, g, alpha] = objective_and_grad_local(X_int, rhs_int, X_bd, rhs_bd, matrix_b, w_basis, b_basis, r, op, opts)

    [A, dA_dr1, dA_dr2] = operator_and_grad_local(opts.family, X_int, r, w_basis, b_basis, op);
    [Phi_b, dPhi_dr1, dPhi_dr2] = basis_and_grad_local(opts.family, X_bd, r, w_basis, b_basis);

    M = [A; opts.eta_bc_train * Phi_b];
    dM_dr1 = [dA_dr1; opts.eta_bc_train * dPhi_dr1];
    dM_dr2 = [dA_dr2; opts.eta_bc_train * dPhi_dr2];

    ncol = size(M,2);
    if opts.lambda_ridge > 0
        alpha = (M.'*M + opts.lambda_ridge * eye(ncol)) \ (M.' * matrix_b);
    else
        alpha = M \ matrix_b;
    end

    res = M * alpha - matrix_b;
    Ntot = size(M,1);

    MSE = (res.' * res) / Ntot;

    v1 = dM_dr1 * alpha;
    v2 = dM_dr2 * alpha;

    g = (2 / Ntot) * [res.' * v1; res.' * v2];
    g = real(g);
end

%% ============================================================
%  Operator matrix and analytic derivative with respect to r
%% ============================================================
function [A, dA_dr1, dA_dr2] = operator_and_grad_local(family, X, r, w_basis, b_basis, op)

    [u, ux, uxx, uy, uyy, uxy, ...
     du1, dux1, duxx1, duy1, duyy1, duxy1, ...
     du2, dux2, duxx2, duy2, duyy2, duxy2] = basis_derivatives_and_rgrad_local(family, X, r, w_basis, b_basis);

    [N, M] = size(u);

    alpha0 = get_op_field_local(op, 'alpha0', 0);
    alpha1 = get_op_field_local(op, 'alpha1', 0);
    alpha2 = get_op_field_local(op, 'alpha2', 0);
    alpha3 = get_op_field_local(op, 'alpha3', 0);
    alpha4 = get_op_field_local(op, 'alpha4', 0);
    alpha5 = get_op_field_local(op, 'alpha5', 0);

    C0 = toNM_local(alpha0, X, N, M);
    C1 = toNM_local(alpha1, X, N, M);
    C2 = toNM_local(alpha2, X, N, M);
    C3 = toNM_local(alpha3, X, N, M);
    C4 = toNM_local(alpha4, X, N, M);
    C5 = toNM_local(alpha5, X, N, M);

    A = C0.*u + C1.*ux + C2.*uxx + C3.*uy + C4.*uyy + C5.*uxy;

    dA_dr1 = C0.*du1 + C1.*dux1 + C2.*duxx1 + C3.*duy1 + C4.*duyy1 + C5.*duxy1;
    dA_dr2 = C0.*du2 + C1.*dux2 + C2.*duxx2 + C3.*duy2 + C4.*duyy2 + C5.*duxy2;
end

%% ============================================================
%  Boundary basis and analytic derivative with respect to r
%% ============================================================
function [Phi, dPhi_dr1, dPhi_dr2] = basis_and_grad_local(family, X, r, w_basis, b_basis)

    [xi, dz1, dz2] = xi_and_rgrad_local(X, r, w_basis, b_basis);

    switch lower(family)
        case 'tanh'
            Phi = tanh(xi);
            s1 = 1 - Phi.^2;
            dPhi_dr1 = s1 .* dz1;
            dPhi_dr2 = s1 .* dz2;

        case {'gauss','gaussian','ridge_gauss'}
            Phi = exp(-(xi.^2));
            dPhi_dr1 = (-2 .* xi .* Phi) .* dz1;
            dPhi_dr2 = (-2 .* xi .* Phi) .* dz2;

        otherwise
            error('Unsupported initial PDAD family: %s. Use tanh or gauss/ridge_gauss.', family);
    end
end

%% ============================================================
%  Basis derivatives and r-gradients
%% ============================================================
function [u, ux, uxx, uy, uyy, uxy, ...
          du1, dux1, duxx1, duy1, duyy1, duxy1, ...
          du2, dux2, duxx2, duy2, duyy2, duxy2] = basis_derivatives_and_rgrad_local(family, X, r, w_basis, b_basis)

    [xi, dz1, dz2, W1, W2, WB1, WB2] = xi_and_rgrad_local(X, r, w_basis, b_basis);

    N = size(X,1);
    W1r = repmat(W1, N, 1);
    W2r = repmat(W2, N, 1);
    WB1r = repmat(WB1, N, 1);
    WB2r = repmat(WB2, N, 1);

    switch lower(family)
        case 'tanh'
            u = tanh(xi);
            s1 = 1 - u.^2;                  % phi'
            s2 = -2 .* u .* s1;             % phi''
            s3 = -2 .* s1.^2 + 4 .* (u.^2) .* s1;  % phi'''

            ux  = s1 .* W1r;
            uy  = s1 .* W2r;
            uxx = s2 .* (W1r.^2);
            uyy = s2 .* (W2r.^2);
            uxy = s2 .* (W1r .* W2r);

            du1 = s1 .* dz1;
            du2 = s1 .* dz2;

            dux1 = s2 .* dz1 .* W1r + s1 .* WB1r;
            dux2 = s2 .* dz2 .* W1r;

            duy1 = s2 .* dz1 .* W2r;
            duy2 = s2 .* dz2 .* W2r + s1 .* WB2r;

            duxx1 = s3 .* dz1 .* (W1r.^2) + s2 .* (2 .* W1r .* WB1r);
            duxx2 = s3 .* dz2 .* (W1r.^2);

            duyy1 = s3 .* dz1 .* (W2r.^2);
            duyy2 = s3 .* dz2 .* (W2r.^2) + s2 .* (2 .* W2r .* WB2r);

            duxy1 = s3 .* dz1 .* (W1r .* W2r) + s2 .* (WB1r .* W2r);
            duxy2 = s3 .* dz2 .* (W1r .* W2r) + s2 .* (W1r .* WB2r);

        case {'gauss','gaussian','ridge_gauss'}
            E = exp(-(xi.^2));
            u = E;
            T = 1 - 2 .* xi.^2;
            U = 12 .* xi - 8 .* xi.^3;

            ux  = -2 .* W1r .* xi .* E;
            uy  = -2 .* W2r .* xi .* E;
            uxx = (4 .* xi.^2 - 2) .* (W1r.^2) .* E;
            uyy = (4 .* xi.^2 - 2) .* (W2r.^2) .* E;
            uxy = (4 .* xi.^2 - 2) .* (W1r .* W2r) .* E;

            du1 = (-2 .* xi .* E) .* dz1;
            du2 = (-2 .* xi .* E) .* dz2;

            dux1 = -2 .* (WB1r .* xi .* E + W1r .* (E .* dz1 .* T));
            dux2 = -2 .* (W1r .* (E .* dz2 .* T));

            duy1 = -2 .* (W2r .* (E .* dz1 .* T));
            duy2 = -2 .* (WB2r .* xi .* E + W2r .* (E .* dz2 .* T));

            duxx1 = E .* (U .* ((W1r.^2) .* dz1) + (4 .* xi.^2 - 2) .* (2 .* W1r .* WB1r));
            duxx2 = E .* (U .* ((W1r.^2) .* dz2));

            duyy1 = E .* (U .* ((W2r.^2) .* dz1));
            duyy2 = E .* (U .* ((W2r.^2) .* dz2) + (4 .* xi.^2 - 2) .* (2 .* W2r .* WB2r));

            duxy1 = E .* (U .* ((W1r .* W2r) .* dz1) + (4 .* xi.^2 - 2) .* (WB1r .* W2r));
            duxy2 = E .* (U .* ((W1r .* W2r) .* dz2) + (4 .* xi.^2 - 2) .* (W1r .* WB2r));

        otherwise
            error('Unsupported initial PDAD family: %s. Use tanh or gauss/ridge_gauss.', family);
    end
end

%% ============================================================
%  xi and derivative with respect to r
%% ============================================================
function [xi, dz1, dz2, W1, W2, WB1, WB2] = xi_and_rgrad_local(X, r, w_basis, b_basis)

    WB1 = w_basis(1,:);
    WB2 = w_basis(2,:);
    BB1 = b_basis(1,:);
    BB2 = b_basis(2,:);

    W1 = r(1) .* WB1;
    W2 = r(2) .* WB2;

    x1 = X(:,1);
    x2 = X(:,2);

    xi = bsxfun(@times, x1 - BB1, W1) + bsxfun(@times, x2 - BB2, W2);

    dz1 = bsxfun(@times, x1 - BB1, WB1);
    dz2 = bsxfun(@times, x2 - BB2, WB2);
end

%% ============================================================
%  Utilities
%% ============================================================
function opts = set_default_opts_local(opts)
    opts.family = get_opt_local(opts, 'family', 'tanh');
    opts.eta_bc_train = get_opt_local(opts, 'eta_bc_train', 5.0);
    opts.lambda_ridge = get_opt_local(opts, 'lambda_ridge', 1e-6);
    opts.maxit = get_opt_local(opts, 'maxit', 40);
    opts.learning_rate = get_opt_local(opts, 'learning_rate', 0.5);
    opts.beta1 = get_opt_local(opts, 'beta1', 0.9);
    opts.beta2 = get_opt_local(opts, 'beta2', 0.999);
    opts.eps_adam = get_opt_local(opts, 'eps_adam', 1e-8);
    opts.clip_min = get_opt_local(opts, 'clip_min', [1;1]);
    opts.clip_max = get_opt_local(opts, 'clip_max', [500;500]);
    opts.normalize_grad = get_opt_local(opts, 'normalize_grad', true);
    opts.grad_eps = get_opt_local(opts, 'grad_eps', 1e-14);
    opts.grad_tol = get_opt_local(opts, 'grad_tol', 1e-12);
    opts.verbose = get_opt_local(opts, 'verbose', true);
    opts.print_every = get_opt_local(opts, 'print_every', 5);
end

function val = get_opt_local(opts, name, default_val)
    if isstruct(opts) && isfield(opts, name) && ~isempty(opts.(name))
        val = opts.(name);
    else
        val = default_val;
    end
end

function X = orient_2_by_m_local(X, name)
    if size(X,1) == 2
        return;
    elseif size(X,2) == 2
        X = X.';
    else
        error('%s must be 2-by-M or M-by-2.', name);
    end
end

function a = get_op_field_local(op, name, default_val)
    if isstruct(op) && isfield(op, name) && ~isempty(op.(name))
        a = op.(name);
    else
        a = default_val;
    end
end

function C = toNM_local(a, X, N, M)
    if isa(a,'function_handle')
        a = a(X);
    end
    if isscalar(a)
        C = a .* ones(N, M);
    else
        a = a(:);
        if numel(a) ~= N
            error('Operator coefficient length must be N or scalar.');
        end
        C = a * ones(1, M);
    end
end

function [params, m, v] = adam_update_local(params, grads, m, v, opt)
    t = opt.t;
    lr = opt.learning_rate;
    beta1 = opt.beta1;
    beta2 = opt.beta2;
    eps0 = opt.epsilon;

    m = beta1 * m + (1 - beta1) * grads;
    v = beta2 * v + (1 - beta2) * (grads.^2);

    m_hat = m / (1 - beta1^t);
    v_hat = v / (1 - beta2^t);

    params = params - lr * m_hat ./ (sqrt(v_hat) + eps0);

    params = max(params, opt.lower_bound(:));
    params = min(params, opt.upper_bound(:));
end
