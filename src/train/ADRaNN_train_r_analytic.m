function varargout = ADRaNN_train_r_analytic(mode,X,target,raw,r0,opts,varargin)
%ADRaNN_TRAIN_R_ANALYTIC
% General r-training for the generic 2D ProblemFunction project.
%
% Main interface used by the RBE family-scoring module:
%   [r_opt,MSE_opt,alpha_opt,term_opt,hist] = ADRaNN_train_r_analytic( ...
%       'operator',X,target,raw,r0,opts,family,lambda,extra)
%
% This implementation only uses the generic BasisFunction/BasisOperator interfaces.
% The objective is
%       min_r min_a || A_family(r) a - target ||^2 + lambda ||a||^2,
% where A_family(r) is assembled by BasisOperator(family,term,X,op).
%
% For robustness across all supported families, the r-gradient is computed
% by central finite differences of the same operator objective. This keeps
% the interface uniform for tanh/ridge_gauss/sin/radial_gauss/mexican_hat
% and avoids hidden problem-specific code paths.

    if nargin < 6 || isempty(opts), opts = struct(); end

    mode = lower(strtrim(mode));

    switch mode
        case 'operator'
            if numel(varargin) < 1 || isempty(varargin{1})
                family = raw.family;
            else
                family = varargin{1};
            end
            if numel(varargin) < 2 || isempty(varargin{2})
                lambda = 1e-6;
            else
                lambda = varargin{2};
            end
            if numel(varargin) < 3 || isempty(varargin{3})
                extra = struct();
            else
                extra = varargin{3};
            end
            if ~isfield(extra,'op') || isempty(extra.op)
                error('ADRaNN_train_r_analytic(''operator'',...) requires extra.op.');
            end
            op = extra.op;

            [r_opt,MSE_opt,alpha_opt,term_opt,hist] = train_operator_local( ...
                X,target,raw,r0,opts,family,lambda,op);

            varargout = cell(1,nargout);
            out = {r_opt,MSE_opt,alpha_opt,term_opt,hist};
            for k = 1:nargout
                varargout{k} = out{k};
            end

        case {'frozen_jacobian','kdv_residual','newton_residual'}
            % Jacobian-based residual r-training (matches the legacy
            % ADRaNN 'kdv_residual' mode): M(r) = J(u) Phi(r), with the
            % ANALYTIC dM/dr for ridge families and a relative-gradient Adam
            % update (so r keeps moving even when the residual is tiny).
            if numel(varargin) < 1 || isempty(varargin{1})
                family = raw.family;
            else
                family = varargin{1};
            end
            if numel(varargin) < 2 || isempty(varargin{2})
                lambda = 1e-6;
            else
                lambda = varargin{2};
            end
            if numel(varargin) < 3 || isempty(varargin{3})
                extra = struct();
            else
                extra = varargin{3};
            end
            if ~isfield(extra,'problem') || ~isfield(extra,'sol')
                error('ADRaNN_train_r_analytic(''frozen_jacobian'',...) requires extra.problem and extra.sol.');
            end
            [r_opt,MSE_opt,alpha_opt,term_opt,hist] = train_frozen_jacobian_local( ...
                X,target,raw,r0,opts,family,lambda,extra);

            varargout = cell(1,nargout);
            out = {r_opt,MSE_opt,alpha_opt,term_opt,hist};
            for k = 1:nargout
                varargout{k} = out{k};
            end

        otherwise
            % Backward-compatible compact interface:
            %   [r_opt,MSE,info] = ADRaNN_train_r_analytic(mode,X,target,raw,r0,opts,lambda)
            % Only generic value/operator-free Laplace-style modes are supported.
            if numel(varargin) < 1 || isempty(varargin{1})
                lambda = 1e-6;
            else
                lambda = varargin{1};
            end
            [r_opt,MSE_opt,alpha_opt,term_opt,hist] = train_simple_local( ...
                mode,X,target,raw,r0,opts,lambda);
            info = hist;
            info.alpha = alpha_opt;
            info.term = term_opt;

            varargout = cell(1,nargout);
            out = {r_opt,MSE_opt,info,term_opt,hist};
            for k = 1:nargout
                varargout{k} = out{k};
            end
    end
end

function [r_opt,MSE_opt,alpha_opt,term_opt,hist] = train_operator_local( ...
    X,target,raw,r0,opts,family,lambda,op)

    X = gather(X);
    target = gather(target(:));
    r = gather(r0(:));
    opts = fill_opts_local(opts,r);

    beta1 = getfield_default_local(opts,'beta1',0.9);
    beta2 = getfield_default_local(opts,'beta2',0.999);
    eps_adam = getfield_default_local(opts,'eps_adam',1e-8);

    clip_min = expand_clip_local(opts.clip_min,numel(r),-inf);
    clip_max = expand_clip_local(opts.clip_max,numel(r), inf);

    m_adam = zeros(size(r));
    v_adam = zeros(size(r));

    MSE_opt = inf;
    r_opt = r;
    alpha_opt = [];
    best_iter = 0;

    hist.MSE = nan(max(opts.maxit,1),1);
    hist.r = nan(numel(r),max(opts.maxit,1));
    hist.grad = nan(numel(r),max(opts.maxit,1));
    hist.grad_norm = nan(max(opts.maxit,1),1);

    for it = 1:opts.maxit
        [MSE,alpha] = objective_operator_local(X,target,raw,r,op,family,lambda);
        g = gradient_fd_local(@(rr)objective_operator_local(X,target,raw,rr,op,family,lambda), ...
            r,clip_min,clip_max,opts.fd_step_rel);

        hist.MSE(it) = MSE;
        hist.r(:,it) = r;
        hist.grad(:,it) = g;
        hist.grad_norm(it) = norm(g);

        if MSE < MSE_opt
            MSE_opt = MSE;
            r_opt = r;
            alpha_opt = alpha;
            best_iter = it;
        end

        if opts.verbose
            fprintf('  r-train operator %-14s it %3d: MSE=%.6e, r=[%s], grad=[%s]\n', ...
                family,it,MSE,vec_string_local(r),vec_string_local(g));
        end

        if it >= opts.minit && norm(g) < opts.grad_tol
            break;
        end

        g_used = g;
        if opts.use_rel_grad
            g_used = g_used ./ (abs(r)+1e-12);
        end

        m_adam = beta1*m_adam + (1-beta1)*g_used;
        v_adam = beta2*v_adam + (1-beta2)*(g_used.^2);
        mh = m_adam/(1-beta1^it);
        vh = v_adam/(1-beta2^it);

        r = r - opts.eta*mh./(sqrt(vh)+eps_adam);
        r = min(max(r,clip_min),clip_max);
    end

    [MSE_final,alpha_final] = objective_operator_local(X,target,raw,r,op,family,lambda);
    if MSE_final < MSE_opt
        MSE_opt = MSE_final;
        r_opt = r;
        alpha_opt = alpha_final;
        best_iter = opts.maxit + 1;   % ?????Adam ??????
    end

    term_opt = build_term_from_r_local(family,raw,r_opt);

    valid = isfinite(hist.MSE);
    hist.MSE = hist.MSE(valid);
    hist.r = hist.r(:,valid);
    hist.grad = hist.grad(:,valid);
    hist.grad_norm = hist.grad_norm(valid);
    hist.best_iter = best_iter;
    hist.best_MSE = MSE_opt;
    hist.best_r = r_opt(:);
end

function [r_opt,MSE_opt,alpha_opt,term_opt,hist] = train_simple_local( ...
    mode,X,target,raw,r0,opts,lambda)

    family = raw.family;
    op = struct('alpha0',0,'alpha1',0,'alpha2',-1,'alpha3',0,'alpha4',-1,'alpha5',0);

    switch lower(mode)
        case 'value'
            op = struct('alpha0',1,'alpha1',0,'alpha2',0,'alpha3',0,'alpha4',0,'alpha5',0);
        case {'pde','laplace','operator'}
            % Use default Poisson operator above.
        otherwise
            error('Unknown r-training mode = %s',mode);
    end

    [r_opt,MSE_opt,alpha_opt,term_opt,hist] = train_operator_local( ...
        X,target,raw,r0,opts,family,lambda,op);
end

function opts = fill_opts_local(opts,r)
    if ~isfield(opts,'maxit') || isempty(opts.maxit), opts.maxit = 30; end
    if ~isfield(opts,'eta') || isempty(opts.eta), opts.eta = 0.25; end
    if ~isfield(opts,'eps_adam') || isempty(opts.eps_adam), opts.eps_adam = 1e-8; end
    if ~isfield(opts,'clip_min') || isempty(opts.clip_min), opts.clip_min = ones(numel(r),1); end
    if ~isfield(opts,'clip_max') || isempty(opts.clip_max), opts.clip_max = 150*ones(numel(r),1); end
    if ~isfield(opts,'minit') || isempty(opts.minit), opts.minit = 5; end
    if ~isfield(opts,'grad_tol') || isempty(opts.grad_tol), opts.grad_tol = 1e-12; end
    if ~isfield(opts,'fd_step_rel') || isempty(opts.fd_step_rel), opts.fd_step_rel = 1e-4; end
    if ~isfield(opts,'verbose') || isempty(opts.verbose), opts.verbose = false; end
    if ~isfield(opts,'use_rel_grad') || isempty(opts.use_rel_grad), opts.use_rel_grad = false; end
    if ~isfield(opts,'mse_tol') || isempty(opts.mse_tol), opts.mse_tol = -inf; end
    if ~isfield(opts,'rel_mse_tol') || isempty(opts.rel_mse_tol), opts.rel_mse_tol = 0; end
    if ~isfield(opts,'stall_patience') || isempty(opts.stall_patience), opts.stall_patience = inf; end
    if ~isfield(opts,'print_every') || isempty(opts.print_every), opts.print_every = 10; end
end

function [MSE,alpha] = objective_operator_local(X,target,raw,r,op,family,lambda)
    term = build_term_from_r_local(family,raw,r);
    A = BasisOperator(family,term,X,op);
    A = gather(A);
    K = size(A,2);
    G = A.'*A + lambda*eye(K);
    rhs = A.'*target(:);
    [L,p] = chol(G,'lower');
    if p == 0
        alpha = L'\(L\rhs);
    else
        alpha = G\rhs;
    end
    res = A*alpha - target(:);
    MSE = mean(res.^2);
end

function g = gradient_fd_local(fun,r,clip_min,clip_max,fd_step_rel)
    g = zeros(size(r));
    for kk = 1:numel(r)
        h = fd_step_rel * max(1,abs(r(kk)));
        rp = r;
        rm = r;
        rp(kk) = min(rp(kk)+h,clip_max(kk));
        rm(kk) = max(rm(kk)-h,clip_min(kk));
        if abs(rp(kk)-rm(kk)) < eps
            g(kk) = 0;
        else
            fp = fun(rp);
            fm = fun(rm);
            g(kk) = (fp-fm)/(rp(kk)-rm(kk));
        end
    end
end

function term = build_term_from_r_local(family,raw,r)
    r = r(:);
    centers = raw.centers;

    switch lower(family)
        case 'tanh'
            W = r .* raw.w_basis;
            term.type = 'tanh';
            term.w = W;
            term.b = -sum(centers .* W,1);

        case {'ridge_gauss','gauss'}
            W = r .* raw.w_basis;
            term.type = 'gauss';
            term.w = W;
            term.b = -sum(centers .* W,1);

        case 'sin'
            W = r .* raw.w_basis;
            term.type = 'sin';
            term.w = W;
            term.b = -sum(centers .* W,1);

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

function clip = expand_clip_local(x,n,default_val)
    if isempty(x)
        clip = default_val*ones(n,1);
        return;
    end
    x = x(:);
    if numel(x) == 1
        clip = x*ones(n,1);
    elseif numel(x) == n
        clip = x;
    else
        error('clip_min/clip_max length mismatch: got %d, expected 1 or %d.',numel(x),n);
    end
end

function val = getfield_default_local(s,name,default_val)
    if isfield(s,name) && ~isempty(s.(name))
        val = s.(name);
    else
        val = default_val;
    end
end

function s = vec_string_local(v)
    s = sprintf(' %.5g',v(:));
    s = strtrim(s);
end

function [r_opt,MSE_opt,alpha_opt,term_opt,hist] = train_frozen_jacobian_local( ...
    X,target,raw,r0,opts,family,lambda_ridge,extra)
%TRAIN_FROZEN_JACOBIAN_LOCAL
% r-training against the frozen Newton-Jacobian response with the ANALYTIC
% r-gradient (matches the legacy ADRaNN 'kdv_residual' mode):
%   min_r min_a || J(u) Phi(r) a - target ||^2 + lambda ||a||^2
% The gradient is normalized by the current MSE when opts.use_rel_grad is
% true, so r keeps moving even when the residual is tiny.
    X = gather(X);
    target = gather(target(:));
    r = gather(r0(:));
    opts = fill_opts_local(opts,r);

    problem = extra.problem;
    sol = extra.sol;
    Jc = ProblemFunction('jacobian', X, problem, sol);

    clip_min = expand_clip_local(opts.clip_min,numel(r),-inf);
    clip_max = expand_clip_local(opts.clip_max,numel(r), inf);

    m_adam = zeros(numel(r),1);
    v_adam = zeros(numel(r),1);

    MSE_opt = inf;
    r_opt = r;
    alpha_opt = [];
    term_opt = build_term_from_r_local(family,raw,r);
    best_iter = 0;

    hist.MSE = nan(max(opts.maxit,1),1);
    hist.r = nan(numel(r),max(opts.maxit,1));
    hist.grad_norm = nan(max(opts.maxit,1),1);

    MSE_prev = inf;
    stall_count = 0;

    for it = 1:opts.maxit
        [M,DM,term] = matrix_frozen_jacobian_dr_local(X,raw,r,family,Jc);
        [MSE,alpha,res] = ridge_objective_frozen_local(M,target,lambda_ridge);

        g = zeros(numel(r),1);
        Nrow = size(M,1);
        for kk = 1:numel(r)
            dMa = DM{kk} * alpha;
            if opts.use_rel_grad
                g(kk) = (res.' * dMa) / (Nrow*MSE + eps);
            else
                g(kk) = (2/Nrow) * (res.' * dMa);
            end
        end
        g(~isfinite(g)) = 0;
        gnorm = norm(g);

        hist.MSE(it) = MSE;
        hist.r(:,it) = r;
        hist.grad_norm(it) = gnorm;

        if MSE < MSE_opt
            MSE_opt = MSE;
            r_opt = r;
            alpha_opt = alpha;
            term_opt = term;
            best_iter = it;
        end

        if opts.verbose && (mod(it,opts.print_every)==0 || it==1)
            fprintf('  r-train frozen %-14s it %3d: MSE=%.6e, r=[%s], |g|=%.3e\n', ...
                family,it,MSE,vec_string_local(r),gnorm);
        end

        if it >= opts.minit && gnorm < opts.grad_tol
            break;
        end

        if opts.rel_mse_tol > 0 && isfinite(MSE_prev)
            rel_drop = (MSE_prev - MSE)/(abs(MSE_prev)+eps);
            if rel_drop >= 0 && rel_drop < opts.rel_mse_tol
                stall_count = stall_count + 1;
            else
                stall_count = 0;
            end
            if stall_count >= opts.stall_patience
                break;
            end
        end

        if MSE <= opts.mse_tol
            break;
        end

        MSE_prev = MSE;

        m_adam = 0.9*m_adam + 0.1*g;
        v_adam = 0.999*v_adam + 0.001*(g.^2);
        mh = m_adam/(1-0.9^it);
        vh = v_adam/(1-0.999^it);
        r = r - opts.eta * mh ./ (sqrt(vh) + opts.eps_adam);
        r = min(max(r,clip_min),clip_max);
    end

    valid = isfinite(hist.MSE);
    hist.MSE = hist.MSE(valid);
    hist.r = hist.r(:,valid);
    hist.grad_norm = hist.grad_norm(valid);
    hist.best_iter = best_iter;
    hist.best_MSE = MSE_opt;
    hist.best_r = r_opt(:);
end

function [MSE,alpha,res] = ridge_objective_frozen_local(M,target,lambda)
    M = gather(M);
    target = gather(target(:));
    K = size(M,2);
    G = M.'*M + lambda*eye(K);
    rhs = M.'*target;
    [L,p] = chol(G,'lower');
    if p == 0
        alpha = L'\(L\rhs);
    else
        alpha = G\rhs;
    end
    res = M*alpha - target;
    MSE = mean(res.^2);
end

function [M,DM,term] = matrix_frozen_jacobian_dr_local(X,raw,r,family,Jc)
%MATRIX_FROZEN_JACOBIAN_DR_LOCAL
% M = J(u) Phi(r) and the analytic derivative DM{kk} = dM/dr_kk.
% Ridge families (tanh / ridge_gauss / sin) use the exact dM/dr; the other
% families fall back to central finite differences in the two scales.
    fam = lower(family);
    if any(strcmp(fam,{'tanh','gauss','ridge_gauss','sin'})) && isfield(raw,'w_basis') && ~isempty(raw.w_basis)
        [Phi,DPhi]       = ridge_basis_and_dr_local(family,raw,r,X,0,0);
        [Phix,DPhix]     = ridge_basis_and_dr_local(family,raw,r,X,1,1);
        [Phit,DPhit]     = ridge_basis_and_dr_local(family,raw,r,X,1,2);
        [Phixxx,DPhixxx] = ridge_basis_and_dr_local(family,raw,r,X,3,1);
        M = Jc.ut .* Phit + Jc.ux .* Phix + Jc.uxxx .* Phixxx;
        has_uxx = isfield(Jc,'uxx') && ~isempty(Jc.uxx);
        if isfield(Jc,'u')
            M = M + Jc.u .* Phi;
        end
        if has_uxx
            [Phixx,DPhixx] = ridge_basis_and_dr_local(family,raw,r,X,2,1);
            M = M + Jc.uxx .* Phixx;
        else
            DPhixx = [];
        end
        DM = cell(numel(r),1);
        for kk = 1:numel(r)
            DM{kk} = Jc.ut .* DPhit{kk} + Jc.ux .* DPhix{kk} + Jc.uxxx .* DPhixxx{kk};
            if isfield(Jc,'u')
                DM{kk} = DM{kk} + Jc.u .* DPhi{kk};
            end
            if has_uxx
                DM{kk} = DM{kk} + Jc.uxx .* DPhixx{kk};
            end
        end
        term = build_term_from_r_local(family,raw,r);
        return;
    end

    % Fallback: central finite differences in the two scale variables.
    M = matrix_frozen_jacobian_local(X,raw,r,family,Jc);
    term = build_term_from_r_local(family,raw,r);
    DM = cell(numel(r),1);
    for kk = 1:numel(r)
        h = 1e-5*max(1,abs(r(kk)));
        rp = r; rm = r;
        rp(kk) = rp(kk) + h;
        rm(kk) = max(1e-12, rm(kk) - h);
        hp = rp(kk) - rm(kk);
        Mp = matrix_frozen_jacobian_local(X,raw,rp,family,Jc);
        Mm = matrix_frozen_jacobian_local(X,raw,rm,family,Jc);
        DM{kk} = (Mp - Mm)/hp;
    end
end

function M = matrix_frozen_jacobian_local(X,raw,r,family,Jc)
    term = build_term_from_r_local(family,raw,r);
    Phi    = BasisFunction(family,term,X,0,0);
    Phix   = BasisFunction(family,term,X,1,1);
    Phit   = BasisFunction(family,term,X,1,2);
    Phixxx = BasisFunction(family,term,X,3,1);
    M = Jc.ut .* Phit + Jc.ux .* Phix + Jc.uxxx .* Phixxx;
    if isfield(Jc,'u')
        M = M + Jc.u .* Phi;
    end
    if isfield(Jc,'uxx')
        M = M + Jc.uxx .* BasisFunction(family,term,X,2,1);
    end
    M = gather(M);
end

function [M,DM] = ridge_basis_and_dr_local(family,raw,r,X,ord,dir)
%RIDGE_BASIS_AND_DR_LOCAL
% Ridge feature matrix M = f^{(ord)}(z) .* P and its analytic dM/dr.
% Ported from the legacy ADRaNN 'kdv_residual' implementation.
    if strcmpi(family,'ridge_gauss'), family = 'gauss'; end
    W0 = raw.w_basis;
    centers = raw.centers;
    W = r(:).*W0;
    z = X*W - sum(centers.*W,1);

    [P,dP] = ridge_derivative_prefactor_local(W,W0,ord,dir);

    dz = cell(numel(r),1);
    dz{1} = bsxfun(@times,X(:,1) - centers(1,:),W0(1,:));
    dz{2} = bsxfun(@times,X(:,2) - centers(2,:),W0(2,:));

    f_ord  = activation_z_deriv_local(family,z,ord);
    f_next = activation_z_deriv_local(family,z,ord+1);
    M = f_ord .* P;

    DM = cell(numel(r),1);
    for kk = 1:numel(r)
        DM{kk} = f_next .* dz{kk} .* P + f_ord .* dP{kk};
    end
end

function [P,dP] = ridge_derivative_prefactor_local(W,W0,ord,dir)
    m = size(W,2);
    dP = {zeros(1,m),zeros(1,m)};
    if ord == 0
        P = ones(1,m);
        return;
    end
    if dir == 1 || dir == 2
        d = dir;
        P = W(d,:).^ord;
        dP{d} = ord * W(d,:).^(ord-1) .* W0(d,:);
    elseif dir == 3 && ord == 2
        P = W(1,:).*W(2,:);
        dP{1} = W0(1,:).*W(2,:);
        dP{2} = W(1,:).*W0(2,:);
    else
        error('Unsupported ridge derivative prefactor ord=%d, dir=%d.',ord,dir);
    end
end

function y = activation_z_deriv_local(family,z,k)
    switch lower(family)
        case 'tanh'
            T = tanh(z);
            switch k
                case 0, y = T;
                case 1, y = 1 - T.^2;
                case 2, y = -2*T.*(1-T.^2);
                case 3, y = -2 + 8*T.^2 - 6*T.^4;
                case 4, y = 16*T - 40*T.^3 + 24*T.^5;
                otherwise, error('tanh derivative order %d unsupported.',k);
            end
        case 'gauss'
            E = exp(-z.^2);
            switch k
                case 0, y = E;
                case 1, y = -2*z.*E;
                case 2, y = (4*z.^2 - 2).*E;
                case 3, y = (12*z - 8*z.^3).*E;
                case 4, y = (12 - 48*z.^2 + 16*z.^4).*E;
                otherwise, error('gauss derivative order %d unsupported.',k);
            end
        case 'sin'
            y = (pi^k) * sin(pi*z + k*pi/2);
        otherwise
            error('Unsupported ridge activation = %s',family);
    end
end
