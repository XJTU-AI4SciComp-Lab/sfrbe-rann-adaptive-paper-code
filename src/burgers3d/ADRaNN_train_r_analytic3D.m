function [r_opt,MSE_opt,alpha_opt,term_opt,hist] = ADRaNN_train_r_analytic3D(mode,X,target,raw,r0,opts,family,lambda_ridge,extra)
%ADRaNN_TRAIN_R_ANALYTIC Analytic-gradient 3D distribution-scale training.
%
% X(:,1)=x, X(:,2)=y, X(:,3)=t.
%
% Inner readout solve is ridge-regularized:
%   alpha(r) = (M(r)'M(r)+lambda I)\(M(r)'target)
% and the r-gradient uses the envelope theorem:
%   d mean(res.^2)/dr_k = (2/N) res' (dM/dr_k) alpha.
%
% modes:
%   value              : M(r)=Phi(r), DDAD/value fitting.
%   operator           : M(r)=linear operator Phi(r).
%   burgers2d_residual : M(r)=J_Burgers2D(u_ref) Phi(r).
%
% Supported analytic families:
%   tanh, ridge_gauss/gauss, sin, radial_gauss, mexican_hat.

    if nargin < 9 || isempty(extra), extra = struct(); end
    if nargin < 8 || isempty(lambda_ridge), lambda_ridge = 1e-6; end
    if nargin < 7 || isempty(family)
        if isfield(raw,'family'), family = raw.family; else, family = 'tanh'; end
    end
    if nargin < 6 || isempty(opts), opts = struct(); end

    X = gather(X);
    target = gather(target(:));
    r = gather(r0(:));

    opts = fill_opts_local(opts,numel(r));
    clip_min = opts.clip_min(:);
    clip_max = opts.clip_max(:);

    m_adam = zeros(numel(r),1);
    v_adam = zeros(numel(r),1);

    MSE_opt = inf;
    r_opt = r;
    alpha_opt = [];
    term_opt = build_term_from_r_helper_local(family,raw,r);

    hist.MSE = zeros(max(opts.maxit,1),1);
    hist.r = zeros(numel(r),max(opts.maxit,1));
    hist.grad_norm = zeros(max(opts.maxit,1),1);

    MSE_prev = inf;
    stall_count = 0;

    if opts.verbose
        fprintf('==== Analytic Adam r-training: mode=%s, family=%s, r_dim=%d ====\n', ...
            mode,family,numel(r));
    end

    % maxit=0: objective-only compatibility path.
    if opts.maxit <= 0
        [M,~,term_opt] = matrix_and_dr_local(mode,X,raw,r,family,extra);
        [MSE_opt,alpha_opt] = ridge_objective_local(M,target,lambda_ridge);
        r_opt = r;
        hist.MSE = MSE_opt;
        hist.r = r;
        hist.grad_norm = 0;
        return;
    end

    for it = 1:opts.maxit
        [M,DM,term] = matrix_and_dr_local(mode,X,raw,r,family,extra);
        [MSE,alpha,res] = ridge_objective_local(M,target,lambda_ridge);

        g = zeros(numel(r),1);
        N = size(M,1);
        for kk = 1:numel(r)
            dMa = DM{kk} * alpha;
            if opts.use_rel_grad
                g(kk) = (res.' * dMa) / (N*MSE + eps);
            else
                g(kk) = (2/N) * (res.' * dMa);
            end
        end

        gnorm = norm(g);
        hist.MSE(it) = MSE;
        hist.r(:,it) = r;
        hist.grad_norm(it) = gnorm;

        if MSE < MSE_opt
            MSE_opt = MSE;
            r_opt = r;
            alpha_opt = alpha;
            term_opt = term;
        end

        if opts.verbose && (opts.verbose_iter || it == 1 || it == opts.maxit || mod(it,opts.print_every)==0)
            fprintf('  Adam %3d: MSE=%.6e, |g|=%.3e, r=(%s)\n', ...
                it,MSE,gnorm,sprintf(' %.6g',r));
        end

        if it >= opts.minit
            if gnorm < opts.grad_tol
                if opts.verbose, fprintf('  stop: grad norm < %.3e\n',opts.grad_tol); end
                break;
            end
            if MSE < opts.mse_tol
                if opts.verbose, fprintf('  stop: MSE < %.3e\n',opts.mse_tol); end
                break;
            end
            rel_drop = (MSE_prev - MSE)/(abs(MSE_prev)+eps);
            if rel_drop >= 0 && rel_drop < opts.rel_mse_tol
                stall_count = stall_count + 1;
            else
                stall_count = 0;
            end
            if stall_count >= opts.stall_patience
                if opts.verbose, fprintf('  stop: MSE stalls.\n'); end
                break;
            end
        end

        MSE_prev = MSE;

        m_adam = opts.beta1*m_adam + (1-opts.beta1)*g;
        v_adam = opts.beta2*v_adam + (1-opts.beta2)*(g.^2);
        mhat = m_adam/(1-opts.beta1^it);
        vhat = v_adam/(1-opts.beta2^it);

        r = r - opts.eta * mhat ./ (sqrt(vhat) + opts.eps_adam);
        r = min(max(r,clip_min),clip_max);
    end

    hist.MSE = hist.MSE(hist.MSE>0);
    hist.r = hist.r(:,1:numel(hist.MSE));
    hist.grad_norm = hist.grad_norm(1:numel(hist.MSE));
end

%% ============================================================
% options / objective
%% ============================================================
function opts = fill_opts_local(opts,rdim)
    if ~isfield(opts,'maxit'), opts.maxit = 20; end
    if ~isfield(opts,'eta'), opts.eta = 0.25; end
    if ~isfield(opts,'beta1'), opts.beta1 = 0.9; end
    if ~isfield(opts,'beta2'), opts.beta2 = 0.999; end
    if ~isfield(opts,'eps_adam'), opts.eps_adam = 1e-8; end
    if ~isfield(opts,'clip_min'), opts.clip_min = ones(rdim,1); end
    if ~isfield(opts,'clip_max'), opts.clip_max = 500*ones(rdim,1); end
    if numel(opts.clip_min) == 1, opts.clip_min = opts.clip_min*ones(rdim,1); end
    if numel(opts.clip_max) == 1, opts.clip_max = opts.clip_max*ones(rdim,1); end
    if ~isfield(opts,'minit'), opts.minit = 5; end
    if ~isfield(opts,'grad_tol'), opts.grad_tol = 1e-12; end
    if ~isfield(opts,'verbose'), opts.verbose = false; end
    if ~isfield(opts,'verbose_iter'), opts.verbose_iter = false; end
    if ~isfield(opts,'print_every'), opts.print_every = 1; end
    if ~isfield(opts,'use_rel_grad'), opts.use_rel_grad = true; end
    if ~isfield(opts,'mse_tol'), opts.mse_tol = -inf; end
    if ~isfield(opts,'rel_mse_tol'), opts.rel_mse_tol = 0; end
    if ~isfield(opts,'stall_patience'), opts.stall_patience = inf; end
end

function [MSE,alpha,res] = ridge_objective_local(M,target,lambda_ridge)
    M = gather(M);
    target = gather(target(:));
    alpha = (M.'*M + lambda_ridge*eye(size(M,2))) \ (M.'*target);
    res = M*alpha - target;
    MSE = mean(res.^2);
end

%% ============================================================
% mode-specific matrix and analytic dM/dr
%% ============================================================
function [M,DM,term] = matrix_and_dr_local(mode,X,raw,r,family,extra)
    term = build_term_from_r_helper_local(family,raw,r);

    switch lower(mode)
        case {'value','ddad','data'}
            [M,DM] = basis_and_dr_local(family,raw,r,X,0,0);

        case {'operator','pdad','residual_operator'}
            if ~isfield(extra,'op'), error('mode=operator requires extra.op'); end
            [M,DM] = operator_and_dr_local(family,raw,r,X,extra.op);

        case {'burgers2d','burgers2d_residual','newton_residual'}
            if ~isfield(extra,'problem') || ~isfield(extra,'u_ref') || ...
               ~isfield(extra,'ux_ref') || ~isfield(extra,'uy_ref')
                error('mode=burgers2d_residual requires extra.problem, u_ref, ux_ref, uy_ref');
            end

            [Phi,DPhi]     = basis_and_dr_local(family,raw,r,X,0,0);
            [Phix,DPhix]   = basis_and_dr_local(family,raw,r,X,1,1);
            [Phiy,DPhiy]   = basis_and_dr_local(family,raw,r,X,1,2);
            [Phit,DPhit]   = basis_and_dr_local(family,raw,r,X,1,3);
            [Phixx,DPhixx] = basis_and_dr_local(family,raw,r,X,2,1);
            [Phiyy,DPhiyy] = basis_and_dr_local(family,raw,r,X,2,2);

            u  = extra.u_ref(:);
            ux = extra.ux_ref(:);
            uy = extra.uy_ref(:);
            epsv = extra.problem.eps_burgers;

            M = Phit + (ux+uy).*Phi + u.*(Phix+Phiy) - epsv*(Phixx+Phiyy);

            DM = cell(numel(r),1);
            for kk = 1:numel(r)
                DM{kk} = DPhit{kk} + (ux+uy).*DPhi{kk} + ...
                         u.*(DPhix{kk}+DPhiy{kk}) - epsv*(DPhixx{kk}+DPhiyy{kk});
            end

        otherwise
            error('Unknown r-training mode = %s',mode);
    end
end

function [A,DA] = operator_and_dr_local(family,raw,r,X,op)
    [U,DU]       = basis_and_dr_local(family,raw,r,X,0,0);
    [Ux,DUx]     = basis_and_dr_local(family,raw,r,X,1,1);
    [Uy,DUy]     = basis_and_dr_local(family,raw,r,X,1,2);
    [Ut,DUt]     = basis_and_dr_local(family,raw,r,X,1,3);
    [Uxx,DUxx]   = basis_and_dr_local(family,raw,r,X,2,1);
    [Uyy,DUyy]   = basis_and_dr_local(family,raw,r,X,2,2);
    [Utt,DUtt]   = basis_and_dr_local(family,raw,r,X,2,3);

    N = size(X,1); Mcol = size(U,2);
    a0  = toNM_local(getfield_default_local(op,'alpha0',0),N,Mcol);
    ax  = toNM_local(getfield_default_local(op,'alpha_x',getfield_default_local(op,'alpha1',0)),N,Mcol);
    ay  = toNM_local(getfield_default_local(op,'alpha_y',0),N,Mcol);
    at  = toNM_local(getfield_default_local(op,'alpha_t',getfield_default_local(op,'alpha3',0)),N,Mcol);
    axx = toNM_local(getfield_default_local(op,'alpha_xx',getfield_default_local(op,'alpha2',0)),N,Mcol);
    ayy = toNM_local(getfield_default_local(op,'alpha_yy',0),N,Mcol);
    att = toNM_local(getfield_default_local(op,'alpha_tt',getfield_default_local(op,'alpha4',0)),N,Mcol);

    A = a0.*U + ax.*Ux + ay.*Uy + at.*Ut + axx.*Uxx + ayy.*Uyy + att.*Utt;
    DA = cell(numel(r),1);
    for kk = 1:numel(r)
        DA{kk} = a0.*DU{kk} + ax.*DUx{kk} + ay.*DUy{kk} + ...
                 at.*DUt{kk} + axx.*DUxx{kk} + ayy.*DUyy{kk} + att.*DUtt{kk};
    end
end

function Acoef = toNM_local(a,N,M)
    if isscalar(a)
        Acoef = a * ones(N,M);
    else
        Acoef = a(:) * ones(1,M);
    end
end

%% ============================================================
% term builders
%% ============================================================
function term = build_term_from_r_helper_local(family,raw,r)
    r = r(:);
    centers = raw.centers;
    switch lower(family)
        case 'tanh'
            W = r .* raw.w_basis;
            term.type = 'tanh'; term.w = W; term.b = -sum(centers.*W,1);
        case {'ridge_gauss','gauss'}
            W = r .* raw.w_basis;
            term.type = 'gauss'; term.w = W; term.b = -sum(centers.*W,1);
        case 'sin'
            W = r .* raw.w_basis;
            term.type = 'sin'; term.w = W; term.b = -sum(centers.*W,1);
        case 'radial_gauss'
            term.type = 'radial_gauss'; term.A = centers; term.r = r;
        case 'mexican_hat'
            term.type = 'mexican_hat'; term.A = centers; term.r = r;
        otherwise
            error('ADRaNN_train_r_analytic3D: unsupported analytic family = %s',family);
    end
end

%% ============================================================
% basis and analytic d/dr
%% ============================================================
function [B,DB] = basis_and_dr_local(family,raw,r,X,ord,dir)
    switch lower(family)
        case {'tanh','ridge_gauss','gauss','sin'}
            [B,DB] = ridge_basis_and_dr_local(family,raw,r,X,ord,dir);
        case 'radial_gauss'
            [B,DB] = radial_gauss_basis_and_dr_local(raw,r,X,ord,dir);
        case 'mexican_hat'
            [B,DB] = mexican_hat_basis_and_dr_local(raw,r,X,ord,dir);
        otherwise
            error('Analytic d/dr is not implemented for family = %s',family);
    end
end

function [B,DB] = ridge_basis_and_dr_local(family,raw,r,X,ord,dir)
    r = r(:);
    Z = raw.w_basis;
    C = raw.centers;
    W = r .* Z;
    S = X*W - sum(C.*W,1);

    nPar = numel(r);
    if ord == 0
        Wfac = ones(1,size(Z,2));
        dWfac = zeros(nPar,size(Z,2));
        nder = 0;
    else
        if dir <= 0 || dir > size(W,1)
            error('Invalid derivative dir=%d',dir);
        end
        Wfac = W(dir,:).^ord;
        dWfac = zeros(nPar,size(Z,2));
        dWfac(dir,:) = ord * W(dir,:).^(ord-1) .* Z(dir,:);
        nder = ord;
    end

    phi_n  = activation_deriv_local(family,S,nder);
    phi_np = activation_deriv_local(family,S,nder+1);
    B = Wfac .* phi_n;

    DB = cell(nPar,1);
    for kk = 1:nPar
        dS = Z(kk,:) .* (X(:,kk) - C(kk,:));
        DB{kk} = dWfac(kk,:) .* phi_n + Wfac .* (phi_np .* dS);
    end
end

function D = activation_deriv_local(family,S,n)
    switch lower(family)
        case 'tanh'
            T = tanh(S); Q = 1 - T.^2;
            switch n
                case 0, D = T;
                case 1, D = Q;
                case 2, D = -2*T.*Q;
                case 3, D = (6*T.^2 - 2).*Q;
                case 4, D = 8*T.*(2 - 3*T.^2).*Q;
                otherwise, error('tanh analytic derivative supports n<=4');
            end
        case {'ridge_gauss','gauss'}
            E = exp(-S.^2);
            switch n
                case 0, D = E;
                case 1, D = -2*S.*E;
                case 2, D = (4*S.^2 - 2).*E;
                case 3, D = (-8*S.^3 + 12*S).*E;
                case 4, D = (16*S.^4 - 48*S.^2 + 12).*E;
                otherwise, error('gauss analytic derivative supports n<=4');
            end
        case 'sin'
            switch mod(n,4)
                case 0, D = sin(S);
                case 1, D = cos(S);
                case 2, D = -sin(S);
                case 3, D = -cos(S);
            end
        otherwise
            error('unknown activation family = %s',family);
    end
end

function [B,DB] = radial_gauss_basis_and_dr_local(raw,r,X,ord,dir)
    r = r(:); C = raw.centers;
    d = numel(r);
    D = cell(d,1);
    S = zeros(size(X,1),size(C,2));
    for kk = 1:d
        D{kk} = X(:,kk) - C(kk,:);
        S = S + r(kk)^2 * D{kk}.^2;
    end
    E = exp(-S);

    if ord == 0
        P = ones(size(E));
        dP = cell(d,1);
        for kk=1:d, dP{kk}=zeros(size(E)); end
    else
        rd = r(dir); dd = D{dir};
        switch ord
            case 1
                P = -2*rd^2*dd;
            case 2
                P = 4*rd^4*dd.^2 - 2*rd^2;
            case 3
                P = -8*rd^6*dd.^3 + 12*rd^4*dd;
            otherwise
                error('radial_gauss supports ord<=3');
        end
        dP = cell(d,1);
        for kk=1:d, dP{kk}=zeros(size(E)); end
        switch ord
            case 1
                dP{dir} = -4*rd*dd;
            case 2
                dP{dir} = 16*rd^3*dd.^2 - 4*rd;
            case 3
                dP{dir} = -48*rd^5*dd.^3 + 48*rd^3*dd;
        end
    end

    B = P.*E;
    DB = cell(d,1);
    for kk = 1:d
        dE = -2*r(kk)*D{kk}.^2.*E;
        DB{kk} = dP{kk}.*E + P.*dE;
    end
end

function [B,DB] = mexican_hat_basis_and_dr_local(raw,r,X,ord,dir)
    r = r(:); C = raw.centers;
    d = numel(r);
    D = cell(d,1);
    s = zeros(size(X,1),size(C,2));
    for kk = 1:d
        D{kk} = X(:,kk) - C(kk,:);
        s = s + r(kk)^2 * D{kk}.^2;
    end

    E = exp(-0.5*s);
    g0 = (2-s).*E;
    g1 = (0.5*s - 2).*E;
    g2 = (1.5 - 0.25*s).*E;
    g3 = (0.125*s - 1).*E;
    g4 = ((10 - s)/16).*E;

    sr = cell(d,1);
    for kk=1:d
        sr{kk} = 2*r(kk).*D{kk}.^2;
    end

    if ord == 0
        B = g0;
        DB = cell(d,1);
        for kk=1:d, DB{kk} = g1.*sr{kk}; end
        return;
    end

    sj  = 2*r(dir)^2.*D{dir};
    sjj = 2*r(dir)^2;

    sj_r = cell(d,1); sjj_r = cell(d,1);
    for kk=1:d
        sj_r{kk} = zeros(size(s));
        sjj_r{kk} = zeros(size(s));
    end
    sj_r{dir} = 4*r(dir).*D{dir};
    sjj_r{dir} = 4*r(dir).*ones(size(s));

    switch ord
        case 1
            B = g1.*sj;
            DB = cell(d,1);
            for kk=1:d
                DB{kk} = g2.*sr{kk}.*sj + g1.*sj_r{kk};
            end
        case 2
            B = g2.*sj.^2 + g1.*sjj;
            DB = cell(d,1);
            for kk=1:d
                DB{kk} = g3.*sr{kk}.*sj.^2 + ...
                         g2.*(2*sj.*sj_r{kk} + sr{kk}.*sjj) + ...
                         g1.*sjj_r{kk};
            end
        case 3
            B = g3.*sj.^3 + 3*g2.*sj.*sjj;
            DB = cell(d,1);
            for kk=1:d
                DB{kk} = g4.*sr{kk}.*sj.^3 + g3.*(3*sj.^2.*sj_r{kk}) + ...
                         3*(g3.*sr{kk}.*sj.*sjj + g2.*(sj_r{kk}.*sjj + sj.*sjj_r{kk}));
            end
        otherwise
            error('mexican_hat supports ord<=3');
    end
end

function val = getfield_default_local(s,name,default_val)
    if isstruct(s) && isfield(s,name) && ~isempty(s.(name))
        val = s.(name);
    else
        val = default_val;
    end
end
