function [coef, info] = FixedOperatorSolver(problem, basisEval, X_int, cons, coef0, cfg)
%FIXEDOPERATORSOLVER Stabilized fixed-operator Picard iteration.
%
% The linear differential operator is assembled once.  The nonlinear term
% is evaluated at the previous iterate and moved to the right-hand side:
%
%   (L + sigma I) u_{k+1} = rhs - N(u_k) + sigma u_k.
%
% The stabilization term vanishes at a fixed point, so the converged
% solution still satisfies the original nonlinear PDE.  A simple residual
% line search/under-relaxation prevents an unstable Picard update from being
% accepted.

    if nargin < 6 || isempty(cfg), cfg = struct(); end
    cfg = fill_cfg_local(cfg);

    coef = coef0(:);
    X_int = gather(X_int);
    Xc = gather(cons.X);
    target = gather(cons.target(:));
    types = cons.types;
    % Periodic KdV uses explicit periodic_* labels for the paired boundary
    % constraints; treat them as the corresponding value/derivative rows.
    idxV = strcmpi(types,'value') | strcmpi(types,'periodic_u');
    idxUx = strcmpi(types,'ux') | strcmpi(types,'periodic_ux');
    idxUxx = strcmpi(types,'uxx') | strcmpi(types,'periodic_uxx');

    % Assemble the constraint rows.  The full basis is still needed for the
    % nonlinear residual, but an accepted RBE block can provide a callback
    % for only its new operator columns, avoiding re-evaluation of the old
    % columns during the SVD update.
    Phi = basisEval(0,0,X_int);
    Pux = basisEval(1,1,X_int);
    Put = basisEval(1,2,X_int);
    Puxx = basisEval(2,1,X_int);
    P22 = basisEval(2,2,X_int);
    P23 = basisEval(2,3,X_int);
    Puxxx = basisEval(3,1,X_int);
    Bv = basisEval(0,0,Xc);
    Bu = basisEval(1,1,Xc);
    Buu = basisEval(2,1,Xc);
    Bc = [Bv(idxV,:); Bu(idxUx,:); Buu(idxUxx,:)];
    yc = [target(idxV); target(idxUx); target(idxUxx)];
    basis_cache = struct('P0',Phi,'Pux',Pux,'Put',Put,'Puxx',Puxx, ...
        'P22',P22,'P23',P23,'Puxxx',Puxxx,'Bv',Bv,'Bu',Bu,'Buu',Buu);

    sigma = cfg.sigma;
    state_old = getfield_default_local(cfg,'svd_state',[]);
    if isfield(cfg,'block_basis_eval') && ~isempty(cfg.block_basis_eval) && ...
            svd_state_compatible_local(state_old,size(Phi,1)+size(Bc,1),size(Phi,2))
        blockEval = cfg.block_basis_eval;
        Anew = assemble_operator_basis_local(blockEval, X_int, cfg.op);
        Phi_new = blockEval(0,0,X_int);
        Bv_new = blockEval(0,0,Xc);
        Bu_new = blockEval(1,1,Xc);
        Buu_new = blockEval(2,1,Xc);
        Bc_new = [Bv_new(idxV,:); Bu_new(idxUx,:); Buu_new(idxUxx,:)];
        Anew_fix = [Anew + sigma*Phi_new; cfg.eta_bc * Bc_new];
        [U,s,V,svd_state,svd_mode] = append_svd_state_local(Anew_fix,state_old);
        clear Anew_fix Anew Phi_new Bv_new Bu_new Buu_new Bc_new
    else
        A = assemble_operator_from_cache_local(basis_cache, X_int, cfg.op);
        Afix = [A + sigma*Phi; cfg.eta_bc * Bc];
        % The fixed operator is unchanged when RBE only appends a new block.
        % Reuse the cached factorization in that case and factor only the
        % orthogonal complement of the appended columns.  A full SVD is used
        % only for the first space (or when the cache is incompatible).
        [U,s,V,svd_state,svd_mode] = update_svd_state_local(Afix, state_old);
        clear Afix A
    end

    [cost, relres] = nonlinear_cost_local(problem,basis_cache,X_int,coef,...
        target,idxV,idxUx,idxUxx,cfg.eta_bc);
    cost0 = max(cost,1e-30);
    accepted_it = 0;
    omega = cfg.omega;
    history = nan(cfg.maxit,3);

    for it = 1:cfg.maxit
        u = basis_cache.P0 * coef;
        ux = basis_cache.Pux * coef;
        ut = basis_cache.Put * coef;
        uxx = basis_cache.Puxx * coef;
        uxxx = basis_cache.Puxxx * coef;
        sol = struct('u',u,'ux',ux,'ut',ut,'uxx',uxx,'uxxx',uxxx);
        Nold = ProblemFunction('nonlinear',X_int,problem,sol);
        rhs_int = getfield_default_local(cfg,'rhs_int',zeros(size(X_int,1),1));
        rhs = [rhs_int - Nold + sigma*u; cfg.eta_bc*yc];
        coef_trial = apply_svd_state_local(U,s,V,gather(rhs));

        improved = false;
        omega_try = omega;
        for ls = 1:cfg.max_line_search
            coef_candidate = (1-omega_try)*coef + omega_try*coef_trial;
            [cost_cand, rel_cand] = nonlinear_cost_local(problem,basis_cache,X_int,...
                coef_candidate,target,idxV,idxUx,idxUxx,cfg.eta_bc);
            if cost_cand <= cost*(1 + cfg.accept_increase)
                improved = true;
                break;
            end
            omega_try = 0.5*omega_try;
            if omega_try < cfg.omega_min, break; end
        end

        if ~improved
            if cfg.verbose
                fprintf('  [FixedOp] no improving Picard update; stopping at it=%d.\n',it);
            end
            break;
        end

        coef = coef_candidate;
        cost = cost_cand;
        relres = rel_cand;
        accepted_it = it;
        omega = min(cfg.omega, max(omega_try, cfg.omega_min));
        history(it,:) = [cost, relres, omega_try];
        if cfg.verbose
            fprintf('  [FixedOp] it=%3d loss=%.6e rel_res=%.4e omega=%.3g\n', ...
                it,cost,relres,omega_try);
        end
        if sqrt(cost/cost0) <= cfg.tol || relres <= cfg.tol
            break;
        end
    end

    info.loss = cost;
    info.rel_res = relres;
    info.iter = accepted_it;
    info.omega = omega;
    info.sigma = sigma;
    info.factorization = svd_mode;
    % RBE commits this state only if the trial block is accepted.  Keeping
    % it in info makes rejected trials harmless and accepted trials cheap.
    info.svd_state = svd_state;
    info.residual_only_line_search = true;
    info.history = history(1:max(accepted_it,1),:);
end

function [U,s,V,state,mode] = update_svd_state_local(A,state_old)
%UPDATE_SVD_STATE_LOCAL Full initialization followed by incremental SVD.
    nrow = size(A,1);
    ncol = size(A,2);
    valid = isstruct(state_old) && isfield(state_old,'U') && ...
        isfield(state_old,'s') && isfield(state_old,'V') && ...
        isfield(state_old,'nrows') && isfield(state_old,'ncols') && ...
        state_old.nrows == nrow && state_old.ncols <= ncol && ...
        ~isempty(state_old.U) && ~isempty(state_old.s) && ~isempty(state_old.V);
    if ~valid
        [U,S,V] = svd(gather(A),'econ');
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
    [U,s,V,state,mode] = append_svd_state_local(gather(A(:,nold+1:ncol)),state_old);
end

function tf = svd_state_compatible_local(state,nrow,ncol)
    tf = isstruct(state) && isfield(state,'U') && isfield(state,'s') && ...
        isfield(state,'V') && isfield(state,'nrows') && isfield(state,'ncols') && ...
        state.nrows == nrow && state.ncols < ncol && ...
        ~isempty(state.U) && ~isempty(state.s) && ~isempty(state.V);
end

function [U,s,V,state,mode] = append_svd_state_local(Anew,state_old)
%APPEND_SVD_STATE_LOCAL Incremental SVD using a small core matrix.
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
    Vaug = blkdiag(Vold,eye(k));
    V = Vaug * Vk;
    state = struct('U',U,'s',s,'V',V,'nrows',size(U,1), ...
        'ncols',size(V,2),'mode','incremental_block_svd');
    mode = 'incremental_block_svd';
end

function coef = apply_svd_state_local(U,s,V,rhs)
    z = U' * rhs;
    tol = max(size(U)) * eps(max([s(:);1]));
    invs = zeros(size(s));
    keep = abs(s) > tol;
    invs(keep) = 1 ./ s(keep);
    coef = V * (invs .* z);
end

function cfg = fill_cfg_local(cfg)
    if ~isfield(cfg,'op') || isempty(cfg.op)
        error('FixedOperatorSolver requires cfg.op.');
    end
    if ~isfield(cfg,'eta_bc'), cfg.eta_bc = 1e5; end
    if ~isfield(cfg,'sigma'), cfg.sigma = 0; end
    if ~isfield(cfg,'omega'), cfg.omega = 0.7; end
    if ~isfield(cfg,'omega_min'), cfg.omega_min = 1e-3; end
    if ~isfield(cfg,'maxit'), cfg.maxit = 25; end
    if ~isfield(cfg,'max_line_search'), cfg.max_line_search = 10; end
    if ~isfield(cfg,'tol'), cfg.tol = 1e-7; end
    if ~isfield(cfg,'accept_increase'), cfg.accept_increase = 0; end
    if ~isfield(cfg,'verbose'), cfg.verbose = false; end
end

function A = assemble_operator_basis_local(basisEval,X,op)
    A = zeros(size(X,1),size(basisEval(0,0,X),2));
    A = A + coef_times_local(op,'alpha0',basisEval(0,0,X),X);
    A = A + coef_times_local(op,'alpha1',basisEval(1,1,X),X);
    A = A + coef_times_local(op,'alpha2',basisEval(2,1,X),X);
    A = A + coef_times_local(op,'alpha3',basisEval(1,2,X),X);
    A = A + coef_times_local(op,'alpha4',basisEval(2,2,X),X);
    A = A + coef_times_local(op,'alpha5',basisEval(2,3,X),X);
    if isfield(op,'alpha6')
        A = A + coef_times_local(op,'alpha6',basisEval(3,1,X),X);
    end
end

function A = assemble_operator_from_cache_local(C,X,op)
    A = zeros(size(C.P0));
    A = A + coef_times_local(op,'alpha0',C.P0,X);
    A = A + coef_times_local(op,'alpha1',C.Pux,X);
    A = A + coef_times_local(op,'alpha2',C.Puxx,X);
    A = A + coef_times_local(op,'alpha3',C.Put,X);
    A = A + coef_times_local(op,'alpha4',C.P22,X);
    A = A + coef_times_local(op,'alpha5',C.P23,X);
    if isfield(op,'alpha6')
        A = A + coef_times_local(op,'alpha6',C.Puxxx,X);
    end
end

function A = coef_times_local(op,name,Phi,X)
    if ~isfield(op,name) || isempty(op.(name)), A = zeros(size(Phi)); return; end
    c = op.(name);
    if isa(c,'function_handle'), c = c(X); end
    if isscalar(c), A = c*Phi; else, A = bsxfun(@times,gather(c(:)),Phi); end
end

function [cost,relres] = nonlinear_cost_local(problem,C,X,coef,target,idxV,idxUx,idxUxx,eta)
    u = C.P0*coef;
    ux = C.Pux*coef;
    ut = C.Put*coef;
    uxx = C.Puxx*coef;
    uxxx = C.Puxxx*coef;
    sol = struct('u',u,'ux',ux,'ut',ut,'uxx',uxx,'uxxx',uxxx);
    F = ProblemFunction('residual',X,problem,sol);
    rv = C.Bv*coef-target; rv(~idxV)=0;
    ru = C.Bu*coef-target; ru(~idxUx)=0;
    ruu = C.Buu*coef-target; ruu(~idxUxx)=0;
    cost = norm(F)^2 + eta^2*(norm(rv)^2+norm(ru)^2+norm(ruu)^2);
    relres = norm(F)/(norm(target)+1e-12);
end

function v = getfield_default_local(s,name,default_val)
    if isstruct(s) && isfield(s,name) && ~isempty(s.(name)), v=s.(name); else, v=default_val; end
end
