function out = ProblemFunction(action, X, problem, extra)
%PROBLEMFUNCTION Periodic KdV double-soliton collision (SFRBE interface).
%
%   PDE:  u_t + u u_x + eps u_xxx = 0,  (x,t) in [0,2]x[0,2], periodic in x.
%   eps  = 4.84e-4.
%   IC:   u(x,0) = 3 c1 sech(k1(x-x1))^2 + 3 c2 sech(k2(x-x2))^2,
%         c1=0.3, c2=0.1, x1=0.4, x2=0.8, k_i = 0.5*sqrt(c_i/eps).
%   Reference: Fourier spectral solution interpolated from
%   data/KdV62_reference_100x100.mat (no analytic closed form).

    if nargin < 1 || isempty(action), action = 'setup'; end
    if nargin < 2, X = []; end
    if nargin < 3 || isempty(problem), problem = setup_local(); end
    problem = complete_local(problem);

    switch lower(action)
        case {'setup','problem','config'}
            out = setup_local();
        case {'complete','normalize','validate'}
            out = complete_local(problem);
        case {'operator','op'}
            out = problem.op;
        case {'exact','u','reference','ref'}
            out = reference_derivative_local(X, problem, 0, 0);
        case {'boundary','bc','g'}
            % Periodic KdV has no prescribed Dirichlet boundary trace.  The
            % solver-facing boundary action therefore uses only the known
            % initial data at t=0 and zero elsewhere; the Fourier reference
            % is reserved for a posteriori error evaluation.
            out = solver_boundary_local(X, problem);
        case {'ux','exact_x'}
            out = reference_derivative_local(X, problem, 1, 1);
        case {'uxx','exact_xx'}
            out = reference_derivative_local(X, problem, 2, 1);
        case {'uxxx','exact_xxx'}
            out = reference_derivative_local(X, problem, 3, 1);
        case {'ut','exact_t'}
            out = reference_derivative_local(X, problem, 1, 2);
        case {'initial','ic','u0'}
            out = initial_local(X, problem);
        case {'rhs','right','right_hand','f'}
            out = zeros(size(X,1), 1);
        case 'residual'
            out = residual_local(X, problem, extra);
        case 'nonlinear'
            out = nonlinear_local(X, problem, extra);
        case 'jacobian'
            out = jacobian_local(X, problem, extra);
        case 'constraints'
            out = constraints_local(problem);
        case 'grid'
            out = grid_local(problem, X);   % X = N = [Nx, Nt]
        case 'derivatives'
            out = derivatives_local(X, problem);
        otherwise
            error('Unknown ProblemFunction action = %s', action);
    end
end

%% ============================================================
function problem = setup_local()
    problem = struct();
    problem.name = 'Periodic_KdV_double_soliton';
    problem.type = 'kdv62_periodic';
    problem.domain = [0, 2, 0, 2];   % [x_min, x_max, t_min, t_max]

    problem.eps_kdv = 4.84e-4;
    problem.c1 = 0.3;
    problem.c2 = 0.1;
    problem.x1 = 0.4;
    problem.x2 = 0.8;
    problem.k1 = 0.5*sqrt(problem.c1/problem.eps_kdv);
    problem.k2 = 0.5*sqrt(problem.c2/problem.eps_kdv);

    % Linear part used for the initial projection / screening:
    %   L u = u_t + eps u_xxx   (the u u_x term is fully nonlinear).
    problem.op.alpha0 = 0;
    problem.op.alpha1 = 0;
    problem.op.alpha2 = 0;
    problem.op.alpha3 = 1;                % u_t
    problem.op.alpha4 = 0;
    problem.op.alpha5 = 0;
    problem.op.alpha6 = problem.eps_kdv;  % u_xxx
end

function problem = complete_local(problem)
    if ~isstruct(problem), problem = setup_local(); end
    if ~isfield(problem, 'domain'), problem.domain = [0,2,0,2]; end
    if ~isfield(problem, 'eps_kdv'), problem.eps_kdv = 4.84e-4; end
    if ~isfield(problem, 'c1'), problem.c1 = 0.3; end
    if ~isfield(problem, 'c2'), problem.c2 = 0.1; end
    if ~isfield(problem, 'x1'), problem.x1 = 0.4; end
    if ~isfield(problem, 'x2'), problem.x2 = 0.8; end
    problem.k1 = 0.5*sqrt(problem.c1/problem.eps_kdv);
    problem.k2 = 0.5*sqrt(problem.c2/problem.eps_kdv);
    if ~isfield(problem, 'op')
        problem.op.alpha0 = 0; problem.op.alpha1 = 0; problem.op.alpha2 = 0;
        problem.op.alpha3 = 1; problem.op.alpha4 = 0; problem.op.alpha5 = 0;
        problem.op.alpha6 = problem.eps_kdv;
    end
end

%% ============================================================
function y = initial_local(X, problem)
    x = X(:,1);
    Lx = problem.domain(2) - problem.domain(1);
    x = problem.domain(1) + mod(x - problem.domain(1), Lx);
    y = 3*problem.c1*sech_local(problem.k1*(x - problem.x1)).^2 + ...
        3*problem.c2*sech_local(problem.k2*(x - problem.x2)).^2;
end

function y = solver_boundary_local(X, problem)
    % Compatibility values for legacy direct-value assembly.  The active
    % nonlinear solve uses constraints_local, which imposes periodic
    % differences and the analytic initial condition directly.
    y = zeros(size(X,1),1);
    if isempty(X), return; end
    t0 = problem.domain(3);
    tol = 1e-12 * max(1,problem.domain(4)-problem.domain(3));
    idx0 = abs(X(:,2)-t0) <= tol;
    if any(idx0)
        y(idx0) = initial_local(X(idx0,:),problem);
    end
end

function s = sech_local(z)
    s = 1./cosh(z);
end

%% ============================================================
% residual / jacobian
%% ============================================================
function F = residual_local(X, problem, sol)
    % F = u_t + u u_x + eps u_xxx
    F = sol.ut + sol.u .* sol.ux + problem.eps_kdv * sol.uxxx;
end

function N = nonlinear_local(~, ~, sol)
    N = sol.u .* sol.ux;
end

function Jc = jacobian_local(X, problem, sol)
    % J(v) = v_t + u v_x + u_x v + eps v_xxx
    Jc.u    = sol.ux;                          % zero-order term
    Jc.ut   = ones(size(X,1), 1);
    Jc.ux   = sol.u;
    Jc.uxxx = problem.eps_kdv * ones(size(X,1), 1);
end

%% ============================================================
% boundary / initial constraints: periodic differences plus analytic initial data
%% ============================================================
function cons = constraints_local(problem)
    dom = problem.domain;
    xmin = dom(1); xmax = dom(2); tmin = dom(3); tmax = dom(4);
    nP = 200; nI = 200;
    tB = linspace(tmin, tmax, nP).';
    xB = linspace(xmin, xmax, nI).';
    XL = [xmin*ones(nP,1), tB];      % periodic left
    XR = [xmax*ones(nP,1), tB];      % periodic right
    XI = [xB, tmin*ones(nI,1)];      % initial
    cons.X = [XL; XR; XL; XR; XL; XR; XI];
    cons.types = [repmat({'periodic_u'}, 2*nP, 1); repmat({'periodic_ux'}, 2*nP, 1); ...
                  repmat({'periodic_uxx'}, 2*nP, 1); repmat({'value'}, nI, 1)];
    cons.target = [zeros(6*nP, 1); initial_local(XI, problem)];
    cons.nPair = nP;
end

%% ============================================================
% grid: interior points + boundary/initial points for the linear start
%% ============================================================
function g = grid_local(problem, N)
    dom = problem.domain;
    xmin = dom(1); xmax = dom(2); tmin = dom(3); tmax = dom(4);
    Nx = N(1); Nt = N(2);
    x = linspace(xmin, xmax, Nx).';
    t = linspace(tmin, tmax, Nt).';
    [Xg, Tg] = meshgrid(x, t);
    Xint = [Xg(:), Tg(:)];
    hx = (xmax-xmin)/(Nx-1); ht = (tmax-tmin)/(Nt-1);
    tol = 0.51*max(hx, ht);
    in = Xint(:,1) > xmin+tol & Xint(:,1) < xmax-tol & ...
         Xint(:,2) > tmin+tol & Xint(:,2) < tmax-tol;
    g.x_point = Xint(in, :);
    nL = 200; nR = 200; nI = 200;
    tB = linspace(tmin, tmax, nL).';
    xB = linspace(xmin, xmax, nI).';
    g.x_0 = [xmin*ones(nL,1), tB; xmax*ones(nR,1), tB; xB, tmin*ones(nI,1)];
end

%% ============================================================
% reference interpolation + finite-difference derivatives
%% ============================================================
function y = reference_derivative_local(X, problem, ord, dir)
    [xq, tq] = wrap_time_local(X, problem);

    if ord == 0
        y = reference_interp_local(xq, tq, problem);
        return;
    end

    Lx = problem.domain(2) - problem.domain(1);
    Lt = problem.domain(4) - problem.domain(3);
    hx = 1e-5*Lx;
    ht = 1e-5*Lt;

    if dir == 1  % x-derivatives
        if ord == 1
            xp = wrap_x_local(xq+hx, problem); xm = wrap_x_local(xq-hx, problem);
            y = (reference_interp_local(xp, tq, problem) - reference_interp_local(xm, tq, problem))/(2*hx);
        elseif ord == 2
            xp = wrap_x_local(xq+hx, problem); xm = wrap_x_local(xq-hx, problem);
            y = (reference_interp_local(xp, tq, problem) - 2*reference_interp_local(xq, tq, problem) + reference_interp_local(xm, tq, problem))/(hx^2);
        elseif ord == 3
            xp2 = wrap_x_local(xq+2*hx, problem); xp1 = wrap_x_local(xq+hx, problem);
            xm1 = wrap_x_local(xq-hx, problem);   xm2 = wrap_x_local(xq-2*hx, problem);
            y = (reference_interp_local(xp2, tq, problem) - 2*reference_interp_local(xp1, tq, problem) + ...
                 2*reference_interp_local(xm1, tq, problem) - reference_interp_local(xm2, tq, problem))/(2*hx^3);
        else
            error('unsupported ord=%d', ord);
        end
    elseif dir == 2  % t-derivative
        tp = min(tq+ht, problem.domain(4));
        tm = max(tq-ht, problem.domain(3));
        y = (reference_interp_local(xq, tp, problem) - reference_interp_local(xq, tm, problem))./(tp - tm + eps);
    else
        error('unsupported dir=%d', dir);
    end
end

function [xq, tq] = wrap_time_local(X, problem)
    x = X(:,1); t = X(:,2);
    x0 = problem.domain(1); Lx = problem.domain(2) - problem.domain(1);
    xq = x0 + mod(x - x0, Lx);
    right = abs(x - problem.domain(2)) < 1e-13;
    xq(right) = problem.domain(2);
    tq = min(max(t, problem.domain(3)), problem.domain(4));
end

function xw = wrap_x_local(x, problem)
    x0 = problem.domain(1); Lx = problem.domain(2) - problem.domain(1);
    xw = x0 + mod(x - x0, Lx);
end

function y = reference_interp_local(xq, tq, problem)
    persistent cache
    ref_file = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'data', 'KdV62_reference_100x100.mat');
    if isempty(cache) || ~isfield(cache, 'file') || ~strcmp(cache.file, ref_file)
        S = load(ref_file, 'x_ref', 't_ref', 'u_ref');
        cache.file = ref_file;
        cache.x_ref = S.x_ref(:);
        cache.t_ref = S.t_ref(:);
        cache.u_ref = S.u_ref;
    end
    y = interp2(cache.x_ref, cache.t_ref, cache.u_ref, xq, tq, 'spline');
    bad = ~isfinite(y);
    if any(bad)
        y(bad) = interp2(cache.x_ref, cache.t_ref, cache.u_ref, xq(bad), tq(bad), 'linear');
    end
    y = y(:);
end

%% ============================================================
% derivatives struct for error evaluation
%% ============================================================
function out = derivatives_local(X, problem)
    out.u   = reference_derivative_local(X, problem, 0, 0);
    out.ux  = reference_derivative_local(X, problem, 1, 1);
    out.uy  = reference_derivative_local(X, problem, 1, 2);
    out.uxx = reference_derivative_local(X, problem, 2, 1);
    out.uyy = zeros(size(X,1), 1);
    out.uxy = zeros(size(X,1), 1);
end
