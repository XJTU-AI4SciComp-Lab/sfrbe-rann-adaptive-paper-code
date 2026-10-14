function out = ProblemFunction(action, X, problem, extra)
%PROBLEMFUNCTION Generalized KdV space-time problem (generic SFRBE interface).
%
%   Coordinates: X(:,1)=x, X(:,2)=t.
%   PDE:  u_t + u_x + u^3 u_x + eps u_xxx = 0,  (x,t) in (-2,3) x (0,2)
%   eps = 0.2058e-4.
%   Exact: u = A sech(K(x-x0)-omega t)^(2/3),  A=0.2275, K=3 sqrt(A^3/(40 eps)),
%          omega = K(1 + A^3/10), x0=0.5.
%   BC/IC: u(x,0)=exact; u(-2,t)=exact; u_x(3,t)=exact_x; u_xx(3,t)=exact_xx.
%
%   Actions (generic + nonlinear):
%     setup/complete/op/exact/ux/uxx/uxxx/ut/boundary/rhs/derivatives
%     residual(X, problem, sol)   : nonlinear residual at X (sol = struct with
%                                   u, ux, ut, uxx, uxxx)
%     jacobian(X, problem, sol)   : linearized operator coefficients
%                                   (struct: ut, ux, uxxx)
%     constraints(problem)        : struct X/types/target for Newton
%     grid(N, problem)            : struct(x_point, x_0)
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
        case {'exact','u','exact_fun','boundary','g','bc'}
            out = exact_derivative_local(X, problem, 0, 0);
        case {'ux','exact_x'}
            out = exact_derivative_local(X, problem, 1, 1);
        case {'uxx','exact_xx'}
            out = exact_derivative_local(X, problem, 2, 1);
        case {'uxxx','exact_xxx'}
            out = exact_derivative_local(X, problem, 3, 1);
        case {'ut','exact_t'}
            out = exact_derivative_local(X, problem, 1, 2);
        case {'initial','ic','u0'}
            out = initial_local(X, problem);
        case {'rhs','right','right_hand','f'}
            out = zeros(size(X,1), 1);
        case {'derivatives','exact_derivatives'}
            out.u   = exact_derivative_local(X, problem, 0, 0);
            out.ux  = exact_derivative_local(X, problem, 1, 1);
            out.uy  = exact_derivative_local(X, problem, 1, 2);
            out.uxx = exact_derivative_local(X, problem, 2, 1);
            out.uyy = zeros(size(X,1), 1);
            out.uxy = zeros(size(X,1), 1);
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
        otherwise
            error('Unknown ProblemFunction action = %s', action);
    end
end

%% ============================================================
function problem = setup_local()
    problem.name = 'Generalized_KdV_space_time';
    problem.type = 'gkdv';
    problem.domain = [-2, 3, 0, 2];   % [x_min, x_max, t_min, t_max]
    problem.A = 0.2275;
    problem.eps_kdv = 0.2058e-4;
    problem.x0 = 0.5;
    problem.K = 3*sqrt(problem.A^3/(40*problem.eps_kdv));
    problem.omega = problem.K*(1 + problem.A^3/10);
    % Linear part used for the initial projection and screening:
    %   L u = u_t + u_x + eps u_xxx
    problem.op.alpha0 = 0;
    problem.op.alpha1 = 1;               % u_x
    problem.op.alpha2 = 0;
    problem.op.alpha3 = 1;               % u_t
    problem.op.alpha4 = 0;
    problem.op.alpha5 = 0;
    problem.op.alpha6 = problem.eps_kdv; % u_xxx
end

function problem = complete_local(problem)
    if ~isstruct(problem), problem = setup_local(); end
    if ~isfield(problem, 'domain'), problem.domain = [-2,3,0,2]; end
    if ~isfield(problem, 'A'), problem.A = 0.2275; end
    if ~isfield(problem, 'eps_kdv'), problem.eps_kdv = 0.2058e-4; end
    if ~isfield(problem, 'x0'), problem.x0 = 0.5; end
    problem.K = 3*sqrt(problem.A^3/(40*problem.eps_kdv));
    problem.omega = problem.K*(1 + problem.A^3/10);
    if ~isfield(problem, 'op')
        problem.op.alpha0 = 0; problem.op.alpha1 = 1; problem.op.alpha2 = 0;
        problem.op.alpha3 = 1; problem.op.alpha4 = 0; problem.op.alpha5 = 0;
        problem.op.alpha6 = problem.eps_kdv;
    end
end

%% ============================================================
function y = exact_derivative_local(X, problem, ord, dir)
    x = X(:,1);
    t = X(:,2);
    A = problem.A; K = problem.K; om = problem.omega; x0 = problem.x0;
    p = 2/3;
    z = K*(x - x0) - om*t;
    S = sech_local(z);
    T = tanh(z);
    f = S.^p;
    if ord == 0
        y = A*f;
    elseif ord == 1
        fz = -p*T.*f;
        if dir == 1
            y = A*K*fz;
        elseif dir == 2
            y = A*(-om)*fz;
        else
            error('exact derivative unsupported dir=%d', dir);
        end
    elseif ord == 2 && dir == 1
        fzz = p*f.*(p*T.^2 - S.^2);
        y = A*K^2*fzz;
    elseif ord == 3 && dir == 1
        fzzz = p*f.*T.*((3*p+2)*S.^2 - p^2*T.^2);
        y = A*K^3*fzzz;
    else
        error('exact derivative unsupported ord=%d, dir=%d', ord, dir);
    end
end

function y = initial_local(X, problem)
    % Initial condition u(x,0): the exact solution evaluated at t = t0.
    X0 = [X(:,1), problem.domain(3)*ones(size(X,1),1)];
    y = exact_derivative_local(X0, problem, 0, 0);
end

function s = sech_local(z)
    s = 1./cosh(z);
end

%% ============================================================
function F = residual_local(X, problem, sol)
    % F = u_t + u_x + u^3 u_x + eps u_xxx
    F = sol.ut + sol.ux + sol.u.^3 .* sol.ux + problem.eps_kdv * sol.uxxx;
end

function N = nonlinear_local(~, ~, sol)
    N = sol.u.^3 .* sol.ux;
end

function Jc = jacobian_local(X, problem, sol)
    % Linearized operator of F = u_t + u_x + u^3 u_x + eps u_xxx:
    %   J(phi) = phi_t + (1+u^3) phi_x + (3 u^2 u_x) phi + eps phi_xxx
    Jc.u    = 3*sol.u.^2 .* sol.ux;          % zero-order term
    Jc.ut   = ones(size(X,1), 1);
    Jc.ux   = 1 + sol.u.^3;
    Jc.uxxx = problem.eps_kdv * ones(size(X,1), 1);
end

function cons = constraints_local(problem)
    dom = problem.domain;
    xmin=dom(1); xmax=dom(2); tmin=dom(3); tmax=dom(4);
    nL = 200; nR = 200; nI = 200;
    tL = linspace(tmin, tmax, nL).';
    xR = linspace(xmin, xmax, nR).';
    xI = linspace(xmin, xmax, nI).';
    XL = [xmin*ones(nL,1), tL];      % u = exact at left
    XR = [xmax*ones(nR,1), tL];      % u_x = exact_x at right
    XR2 = [xmax*ones(nR,1), tL];     % u_xx = exact_xx at right
    XI = [xI, tmin*ones(nI,1)];      % u = exact at t=0
    cons.X = [XL; XR; XR2; XI];
    cons.types = [repmat({'value'}, nL, 1); repmat({'ux'}, nR, 1); ...
                  repmat({'uxx'}, nR, 1); repmat({'value'}, nI, 1)];
    cons.target = [exact_derivative_local(XL, problem, 0, 0); ...
                   exact_derivative_local(XR, problem, 1, 1); ...
                   exact_derivative_local(XR2, problem, 2, 1); ...
                   exact_derivative_local(XI, problem, 0, 0)];
end

function g = grid_local(problem, N)
    dom = problem.domain;
    xmin=dom(1); xmax=dom(2); tmin=dom(3); tmax=dom(4);
    Nx=N(1); Nt=N(2);
    x = linspace(xmin, xmax, Nx).';
    t = linspace(tmin, tmax, Nt).';
    [Xg, Tg] = meshgrid(x, t);
    Xint = [Xg(:), Tg(:)];
    hx = (xmax-xmin)/(Nx-1); ht = (tmax-tmin)/(Nt-1);
    tol = 0.51*max(hx, ht);
    in = Xint(:,1) > xmin+tol & Xint(:,1) < xmax-tol & ...
         Xint(:,2) > tmin+tol & Xint(:,2) < tmax-tol;
    g.x_point = Xint(in, :);
    % boundary/initial points for the linear start (value constraints)
    nL = 200; nR = 200; nI = 200;
    tB = linspace(tmin, tmax, nL).';
    xB = linspace(xmin, xmax, nI).';
    g.x_0 = [xmin*ones(nL,1), tB; xmax*ones(nR,1), tB; xB, tmin*ones(nI,1)];
end
