function out = ProblemFunction(action, X, problem)
%PROBLEMFUNCTION L-shaped singular Poisson problem (generic SFRBE interface).
%
%   Domain:  Omega = [-1,1]^2 \ ([0,1] x [-1,0])
%   Exact:   u = r^p sin(k*theta),  p = k = 2/3,  corner at (0,0)
%   PDE:     -Delta u = 0   (f = 0 since p = k)
%   Boundary: Dirichlet from the exact solution.
%
%   Supported actions (generic pipeline):
%     setup / complete / op / exact / boundary / rhs / derivatives
%     grid(N)            -> struct(x_point, x_0)          (non-rectangular support)
%     boundary_points(b) -> boundary point set
%     eval_grid(pp)      -> struct(x, y, Xfull, mask, X)  (masked eval grid)
%     inside(X)          -> logical mask of the L-shape domain
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
        case {'exact','u','exact_fun'}
            out = exact_local(X, problem);
        case {'boundary','g','bc'}
            out = exact_local(X, problem);
        case {'rhs','right','right_hand','f'}
            out = rhs_local(X, problem);
        case {'derivatives','exact_derivatives'}
            [u, ux, uy] = derivatives_local(X, problem);
            out.u = u; out.ux = ux; out.uy = uy;
            out.uxx = zeros(size(u)); out.uyy = zeros(size(u)); out.uxy = zeros(size(u));
        case 'grid'
            out = grid_local(problem, X);   % X = N = [Nx, Ny]
        case 'boundary_points'
            out = boundary_points_local(problem, X);  % X = b_point
        case 'eval_grid'
            out = eval_grid_local(problem, X);        % X = pp
        case 'inside'
            out = inside_lshape_local(X, problem);
        otherwise
            error('Unknown ProblemFunction action = %s', action);
    end
end

%% ============================================================
function problem = setup_local()
    problem.name = 'L-shape singular Poisson: u = r^{2/3} sin(2 theta/3)';
    problem.domain = [-1, 1, -1, 1];
    problem.p_power = 2/3;
    problem.theta_coef = 2/3;
    problem.corner = [0, 0];
    problem.eps0 = 1e-14;
    problem.op.alpha0 = 0; problem.op.alpha1 = 0; problem.op.alpha2 = -1;
    problem.op.alpha3 = 0; problem.op.alpha4 = -1; problem.op.alpha5 = 0;
end

function problem = complete_local(problem)
    if ~isstruct(problem), problem = setup_local(); end
    if ~isfield(problem, 'domain'), problem.domain = [-1,1,-1,1]; end
    if ~isfield(problem, 'p_power'), problem.p_power = 2/3; end
    if ~isfield(problem, 'theta_coef'), problem.theta_coef = 2/3; end
    if ~isfield(problem, 'corner'), problem.corner = [0,0]; end
    if ~isfield(problem, 'eps0'), problem.eps0 = 1e-14; end
    if ~isfield(problem, 'op')
        problem.op.alpha0 = 0; problem.op.alpha1 = 0; problem.op.alpha2 = -1;
        problem.op.alpha3 = 0; problem.op.alpha4 = -1; problem.op.alpha5 = 0;
    end
end

%% ============================================================
function [r, theta] = polar_local(X, problem)
    x = X(:,1) - problem.corner(1);
    y = X(:,2) - problem.corner(2);
    r = sqrt(x.^2 + y.^2);
    theta = atan2(y, x);
    theta(theta < 0) = theta(theta < 0) + 2*pi;
end

function u = exact_local(X, problem)
    [r, theta] = polar_local(X, problem);
    u = r.^problem.p_power .* sin(problem.theta_coef * theta);
    u(r < problem.eps0) = 0;
end

function f = rhs_local(X, problem)
    % p = k = 2/3 -> -Delta u = 0 everywhere in the L-shape.
    f = zeros(size(X,1), 1);
end

function [u, ux, uy] = derivatives_local(X, problem)
    x = X(:,1) - problem.corner(1);
    y = X(:,2) - problem.corner(2);
    [r, theta] = polar_local(X, problem);
    p = problem.p_power; k = problem.theta_coef;
    u  = r.^p .* sin(k*theta);
    rp = max(r, problem.eps0);
    ux = p*rp.^(p-2) .* x .* sin(k*theta) - k*rp.^(p-2) .* y .* cos(k*theta);
    uy = p*rp.^(p-2) .* y .* sin(k*theta) + k*rp.^(p-2) .* x .* cos(k*theta);
    u(r < problem.eps0) = 0; ux(r < problem.eps0) = 0; uy(r < problem.eps0) = 0;
    ux(~isfinite(ux)) = 0; uy(~isfinite(uy)) = 0;
end

%% ============================================================
function X = interior_grid_local(problem, N)
    dom = problem.domain;
    xmin=dom(1); xmax=dom(2); ymin=dom(3); ymax=dom(4);
    Nx=N(1); Ny=N(2);
    x = linspace(xmin,xmax,Nx).';
    y = linspace(ymin,ymax,Ny).';
    [Xg,Yg] = meshgrid(x,y);
    P = [Xg(:),Yg(:)];
    hx = (xmax-xmin)/(Nx-1); hy = (ymax-ymin)/(Ny-1);
    tol = 0.51*max(hx,hy);
    xx=P(:,1); yy=P(:,2);
    inside_outer = xx > xmin+tol & xx < xmax-tol & yy > ymin+tol & yy < ymax-tol;
    inside_l = inside_outer & ~((xx >= -tol) & (yy <= tol));
    rr = sqrt((xx-problem.corner(1)).^2 + (yy-problem.corner(2)).^2);
    X = P(inside_l & rr > tol,:);
end

function X = boundary_points_local(problem, b_point)
    dom = problem.domain;
    xmin=dom(1); xmax=dom(2); ymin=dom(3); ymax=dom(4);
    x1 = linspace(xmin,0,b_point).'; y1 = ymin*ones(b_point,1);
    y2 = linspace(ymin,0,b_point).'; x2 = zeros(b_point,1);
    x3 = linspace(0,xmax,b_point).'; y3 = zeros(b_point,1);
    y4 = linspace(0,ymax,b_point).'; x4 = xmax*ones(b_point,1);
    x5 = linspace(xmax,xmin,b_point).'; y5 = ymax*ones(b_point,1);
    y6 = linspace(ymax,ymin,b_point).'; x6 = xmin*ones(b_point,1);
    X = [x1,y1; x2,y2; x3,y3; x4,y4; x5,y5; x6,y6];
end

function grid = eval_grid_local(problem, pp)
    dom = problem.domain;
    xmin=dom(1); xmax=dom(2); ymin=dom(3); ymax=dom(4);
    Nx = pp; Ny = pp;
    x = linspace(xmin,xmax,pp).';
    y = linspace(ymin,ymax,pp).';
    [Xg,Yg] = meshgrid(x,y);
    P = [Xg(:),Yg(:)];
    mask = reshape(inside_lshape_local(P, problem), Ny, Nx);
    grid.x = x; grid.y = y;
    grid.Xfull = P;
    grid.mask = mask;
    grid.X = P(mask,:);
    grid.Nx = pp; grid.Ny = pp;
end

function in = inside_lshape_local(X, problem)
    % Omega = [-1,1]^2 \ ([0,1] x [-1,0]), with a small tolerance on boundary.
    dom = problem.domain;
    xmin=dom(1); xmax=dom(2); ymin=dom(3); ymax=dom(4);
    xx = X(:,1); yy = X(:,2);
    in = (xx >= xmin) & (xx <= xmax) & (yy >= ymin) & (yy <= ymax) & ...
         ~((xx >= 0) & (yy <= 0));
end

function g = grid_local(problem, N)
    g = struct();
    g.x_point = interior_grid_local(problem, N);
    g.x_0 = boundary_points_local(problem, 250);
end