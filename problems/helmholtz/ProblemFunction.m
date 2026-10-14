function out = ProblemFunction(action, X, problem)
%PROBLEMFUNCTION Multiscale Helmholtz problem (generic SFRBE interface).
%
%   PDE:    -Delta u - kappa^2 u = f   on (0,1)^2
%   Exact:  u = sin(3 pi x) sin(2 pi y)
%            + 0.5 sin(25 pi x) sin(22 pi y)
%            + 0.2 sin(50 pi x) sin(57 pi y)
%   RHS:    f = L u_exact
%   Boundary: Dirichlet from the exact solution.
%
%   Actions: setup / complete / op / exact / boundary / rhs / derivatives.
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
            out = exact_solution_local(X, problem);
        case {'boundary','g','bc'}
            out = exact_solution_local(X, problem);
        case {'rhs','right','right_hand','f'}
            out = rhs_local(X, problem);
        case {'derivatives','exact_derivatives'}
            [u, ux, uy, uxx, uyy, uxy] = exact_derivatives_local(X, problem);
            out.u = u; out.ux = ux; out.uy = uy;
            out.uxx = uxx; out.uyy = uyy; out.uxy = uxy;
        otherwise
            error('Unknown ProblemFunction action = %s', action);
    end
end

%% ============================================================
function problem = setup_local()
    problem.name = 'linear_multiscale_sine_2d';
    problem.domain = [0, 1, 0, 1];
    problem.kappa = 1;
    problem.op.alpha0 = -problem.kappa^2;
    problem.op.alpha1 = 0;
    problem.op.alpha2 = -1;
    problem.op.alpha3 = 0;
    problem.op.alpha4 = -1;
    problem.op.alpha5 = 0;
    % modes: [mx, my, amplitude]
    problem.modes = [
         3,   2, 1.00;
        25,  22, 0.50;
        50,  57, 0.20
    ];
    problem.rhs_case = 'linear_operator_on_sine_exact';
end

function problem = complete_local(problem)
    if ~isstruct(problem), problem = setup_local(); end
    if ~isfield(problem, 'domain'), problem.domain = [0,1,0,1]; end
    if ~isfield(problem, 'kappa'), problem.kappa = 1; end
    if ~isfield(problem, 'op')
        problem.op.alpha0 = -problem.kappa^2;
        problem.op.alpha1 = 0; problem.op.alpha2 = -1;
        problem.op.alpha3 = 0; problem.op.alpha4 = -1; problem.op.alpha5 = 0;
    end
    if ~isfield(problem, 'modes')
        problem.modes = [3,2,1.00; 25,22,0.50; 50,57,0.20];
    end
    if ~isfield(problem, 'rhs_case'), problem.rhs_case = 'linear_operator_on_sine_exact'; end
end

%% ============================================================
function [u, ux, uy, uxx, uyy, uxy] = exact_derivatives_local(X, problem)
    x = X(:,1);
    y = X(:,2);
    modes = problem.modes;
    u = zeros(size(x)); ux = u; uy = u; uxx = u; uyy = u; uxy = u;
    for j = 1:size(modes,1)
        mx = modes(j,1); my = modes(j,2); A = modes(j,3);
        sx = sin(mx*pi*x); cx = cos(mx*pi*x);
        sy = sin(my*pi*y); cy = cos(my*pi*y);
        kx = mx*pi; ky = my*pi;
        u   = u   + A * sx .* sy;
        ux  = ux  + A * kx * cx .* sy;
        uy  = uy  + A * ky * sx .* cy;
        uxx = uxx - A * kx^2 * sx .* sy;
        uyy = uyy - A * ky^2 * sx .* sy;
        uxy = uxy + A * kx * ky * cx .* cy;
    end
end

function u = exact_solution_local(X, problem)
    [u, ~, ~, ~, ~, ~] = exact_derivatives_local(X, problem);
end

function f = rhs_local(X, problem)
    [u, ux, uy, uxx, uyy, uxy] = exact_derivatives_local(X, problem);
    op = problem.op;
    f = op.alpha0.*u + op.alpha1.*ux + op.alpha2.*uxx + ...
        op.alpha3.*uy + op.alpha4.*uyy + op.alpha5.*uxy;
end