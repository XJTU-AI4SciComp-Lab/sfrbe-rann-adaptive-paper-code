function out = ProblemFunction(action,X,problem)
%PROBLEMFUNCTION unified definition of a 2D problem: operator, exact solution, boundary values, right-hand side.
%
% Current version: 
%   the original double-Gaussian Poisson problem
%
% PDE:
%   -Delta u = f
%
% exact:
%   u(x,y) = A1 exp(-|x-c1|^2 / eps1^2)
%          + A2 exp(-|x-c2|^2 / eps2^2)
%
% generic 2D linear operator:
%   L u = alpha0*u + alpha1*u_x + alpha2*u_xx
%       + alpha3*u_y + alpha4*u_yy + alpha5*u_xy
%
% Poisson:
%   alpha0 = 0
%   alpha2 = -1
%   alpha4 = -1

    if nargin < 1 || isempty(action)
        action = 'setup';
    end

    if nargin < 2
        X = [];
    end

    switch lower(action)
        case {'setup','problem','config'}
            out = setup_problem_local();
            return;

        case {'complete','normalize','validate'}
            if nargin < 3 || isempty(problem)
                problem = setup_problem_local();
            end
            out = complete_problem_local(problem);
            return;
    end

    if nargin < 3 || isempty(problem)
        problem = setup_problem_local();
    end
    problem = complete_problem_local(problem);

    switch lower(action)
        case {'operator','op'}
            out = problem.op;

        case {'exact','u','exact_fun'}
            out = exact_solution_local(X,problem);

        case {'boundary','g','bc'}
            out = boundary_value_local(X,problem);

        case {'rhs','right','right_hand','f'}
            out = rhs_local(X,problem);

        case {'derivatives','exact_derivatives'}
            [u,ux,uy,uxx,uyy,uxy] = exact_derivatives_local(X,problem);
            out.u = u;
            out.ux = ux;
            out.uy = uy;
            out.uxx = uxx;
            out.uyy = uyy;
            out.uxy = uxy;

        otherwise
            error('Unknown ProblemFunction action = %s', action);
    end
end

%% ============================================================
% Modify here: problem name, domain, operator, Gaussian parameters
%% ============================================================
function problem = setup_problem_local()

    problem.name = 'double_gaussian_poisson_2d';

    % Domain [a1,a2] x [a3,a4]
    problem.domain = [0,1,0,1];

    % ------------------------------------------------------------
    % Poisson:
    %   L u = -u_xx - u_yy
    % ------------------------------------------------------------
    problem.op.alpha0 = 0;
    problem.op.alpha1 = 0;
    problem.op.alpha2 = -1;
    problem.op.alpha3 = 0;
    problem.op.alpha4 = -1;
    problem.op.alpha5 = 0;

    % ------------------------------------------------------------
    % double Gaussian exact-solution parameters
    %
    % u = A1 exp(-r1sq/eps1^2) + A2 exp(-r2sq/eps2^2)
    % ------------------------------------------------------------

    % original centers
    problem.centers = [
        0.30, 0.35;
        0.72, 0.68
    ];

    % original amplitudes
    problem.amplitudes = [
        1.00;
        0.85
    ];

    % original scales. eps_base corresponds to epslion in your legacy code.
    % If you previously used k = 0.01, then:
    %   eps1 = 2*k = 0.02
    %   eps2 = 5*k = 0.05
    problem.eps_base = 0.01;
    problem.eps_list = [
        2.0 * problem.eps_base;
        5.0 * problem.eps_base
    ];

    % RHS = L u_exact
    problem.rhs_case = 'linear_operator_on_gaussian_exact';
end

%% ============================================================
% complete and validate problem
%% ============================================================
function problem = complete_problem_local(problem)

    if ~isstruct(problem)
        error('ProblemFunction third input must be a complete problem structure. Use: problem = ProblemFunction(''setup'');');
    end

    op_names = {'alpha0','alpha1','alpha2','alpha3','alpha4','alpha5'};
    is_op_only = true;
    for j = 1:numel(op_names)
        is_op_only = is_op_only && isfield(problem,op_names{j});
    end

    if is_op_only && ~isfield(problem,'op')
        error(['You passed an op structure, not a complete problem structure.\n', ...
               'Correct usage:\n', ...
               '  problem = ProblemFunction(''setup'');\n', ...
               '  op = problem.op;\n', ...
               '  f = ProblemFunction(''rhs'',X,problem);']);
    end

    base = setup_problem_local();
    problem = merge_struct_local(base,problem);

    if ~isfield(problem,'op') || isempty(problem.op)
        error('problem.op does not exist. Define it in ProblemFunction(''setup'').');
    end

    validate_op_local(problem.op);

    if ~isfield(problem,'domain') || numel(problem.domain) ~= 4
        error('problem.domain must be [a1,a2,a3,a4].');
    end

    if ~isfield(problem,'rhs_case') || isempty(problem.rhs_case)
        error('problem.rhs_case must not be empty.');
    end

    if ~isfield(problem,'centers') || isempty(problem.centers)
        error('problem.centers does not exist. The double Gaussian needs centers.');
    end

    if size(problem.centers,2) ~= 2
        error('problem.centers must be a K-by-2 matrix, each row [cx,cy].');
    end

    K = size(problem.centers,1);

    if ~isfield(problem,'amplitudes') || numel(problem.amplitudes) ~= K
        error('problem.amplitudes must be K-by-1, with length matching the number of centers rows.');
    end

    if ~isfield(problem,'eps_list') || numel(problem.eps_list) ~= K
        error('problem.eps_list must be K-by-1, with length matching the number of centers rows.');
    end

    problem.amplitudes = problem.amplitudes(:);
    problem.eps_list = problem.eps_list(:);
end

function s = merge_struct_local(base,override)
    s = base;
    names = fieldnames(override);
    for i = 1:numel(names)
        name = names{i};
        if isstruct(override.(name)) && isfield(base,name) && isstruct(base.(name))
            s.(name) = merge_struct_local(base.(name),override.(name));
        else
            s.(name) = override.(name);
        end
    end
end

function validate_op_local(op)
    names = {'alpha0','alpha1','alpha2','alpha3','alpha4','alpha5'};
    for j = 1:numel(names)
        if ~isfield(op,names{j})
            error('problem.op is missing field %s. Fully define alpha0...alpha5.',names{j});
        end
    end
end

%% ============================================================
% exact / boundary / rhs
%% ============================================================
function u = exact_solution_local(X,problem)
    [u,~,~,~,~,~] = exact_derivatives_local(X,problem);
end

function g = boundary_value_local(X,problem)
    % Dirichlet boundary uses the exact solution directly
    g = exact_solution_local(X,problem);
end

function f = rhs_local(X,problem)

    [u,ux,uy,uxx,uyy,uxy] = exact_derivatives_local(X,problem);

    switch lower(problem.rhs_case)
        case 'linear_operator_on_gaussian_exact'
            f = apply_operator_to_exact_values_local(X,u,ux,uxx,uy,uyy,uxy,problem.op);

        otherwise
            error('Unknown problem.rhs_case = %s', problem.rhs_case);
    end
end

%% ============================================================
% double Gaussian exact solution and its derivatives
%% ============================================================
function [u,ux,uy,uxx,uyy,uxy] = exact_derivatives_local(X,problem)

    x = X(:,1);
    y = X(:,2);

    centers = problem.centers;
    amps = problem.amplitudes(:);
    eps_list = problem.eps_list(:);

    u   = zeros(size(x));
    ux  = zeros(size(x));
    uy  = zeros(size(x));
    uxx = zeros(size(x));
    uyy = zeros(size(x));
    uxy = zeros(size(x));

    for j = 1:size(centers,1)

        cx = centers(j,1);
        cy = centers(j,2);
        A  = amps(j);
        ep = eps_list(j);

        dx = x - cx;
        dy = y - cy;

        ep2 = ep^2;
        ep4 = ep^4;

        r2 = dx.^2 + dy.^2;

        G = A * exp(-r2 / ep2);

        % first derivative
        Gx = G .* (-2 * dx / ep2);
        Gy = G .* (-2 * dy / ep2);

        % second derivative
        Gxx = G .* (4 * dx.^2 / ep4 - 2 / ep2);
        Gyy = G .* (4 * dy.^2 / ep4 - 2 / ep2);

        % mixed derivative
        Gxy = G .* (4 * dx .* dy / ep4);

        u   = u   + G;
        ux  = ux  + Gx;
        uy  = uy  + Gy;
        uxx = uxx + Gxx;
        uyy = uyy + Gyy;
        uxy = uxy + Gxy;
    end
end

%% ============================================================
% apply problem.op to the exact derivative values
%% ============================================================
function f = apply_operator_to_exact_values_local(X,u,ux,uxx,uy,uyy,uxy,op)

    N = size(X,1);

    a0 = coef_to_N_local(op.alpha0,X,N);
    a1 = coef_to_N_local(op.alpha1,X,N);
    a2 = coef_to_N_local(op.alpha2,X,N);
    a3 = coef_to_N_local(op.alpha3,X,N);
    a4 = coef_to_N_local(op.alpha4,X,N);
    a5 = coef_to_N_local(op.alpha5,X,N);

    f = a0.*u + a1.*ux + a2.*uxx + a3.*uy + a4.*uyy + a5.*uxy;
end

function c = coef_to_N_local(a,X,N)

    if isa(a,'function_handle')
        a = a(X);
    end

    if isempty(a)
        c = zeros(N,1);
    elseif isscalar(a)
        c = a * ones(N,1);
    else
        c = a(:);
        if length(c) ~= N
            error('operator coefficient length should be N=%d, but got %d.',N,length(c));
        end
    end
end