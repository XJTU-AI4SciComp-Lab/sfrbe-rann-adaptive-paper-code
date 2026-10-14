function out = ProblemFunction3D(action,X,problem)
%PROBLEMFUNCTION 2D viscous Burgers equation in 3D space-time input.
%
% Coordinates:
%   X(:,1)=x, X(:,2)=y, X(:,3)=t
%
% PDE:
%   u_t + u*(u_x + u_y) - eps*(u_xx + u_yy) = 0,
%   (x,y) in (0,1)^2, t in (0,1].
%
% Exact solution:
%   u = 1/(1 + exp((x+y-t)/(2 eps))).
%
% Boundary/initial constraints:
%   u = exact on boundary x=0, x=1, y=0, y=1;
%   u(x,y,0) = exact(x,y,0).

    if nargin < 1 || isempty(action), action = 'setup'; end
    if nargin < 2, X = []; end

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
        case {'exact','u'}
            out = exact_local(X,problem);

        case {'initial','ic','h0'}
            X0 = X;
            if size(X0,2) == 2
                X0 = [X0, zeros(size(X0,1),1)];
            else
                X0(:,3) = problem.t0;
            end
            out = exact_local(X0,problem);

        case {'boundary','g'}
            out = exact_local(X,problem);

        case {'rhs','f'}
            out = zeros(size(X,1),1);

        case 'ux'
            [out,~,~,~,~] = derivatives_local(X,problem);

        case 'uy'
            [~,out,~,~,~] = derivatives_local(X,problem);

        case 'ut'
            [~,~,out,~,~] = derivatives_local(X,problem);

        case 'uxx'
            [~,~,~,out,~] = derivatives_local(X,problem);

        case 'uyy'
            [~,~,~,~,out] = derivatives_local(X,problem);

        otherwise
            error('Unknown ProblemFunction3D action = %s',action);
    end
end

function problem = setup_problem_local()
    problem = struct();
    problem.name = 'Burgers2D_space_time_eps001';
    problem.eps_burgers = 1e-2;
    problem.epsilon = problem.eps_burgers;
    problem.domain = [0,1,0,1,0,1]; % [x0,x1,y0,y1,t0,t1]
    problem.x0 = 0; problem.x1 = 1;
    problem.y0 = 0; problem.y1 = 1;
    problem.t0 = 0; problem.t1 = 1;
    problem.dim = 3;

    % Kept for compatibility with older linear-operator utilities.
    problem.op = struct();
    problem.op.name = 'burgers2d_nonlinear';
end

function problem = complete_problem_local(problem)
    if ~isfield(problem,'eps_burgers')
        if isfield(problem,'epsilon')
            problem.eps_burgers = problem.epsilon;
        else
            problem.eps_burgers = 1e-2;
        end
    end
    problem.epsilon = problem.eps_burgers;

    if ~isfield(problem,'domain') || isempty(problem.domain)
        problem.domain = [0,1,0,1,0,1];
    end

    dom = problem.domain(:).';
    problem.x0 = dom(1); problem.x1 = dom(2);
    problem.y0 = dom(3); problem.y1 = dom(4);
    problem.t0 = dom(5); problem.t1 = dom(6);
    problem.dim = 3;

    if ~isfield(problem,'op'), problem.op = struct(); end
    problem.op.name = 'burgers2d_nonlinear';
end

function u = exact_local(X,problem)
    X = normalize_X_local(X);
    epsv = problem.eps_burgers;
    z = (X(:,1) + X(:,2) - X(:,3)) ./ (2*epsv);

    % Stable logistic 1/(1+exp(z)).
    u = zeros(size(z));
    pos = z >= 0;
    ez = exp(-z(pos));
    u(pos) = ez ./ (1 + ez);
    ez = exp(z(~pos));
    u(~pos) = 1 ./ (1 + ez);
end

function [ux,uy,ut,uxx,uyy] = derivatives_local(X,problem)
    X = normalize_X_local(X);
    epsv = problem.eps_burgers;
    u = exact_local(X,problem);

    a = 1/(2*epsv);
    us  = -u.*(1-u);
    uss =  u.*(1-u).*(1-2*u);

    ux  =  a * us;
    uy  =  a * us;
    ut  = -a * us;
    uxx =  a^2 * uss;
    uyy =  a^2 * uss;
end

function X = normalize_X_local(X)
    if isempty(X)
        X = zeros(0,3);
        return;
    end
    if size(X,2) == 2
        X = [X, zeros(size(X,1),1)];
    end
    if size(X,2) ~= 3
        error('ProblemFunction3D expects X with 3 columns: [x,y,t].');
    end
end
