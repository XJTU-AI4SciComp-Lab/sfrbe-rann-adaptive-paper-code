function out = DarcyPhiProblemFunction(action,X,problem,normalize_residual,data_name,sigma_default)
%DARCYPHIPROBLEMFUNCTION Fixed smooth Darcy coefficient with two residual forms.
% The PDE is -div(K grad u)=1, K=exp(phi), with the prescribed mixed BCs.

    if nargin < 1 || isempty(action), action = 'setup'; end
    if nargin < 2, X = []; end
    if nargin < 3 || isempty(problem), problem = struct(); end
    if nargin < 4 || isempty(normalize_residual), normalize_residual = false; end
    if nargin < 5 || isempty(data_name), data_name = 'darcy_phi.mat'; end
    if nargin < 6 || isempty(sigma_default), sigma_default = 0.03; end

    switch lower(action)
        case {'setup','problem','config'}
            out = setup_local(normalize_residual,data_name,sigma_default);
            return;
        case {'complete','normalize','validate'}
            out = complete_local(problem,normalize_residual,data_name,sigma_default);
            return;
    end

    p = complete_local(problem,normalize_residual,data_name,sigma_default);
    switch lower(action)
        case {'operator','op'}
            out = p.op;
        case {'rhs','right','right_hand','f'}
            if p.normalize_residual
                out = exp(-phi_local(X(:,1),X(:,2),p.sigma));
            else
                out = ones(size(X,1),1);
            end
        case {'boundary','g','bc'}
            out = boundary_values_local(X);
        case {'exact','u','exact_fun'}
            out = reference_values_local(X,p);
        case {'derivatives','exact_derivatives'}
            [u,ux,uy] = reference_derivatives_local(X,p);
            out = struct('u',u,'ux',ux,'uy',uy);
        case {'eval_grid'}
            out = reference_grid_local(p);
        case {'grid'}
            out = grid_local(X);
        case {'boundary_points'}
            out = boundary_points_local(X);
        case {'neumann','natural_boundary'}
            out = neumann_data_local(p);
        case {'inside'}
            out = all(X >= 0 & X <= 1,2);
        otherwise
            error('Unknown Darcy phi action = %s',action);
    end
end

function p = setup_local(normalize_residual,data_name,sigma)
    [~,p.name] = fileparts(data_name);
    p.domain = [0,1,0,1];
    p.data_name = data_name;
    p.sigma = sigma;
    p.normalize_residual = logical(normalize_residual);
    p = assign_operator_local(p);
end

function p = complete_local(problem,normalize_residual,data_name,sigma_default)
    p = setup_local(normalize_residual,data_name,sigma_default);
    if ~isstruct(problem), error('Darcy phi problem must be a struct.'); end
    names = fieldnames(problem);
    for k = 1:numel(names), p.(names{k}) = problem.(names{k}); end
    if ~isfield(p,'data_name') || isempty(p.data_name), p.data_name = data_name; end
    if ~isfield(p,'sigma') || isempty(p.sigma), p.sigma = sigma_default; end
    if ~isfield(p,'normalize_residual') || isempty(p.normalize_residual)
        p.normalize_residual = logical(normalize_residual);
    end
    p = assign_operator_local(p);
end

function p = assign_operator_local(p)
    p.op.alpha0 = 0;
    if p.normalize_residual
        p.op.alpha1 = @(X) -phi_x_local(X,p.sigma);
        p.op.alpha2 = -1;
        p.op.alpha3 = @(X) -phi_y_local(X,p.sigma);
        p.op.alpha4 = -1;
    else
        p.op.alpha1 = @(X) -coefficient_x_local(X,p.sigma);
        p.op.alpha2 = @(X) -coefficient_local(X,p.sigma);
        p.op.alpha3 = @(X) -coefficient_y_local(X,p.sigma);
        p.op.alpha4 = @(X) -coefficient_local(X,p.sigma);
    end
    p.op.alpha5 = 0;
end

function K = coefficient_local(X,sigma)
    X = gather(X);
    K = exp(phi_local(X(:,1),X(:,2),sigma));
end

function [Kx,Ky] = coefficient_derivatives_local(X,sigma)
    X = gather(X);
    K = coefficient_local(X,sigma);
    [px,py] = phi_derivatives_local(X,sigma);
    Kx = K.*px; Ky = K.*py;
end

function Kx = coefficient_x_local(X,sigma), [Kx,~] = coefficient_derivatives_local(X,sigma); end
function Ky = coefficient_y_local(X,sigma), [~,Ky] = coefficient_derivatives_local(X,sigma); end

function [px,py] = phi_derivatives_local(X,sigma)
    x = X(:,1); y = X(:,2);
    g = exp(-((x-0.7).^2+(y-0.3).^2)/(sigma^2));
    px = 20*pi*cos(4*pi*x).*sin(4*pi*y) ...
        + 25*pi*cos(10*pi*x).*sin(6*pi*y) ...
        - (14/sigma^2)*(x-0.7).*g;
    py = 20*pi*sin(4*pi*x).*cos(4*pi*y) ...
        + 15*pi*sin(10*pi*x).*cos(6*pi*y) ...
        - (14/sigma^2)*(y-0.3).*g;
end

function px = phi_x_local(X,sigma)
    [px,~] = phi_derivatives_local(X,sigma);
end

function py = phi_y_local(X,sigma)
    [~,py] = phi_derivatives_local(X,sigma);
end

function phi = phi_local(x,y,sigma)
    g = exp(-((x-0.7).^2+(y-0.3).^2)/(sigma^2));
    phi = 5*sin(4*pi*x).*sin(4*pi*y) ...
        + 2.5*sin(10*pi*x).*sin(6*pi*y) + 7*g;
end

function g = boundary_values_local(X)
    X = gather(X); g = zeros(size(X,1),1);
    g(abs(X(:,1)) <= 1e-10) = 1;
    g(abs(X(:,1)-1) <= 1e-10) = 0;
end

function out = grid_local(N)
    N = max(2,round(double(N(:).'))); nx = N(1); ny = N(min(2,numel(N)));
    x = linspace(0,1,nx+2); x = x(2:end-1); y = linspace(0,1,ny+2); y = y(2:end-1);
    [X,Y] = meshgrid(x,y); out.x_point = [X(:),Y(:)]; out.x_0 = boundary_points_local(max(2,ny+2));
end

function Xb = boundary_points_local(n)
    n = max(2,round(double(n(1)))); y = linspace(0,1,n).';
    Xb = [zeros(n,1),y; ones(n,1),y];
end

function data = neumann_data_local(p)
    n = 100; t = linspace(0,1,n+2).'; t = t(2:end-1);
    data.X = [t,zeros(size(t)); t,ones(size(t))];
    data.normal = [repmat([0,-1],numel(t),1); repmat([0,1],numel(t),1)];
    data.coefficient = coefficient_local(data.X,p.sigma); data.target = zeros(size(data.X,1),1);
end

function data = load_reference_local(p)
    persistent cache
    if isempty(cache), cache = struct(); end
    key = matlab.lang.makeValidName(p.data_name);
    if isfield(cache,key), data = cache.(key); return; end
    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    file_name = fullfile(root,'data',p.data_name);
    if exist(file_name,'file') ~= 2, error('Reference file not found: %s',file_name); end
    data = load(file_name,'x','y','u'); cache.(key) = data;
end

function val = reference_values_local(X,p)
    data = load_reference_local(p); [ix,iy] = reference_indices_local(X,data);
    val = data.u(sub2ind(size(data.u),iy,ix));
end

function [u,ux,uy] = reference_derivatives_local(X,p)
    data = load_reference_local(p); [ix,iy] = reference_indices_local(X,data);
    dx = data.x(2)-data.x(1); dy = data.y(2)-data.y(1);
    [ux_grid,uy_grid] = gradient(data.u,dx,dy); ind = sub2ind(size(data.u),iy,ix);
    u = data.u(ind); ux = ux_grid(ind); uy = uy_grid(ind);
end

function [ix,iy] = reference_indices_local(X,data)
    X = gather(X); dx = data.x(2)-data.x(1); dy = data.y(2)-data.y(1);
    ix = round((X(:,1)-data.x(1))/dx)+1; iy = round((X(:,2)-data.y(1))/dy)+1;
    if any(ix<1|ix>numel(data.x)|iy<1|iy>numel(data.y)), error('Point outside reference grid.'); end
    if any(abs(X(:,1)-data.x(ix))>5e-10|abs(X(:,2)-data.y(iy))>5e-10)
        error('Reference lookup requires points on saved grid; no interpolation is used.');
    end
end

function out = reference_grid_local(p)
    data = load_reference_local(p); [X,Y] = meshgrid(data.x,data.y);
    out.x=data.x; out.y=data.y; out.X=[X(:),Y(:)]; out.mask=true(size(X));
end
