function Phi = BasisFunction3D(basis_type, term, X, ord, dir)
%BASISFUNCTION Analytic basis values/derivatives for 3D space-time input.
%
% X(:,1)=x, X(:,2)=y, X(:,3)=t.
%
% ord:
%   0 value
%   1 first derivative along dir
%   2 second derivative along dir
%   3 third derivative along dir, kept for compatibility
%
% dir:
%   1 x, 2 y, 3 t. For ord=0, dir can be 0.

    if nargin < 4 || isempty(ord), ord = 0; end
    if nargin < 5 || isempty(dir), dir = 0; end

    if size(X,2) ~= 3
        error('BasisFunction3D expects X with 3 columns: [x,y,t].');
    end

    switch lower(basis_type)
        case 'tanh'
            Phi = ridge_dif_local('tanh',term.w,term.b,X,ord,dir);

        case {'ridge_gauss','gauss'}
            Phi = ridge_dif_local('gauss',term.w,term.b,X,ord,dir);

        case 'sin'
            Phi = ridge_dif_local('sin',term.w,term.b,X,ord,dir);

        case 'radial_gauss'
            Phi = radial_gauss_dif_local(term.A,term.r,X,ord,dir);

        case 'mexican_hat'
            Phi = mexican_hat_dif_local(term.A,term.r,X,ord,dir);

        otherwise
            error('Unknown basis_type = %s',basis_type);
    end
end

%% ============================================================
% Ridge functions: phi(s), s = X*w + b
%% ============================================================
function Phi = ridge_dif_local(kind,W,b,X,ord,dir)
    S = X*W + b;

    if ord == 0
        fac = 1;
        nder = 0;
    else
        if dir < 1 || dir > size(W,1)
            error('Invalid derivative dir=%d for 3D ridge basis.',dir);
        end
        fac = W(dir,:).^ord;
        nder = ord;
    end

    Phi = fac .* activation_deriv_local(kind,S,nder);
end

function D = activation_deriv_local(kind,S,n)
    switch lower(kind)
        case 'tanh'
            T = tanh(S);
            Q = 1 - T.^2;
            switch n
                case 0
                    D = T;
                case 1
                    D = Q;
                case 2
                    D = -2*T.*Q;
                case 3
                    D = (6*T.^2 - 2).*Q;
                case 4
                    D = 8*T.*(2 - 3*T.^2).*Q;
                otherwise
                    error('tanh derivative supports n<=4');
            end

        case 'gauss'
            E = exp(-S.^2);
            switch n
                case 0
                    D = E;
                case 1
                    D = -2*S.*E;
                case 2
                    D = (4*S.^2 - 2).*E;
                case 3
                    D = (-8*S.^3 + 12*S).*E;
                case 4
                    D = (16*S.^4 - 48*S.^2 + 12).*E;
                otherwise
                    error('gauss derivative supports n<=4');
            end

        case 'sin'
            switch mod(n,4)
                case 0
                    D = sin(S);
                case 1
                    D = cos(S);
                case 2
                    D = -sin(S);
                case 3
                    D = -cos(S);
            end

        otherwise
            error('Unknown activation kind = %s',kind);
    end
end

%% ============================================================
% Anisotropic radial Gaussian:
%   exp(-sum_i r_i^2 (x_i-a_i)^2)
%% ============================================================
function Phi = radial_gauss_dif_local(A,r,X,ord,dir)
    r = r(:);
    D = shifted_local(X,A);
    S = zeros(size(X,1),size(A,2));
    for kk = 1:numel(r)
        S = S + r(kk)^2 * D{kk}.^2;
    end
    E = exp(-S);

    if ord == 0
        P = ones(size(E));
    else
        if dir < 1 || dir > numel(r), error('Invalid dir=%d',dir); end
        d = D{dir}; rd = r(dir);
        switch ord
            case 1
                P = -2*rd^2*d;
            case 2
                P = 4*rd^4*d.^2 - 2*rd^2;
            case 3
                P = -8*rd^6*d.^3 + 12*rd^4*d;
            otherwise
                error('radial_gauss supports ord<=3');
        end
    end
    Phi = P.*E;
end

%% ============================================================
% Mexican hat:
%   (2-s)*exp(-s/2), s=sum_i r_i^2 (x_i-a_i)^2
%% ============================================================
function Phi = mexican_hat_dif_local(A,r,X,ord,dir)
    r = r(:);
    D = shifted_local(X,A);
    s = zeros(size(X,1),size(A,2));
    for kk = 1:numel(r)
        s = s + r(kk)^2 * D{kk}.^2;
    end
    E = exp(-0.5*s);

    g0 = (2 - s).*E;
    g1 = (0.5*s - 2).*E;
    g2 = (1.5 - 0.25*s).*E;
    g3 = (0.125*s - 1).*E;

    if ord == 0
        Phi = g0;
        return;
    end

    if dir < 1 || dir > numel(r), error('Invalid dir=%d',dir); end
    d = D{dir}; rd = r(dir);
    sj  = 2*rd^2*d;
    sjj = 2*rd^2;

    switch ord
        case 1
            Phi = g1.*sj;
        case 2
            Phi = g2.*sj.^2 + g1.*sjj;
        case 3
            Phi = g3.*sj.^3 + 3*g2.*sj.*sjj;
        otherwise
            error('mexican_hat supports ord<=3');
    end
end

function D = shifted_local(X,A)
    d = size(X,2);
    D = cell(d,1);
    for kk = 1:d
        D{kk} = X(:,kk) - A(kk,:);
    end
end
