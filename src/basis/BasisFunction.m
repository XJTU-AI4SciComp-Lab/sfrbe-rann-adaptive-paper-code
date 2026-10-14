function Phi = BasisFunction(basis_type, term, X, ord, dir)
%BASISFUNCTION all basis function values and ordinary derivatives in one file.
%
% basis_type:
%   'tanh'
%   'ridge_gauss' or 'gauss'
%   'radial_gauss'
%   'sin'
%   'mexican_hat'

    switch lower(basis_type)
        case 'tanh'
            Phi = tanh_dif_local(term.w,term.b,X,ord,dir);

        case {'ridge_gauss','gauss'}
            Phi = gauss_dif_local(term.w,term.b,X,ord,dir);

        case 'radial_gauss'
            Phi = radial_gauss_dif_local(term.A,term.r,X,ord,dir);

        case 'sin'
            Phi = sin_dif_local(term.w,term.b,X,ord,dir);

        case 'mexican_hat'
            Phi = mexican_hat_dif_local(term.A,term.r,X,ord,dir);

        otherwise
            error('Unknown basis_type = %s', basis_type);
    end

    Phi = gather(Phi);
end

function out = tanh_dif_local(W,b,X,ord,dim)
    if isa(W,'gpuArray') || isa(b,'gpuArray')
        X = gpuArray(X);
    end

    if iscolumn(b)
        b = b.';
    end

    Z = X*W + b;
    T = tanh(Z);
    S2 = 1 - T.^2;

    if ord == 0
        out = T;
        return;
    end

    if ord == 1
        if dim <= 0 || dim > size(W,1)
            error('For ord=1, dim must be between 1 and size(W,1).');
        end
        wj = W(dim,:);
        out = S2 .* wj;
    elseif ord == 2
        if dim == 1 || dim == 2
            wj = W(dim,:);
            out = (-2) .* T .* S2 .* (wj.^2);
        elseif dim == 3
            out = (-2) .* T .* S2 .* (W(1,:).*W(2,:));
        else
            error('tanh does not support ord=%d, dim=%d.',ord,dim);
        end
    elseif ord == 3 && dim == 1
        wj = W(1,:);
        out = (-2) .* (wj.^3) .* S2 .* (1 - 3*T.^2);
    else
        error('tanh only supports ord=0,1,2,3 (ord=3 needs dim=1).');
    end
end

function out = gauss_dif_local(w,b,x,ord,dim)
    if isa(w,'gpuArray') || isa(b,'gpuArray')
        x = gpuArray(x);
    end

    xin = x*w + b;

    if ord == 0
        out = exp(-xin.*xin);

    elseif dim == 1
        if ord == 1
            out = -2 .* w(1,:) .* xin .* exp(-xin.*xin);
        elseif ord == 2
            out = (4.*xin.*xin - 2) .* (w(1,:).^2) .* exp(-xin.*xin);
        elseif ord == 3
            out = (12.*xin - 8.*xin.^3) .* (w(1,:).^3) .* exp(-xin.*xin);
        else
            error('gauss only supports ord=0,1,2,3.');
        end

    elseif dim == 2
        if ord == 1
            out = -2 .* w(2,:) .* xin .* exp(-xin.*xin);
        elseif ord == 2
            out = (4.*xin.*xin - 2) .* (w(2,:).^2) .* exp(-xin.*xin);
        else
            error('gauss only supports ord=0,1,2.');
        end

    elseif dim == 3 && ord == 2
        out = (4.*xin.*xin - 2) .* (w(1,:).*w(2,:)) .* exp(-xin.*xin);

    else
        error('gauss does not support ord=%d, dim=%d.', ord, dim);
    end
end

function Phi = radial_gauss_dif_local(A_center,r,X,ord,dir)
    X = gather(X);
    A_center = gather(A_center);
    r = gather(r(:));

    x1 = X(:,1);
    x2 = X(:,2);

    a1 = A_center(1,:);
    a2 = A_center(2,:);

    dx = bsxfun(@minus,x1,a1);
    dy = bsxfun(@minus,x2,a2);

    r1 = r(1);
    r2 = r(2);

    E = exp(-(r1^2 .* dx.^2 + r2^2 .* dy.^2));

    if ord == 0
        Phi = E;
    elseif ord == 1 && dir == 1
        Phi = -2 * r1^2 .* dx .* E;
    elseif ord == 1 && dir == 2
        Phi = -2 * r2^2 .* dy .* E;
    elseif ord == 2 && dir == 1
        Phi = (4 * r1^4 .* dx.^2 - 2 * r1^2) .* E;
    elseif ord == 2 && dir == 2
        Phi = (4 * r2^4 .* dy.^2 - 2 * r2^2) .* E;
    elseif ord == 2 && dir == 3
        Phi = 4 * r1^2 * r2^2 .* dx .* dy .* E;
    elseif ord == 3 && dir == 1
        Phi = (12 * r1^4 .* dx - 8 * r1^6 .* dx.^3) .* E;
    elseif ord == 3 && dir == 2
        Phi = (12 * r2^4 .* dy - 8 * r2^6 .* dy.^3) .* E;
    else
        error('radial_gauss does not support ord=%d, dir=%d.', ord, dir);
    end
end

function out = sin_dif_local(w,b,x,ord,dim)
    freq_scale = pi;
    xin = x*w + b;
    zin = freq_scale * xin;

    if ord == 0
        out = sin(zin);
    elseif ord == 1
        wk = w(dim,:);
        out = (freq_scale .* wk) .* cos(zin);
    elseif ord == 2
        if dim == 1 || dim == 2
            wk = w(dim,:);
            out = -(freq_scale .* wk).^2 .* sin(zin);
        elseif dim == 3
            out = -(freq_scale^2) .* (w(1,:).*w(2,:)) .* sin(zin);
        else
            error('sin does not support ord=%d, dim=%d.',ord,dim);
        end
    elseif ord == 3 && dim == 1
        out = -(freq_scale .* w(1,:)).^3 .* cos(zin);
    else
        error('sin only supports ord=0,1,2,3 (ord=3 needs dim=1).');
    end
end

function Phi = mexican_hat_dif_local(A_center,r,X,ord,dir)
%MEXICAN_HAT_DIF_LOCAL
%
% phi = (2 - s) * exp(-s/2)
% s = r1^2 (x-a)^2 + r2^2 (y-b)^2

    X = gather(X);
    A_center = gather(A_center);
    r = gather(r(:));

    x1 = X(:,1);
    x2 = X(:,2);

    a1 = A_center(1,:);
    a2 = A_center(2,:);

    dx = bsxfun(@minus,x1,a1);
    dy = bsxfun(@minus,x2,a2);

    r1 = r(1);
    r2 = r(2);

    s = r1^2 .* dx.^2 + r2^2 .* dy.^2;
    E = exp(-0.5*s);

    % q(s) = (2-s) exp(-s/2)
    q0 = (2 - s) .* E;
    q1 = (0.5*s - 2) .* E;
    q2 = (1.5 - 0.25*s) .* E;
    q3 = (0.125*s - 1) .* E;

    sx  = 2*r1^2 .* dx;
    sy  = 2*r2^2 .* dy;
    sxx = 2*r1^2;
    syy = 2*r2^2;

    if ord == 0
        Phi = q0;

    elseif ord == 1 && dir == 1
        Phi = q1 .* sx;

    elseif ord == 1 && dir == 2
        Phi = q1 .* sy;

    elseif ord == 2 && dir == 1
        Phi = q2 .* sx.^2 + q1 .* sxx;

    elseif ord == 2 && dir == 2
        Phi = q2 .* sy.^2 + q1 .* syy;

    elseif ord == 2 && dir == 3
        Phi = q2 .* sx .* sy;

    elseif ord == 3 && dir == 1
        Phi = q3 .* sx.^3 + 3*q2 .* sx .* sxx;

    elseif ord == 3 && dir == 2
        Phi = q3 .* sy.^3 + 3*q2 .* sy .* syy;

    else
        error('mexican_hat does not support ord=%d, dir=%d.', ord, dir);
    end
end
