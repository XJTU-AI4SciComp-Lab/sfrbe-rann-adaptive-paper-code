function A = BasisOperator(basis_type, term, X, op)
%BASISOPERATOR Assemble a generic 2D linear differential operator matrix.
%
%   A = BasisOperator(basis_type, term, X, op)
%
% The operator is consistent with ProblemFunction.op:
%   L phi = alpha0*phi + alpha1*phi_x + alpha2*phi_xx
%         + alpha3*phi_y + alpha4*phi_yy + alpha5*phi_xy.
%
% Coefficients alpha0,...,alpha5 may be scalars, N-vectors, or function
% handles evaluated at X. This file is intentionally separated from
% ProblemFunction so that all feature families use the same PDE operator.

    if nargin < 4 || isempty(op)
        error('BasisOperator requires op with fields alpha0,...,alpha5.');
    end

    names = {'alpha0','alpha1','alpha2','alpha3','alpha4','alpha5'};
    for kk = 1:numel(names)
        if ~isfield(op,names{kk})
            error('BasisOperator: op is missing field %s.', names{kk});
        end
    end

    N = size(X,1);
    m = get_term_width_local(term);
    A = zeros(N,m);

    A = add_operator_part_local(A,basis_type,term,X,op.alpha0,0,0);
    A = add_operator_part_local(A,basis_type,term,X,op.alpha1,1,1);
    A = add_operator_part_local(A,basis_type,term,X,op.alpha2,2,1);
    A = add_operator_part_local(A,basis_type,term,X,op.alpha3,1,2);
    A = add_operator_part_local(A,basis_type,term,X,op.alpha4,2,2);
    A = add_operator_part_local(A,basis_type,term,X,op.alpha5,2,3);
    if isfield(op,'alpha6')
        A = add_operator_part_local(A,basis_type,term,X,op.alpha6,3,1);  % u_xxx
    end
end

function A = add_operator_part_local(A,basis_type,term,X,coef,ord,dir)
    [c,skip] = coef_to_column_local(coef,X);
    if skip
        return;
    end

    Phi = BasisFunction(basis_type,term,X,ord,dir);

    if isscalar(c)
        A = A + c .* Phi;
    else
        A = A + bsxfun(@times,c,Phi);
    end
end

function [c,skip] = coef_to_column_local(coef,X)
    N = size(X,1);

    if isempty(coef)
        c = 0;
        skip = true;
        return;
    end

    if isa(coef,'function_handle')
        c = coef(X);
        c = gather(c(:));
        if numel(c) ~= N
            error('BasisOperator: function-handle coefficient must return N values.');
        end
        skip = all(c == 0);
        return;
    end

    if isscalar(coef)
        c = gather(coef);
        skip = (c == 0);
        return;
    end

    c = gather(coef(:));
    if numel(c) ~= N
        error('BasisOperator: coefficient length should be N=%d, got %d.',N,numel(c));
    end
    skip = all(c == 0);
end

function m = get_term_width_local(term)
    if isfield(term,'w') && ~isempty(term.w)
        m = size(term.w,2);
    elseif isfield(term,'A') && ~isempty(term.A)
        m = size(term.A,2);
    elseif isfield(term,'centers') && ~isempty(term.centers)
        m = size(term.centers,2);
    else
        error('BasisOperator: cannot infer number of basis functions from term.');
    end
end
