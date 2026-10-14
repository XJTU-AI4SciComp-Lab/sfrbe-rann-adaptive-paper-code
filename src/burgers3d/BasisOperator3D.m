function A = BasisOperator3D(basis_type,term,X,op)
%BASISOPERATOR Compatibility wrapper for 3D linear operators.
%
% Supported coefficient fields:
%   alpha0*u
% + alpha_x*ux + alpha_y*uy + alpha_t*ut
% + alpha_xx*uxx + alpha_yy*uyy + alpha_tt*utt
%
% Old aliases are also accepted:
%   alpha1 -> alpha_x, alpha2 -> alpha_xx,
%   alpha3 -> alpha_t, alpha4 -> alpha_tt.

    if nargin < 4 || isempty(op)
        error('BasisOperator3D requires op.');
    end

    u   = BasisFunction3D(basis_type,term,X,0,0);
    ux  = BasisFunction3D(basis_type,term,X,1,1);
    uy  = BasisFunction3D(basis_type,term,X,1,2);
    ut  = BasisFunction3D(basis_type,term,X,1,3);
    uxx = BasisFunction3D(basis_type,term,X,2,1);
    uyy = BasisFunction3D(basis_type,term,X,2,2);
    utt = BasisFunction3D(basis_type,term,X,2,3);

    A = 0*u;
    if isfield(op,'alpha0'), A = A + op.alpha0.*u; end

    if isfield(op,'alpha_x'), A = A + op.alpha_x.*ux;
    elseif isfield(op,'alpha1'), A = A + op.alpha1.*ux; end

    if isfield(op,'alpha_y'), A = A + op.alpha_y.*uy; end

    if isfield(op,'alpha_t'), A = A + op.alpha_t.*ut;
    elseif isfield(op,'alpha3'), A = A + op.alpha3.*ut; end

    if isfield(op,'alpha_xx'), A = A + op.alpha_xx.*uxx;
    elseif isfield(op,'alpha2'), A = A + op.alpha2.*uxx; end

    if isfield(op,'alpha_yy'), A = A + op.alpha_yy.*uyy; end

    if isfield(op,'alpha_tt'), A = A + op.alpha_tt.*utt;
    elseif isfield(op,'alpha4'), A = A + op.alpha4.*utt; end
end
