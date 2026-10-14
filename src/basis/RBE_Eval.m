function out = RBE_Eval(action, orb_terms, X, ord, dir, gamma)
%RBE_EVAL_COMPAT unified evaluation of the RBE-corrected trial basis or current solution.
%
% Usage: 
%   Psi = RBE_Eval('terms', orb_terms, X, ord, dir)
%   val = RBE_Eval('solution', orb_terms, X, ord, dir, gamma)

    switch lower(action)
        case 'terms'
            out = eval_orb_terms_local(orb_terms,X,ord,dir);

        case 'solution'
            if nargin < 6
                error('RBE_Eval(''solution'',...) requires gamma.');
            end
            out = eval_orb_solution_local(orb_terms,X,ord,dir,gamma);

        otherwise
            error('Unknown RBE_Eval action = %s', action);
    end
end

function Psi = eval_orb_terms_local(orb_terms,X,ord,dir)
    r_total = size(orb_terms{1}.T,2);
    Psi = zeros(size(X,1),r_total);

    for tt = 1:numel(orb_terms)
        term = orb_terms{tt};

        if size(term.T,2) ~= r_total
            error('orb_terms{%d}.T column count does not match the total rank.', tt);
        end

        switch lower(term.type)
            case 'tanh'
                Phi = BasisFunction('tanh',term,X,ord,dir);
            case 'mexican_hat'
                Phi = BasisFunction('mexican_hat',term,X,ord,dir);

            case 'gauss'
                Phi = BasisFunction('ridge_gauss',term,X,ord,dir);

            case 'radial_gauss'
                Phi = BasisFunction('radial_gauss',term,X,ord,dir);

            case 'sin'
                Phi = BasisFunction('sin',term,X,ord,dir);

            case 'layer_growth'
                if ord ~= 0 || dir ~= 0
                    error('layer_growth supports only function values ord=0, dir=0. Implement derivatives for multiple LG passes.');
                end
                Phi = LayerGrowth_module('value', ...
                    X, term.construct_terms, term.gamma_construct, ...
                    term.centers, term.H0, term.Hnorm, term.bias, ...
                    term.act, term.use_localization, term.block_rows);

            otherwise
                error('Unknown orb_terms{%d}.type = %s', tt, term.type);
        end

        Psi = Psi + Phi * term.T;
    end
end

function val = eval_orb_solution_local(orb_terms,X,ord,dir,gamma)
    r_total = size(orb_terms{1}.T,2);

    if length(gamma) ~= r_total
                error('gamma length %d does not match the current RBE rank %d.', length(gamma), r_total);
    end

    val = zeros(size(X,1),1);

    for tt = 1:numel(orb_terms)
        term = orb_terms{tt};

        if size(term.T,2) ~= r_total
            error('orb_terms{%d}.T column count does not match the total rank.', tt);
        end

        beta = term.T * gamma;

        switch lower(term.type)
            case 'mexican_hat'
                Phi = BasisFunction('mexican_hat',term,X,ord,dir);

            case 'tanh'
                Phi = BasisFunction('tanh',term,X,ord,dir);

            case 'gauss'
                Phi = BasisFunction('ridge_gauss',term,X,ord,dir);

            case 'radial_gauss'
                Phi = BasisFunction('radial_gauss',term,X,ord,dir);

            case 'sin'
                Phi = BasisFunction('sin',term,X,ord,dir);

            case 'layer_growth'
                if ord ~= 0 || dir ~= 0
                    error('layer_growth supports only function values ord=0, dir=0. Set Lmax_lg=1.');
                end
                Phi = LayerGrowth_module('value', ...
                    X, term.construct_terms, term.gamma_construct, ...
                    term.centers, term.H0, term.Hnorm, term.bias, ...
                    term.act, term.use_localization, term.block_rows);

            otherwise
                error('Unknown orb_terms{%d}.type = %s', tt, term.type);
        end

        val = val + Phi * beta;
    end
end
