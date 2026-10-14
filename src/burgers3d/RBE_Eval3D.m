function out = RBE_Eval3D(action,terms,X,ord,dir,coef)
%RBE_EVAL_COMPAT Compatibility wrapper for the Burgers3D RBE basis.
    switch lower(action)
        case {'solution','eval'}
            out = Burgers2D_Newton_module3D('eval',terms,coef,X,ord,dir);
        otherwise
            error('Unknown RBE_Eval3D action = %s',action);
    end
end
