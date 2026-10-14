function out = ORB_Eval3D(action,terms,X,ord,dir,coef)
%ORB_EVAL Compatibility wrapper for Burgers2D project.
    switch lower(action)
        case {'solution','eval'}
            out = Burgers2D_Newton_module3D('eval',terms,coef,X,ord,dir);
        otherwise
            error('Unknown ORB_Eval3D action = %s',action);
    end
end
