function varargout = SAB_module3D(action,varargin)
%SAB_MODULE3D Compatibility wrapper for the paper-aligned SFR implementation.
% The Burgers driver uses SFR_module3D directly. This wrapper keeps older
% calls working without maintaining a second, divergent SFR implementation.
    if nargout == 0
        SFR_module3D(action,varargin{:});
    else
        [varargout{1:nargout}] = SFR_module3D(action,varargin{:});
    end
end
