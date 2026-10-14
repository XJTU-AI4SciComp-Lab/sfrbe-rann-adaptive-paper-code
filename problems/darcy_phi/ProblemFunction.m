function out = ProblemFunction(action,X,problem)
    if nargin < 2, X = []; end
    if nargin < 3, problem = []; end
    out = DarcyPhiProblemFunction(action,X,problem,false,'darcy_phi.mat');
end
