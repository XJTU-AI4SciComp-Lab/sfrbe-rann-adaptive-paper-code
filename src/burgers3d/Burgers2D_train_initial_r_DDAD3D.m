function [r_opt,info] = Burgers2D_train_initial_r_DDAD3D(X_all,y_all,w_basis,b_basis,r0,cfg)
%BURGERS2D_TRAIN_INITIAL_R_DDAD
% Train the initial/global scale r using the DDAD approach.
%
% Objective:
%   min_r min_alpha || Phi(r)*alpha - y_all ||^2 + lambda ||alpha||^2
%
% Notes:
%   1. Keep ridge regularization during this r-training stage.
%   2. Use only neuron_train neurons to train r here.
%   3. The main program still uses unregularized SVD for the subsequent
%      production PDE/Newton solve.
%   4. y_all should be the current post-Newton numerical solution, not the
%      original u_guess.

    if nargin < 6
        cfg = struct();
    end

    if ~isfield(cfg,'family'), cfg.family = 'tanh'; end
    if ~isfield(cfg,'neuron_train'), cfg.neuron_train = 300; end
    if ~isfield(cfg,'max_train_points'), cfg.max_train_points = inf; end
    if ~isfield(cfg,'lambda_reg'), cfg.lambda_reg = 1e-6; end

    if ~isfield(cfg,'adam_opts')
        cfg.adam_opts = struct();
    end
    if ~isfield(cfg.adam_opts,'eta'), cfg.adam_opts.eta = 0.5; end
    if ~isfield(cfg.adam_opts,'eps_adam'), cfg.adam_opts.eps_adam = 1e-8; end
    if ~isfield(cfg.adam_opts,'maxit'), cfg.adam_opts.maxit = 40; end
    if ~isfield(cfg.adam_opts,'clip_min'), cfg.adam_opts.clip_min = ones(numel(r0),1); end
    if ~isfield(cfg.adam_opts,'clip_max'), cfg.adam_opts.clip_max = 300*ones(numel(r0),1); end
    if ~isfield(cfg.adam_opts,'minit'), cfg.adam_opts.minit = 5; end
    if ~isfield(cfg.adam_opts,'grad_tol'), cfg.adam_opts.grad_tol = 0; end
    if ~isfield(cfg.adam_opts,'fd_step_rel'), cfg.adam_opts.fd_step_rel = 1e-3; end
    if ~isfield(cfg.adam_opts,'verbose'), cfg.adam_opts.verbose = true; end

    X_all = gather(X_all);
    y_all = gather(y_all(:));

    n_all = size(X_all,1);

    if isinf(cfg.max_train_points)
        n_train = n_all;
    else
        n_train = min(n_all,cfg.max_train_points);
    end

    rng(20260524);
    if n_train < n_all
        idx = randperm(n_all,n_train);
    else
        idx = 1:n_all;
    end

    Xtrain = X_all(idx,:);
    ytrain = y_all(idx);

    m_train = min(cfg.neuron_train,size(w_basis,2));

    % Train r using the first m_train random directions and centers from the
    % production global basis.
    raw = struct();
    raw.family = cfg.family;
    raw.m = m_train;
    raw.w_basis = w_basis(:,1:m_train);
    raw.centers = b_basis(:,1:m_train);

    fprintf('\n========== DDAD training for global r ==========\n');
    fprintf('family = %s, neuron_train = %d, train_points = %d, lambda = %.3e\n', ...
        cfg.family,m_train,n_train,cfg.lambda_reg);
    fprintf('r0 = [%s ]\n',sprintf(' %.6f',r0(:)));

    [r_opt,MSE,term_opt] = Burgers2D_Newton_module3D( ...
        'train_r_value', ...
        Xtrain, ...
        ytrain, ...
        raw, ...
        r0, ...
        cfg.adam_opts, ...
        cfg.family, ...
        cfg.lambda_reg);

    info = struct();
    info.r0 = r0(:);
    info.r_opt = r_opt(:);
    info.MSE = MSE;
    info.term_opt = term_opt;
    info.neuron_train = m_train;
    info.train_points = n_train;

    fprintf('DDAD global-r training done: r = [%s ], MSE = %.6e\n', ...
        sprintf(' %.6f',r_opt(:)),MSE);
end
