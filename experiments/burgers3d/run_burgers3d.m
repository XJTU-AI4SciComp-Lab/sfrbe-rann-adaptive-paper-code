function run_burgers3d(cfg_override)
%RUN_BURGERS3D Isolated 3D Burgers (x,y,t) pipeline.
%   Standalone stabilized fixed-operator SFR/RBE Burgers3D experiment.
    t_total = tic;
    warning('off','MATLAB:nearlySingularMatrix');
    warning('off','MATLAB:singularMatrix');
    warning('off','MATLAB:illConditionedMatrix');
    THIS_DIR = fileparts(mfilename('fullpath'));
    addpath(genpath(fullfile(fileparts(fileparts(THIS_DIR)),'src')));   % loads src + src/burgers3d (3D tools)
    if nargin >= 1 && ~isempty(cfg_override)
        cfg = cfg_override;
    else
        cfg = burgers3d_config();
    end

    % Create the run directory and capture the complete console transcript.
    % The standalone Burgers driver previously created only MAT summaries at
    % the very end, so no run.log was available when a run stopped early.
    runDir = fullfile(cfg.output.results_root, cfg.output.run_name);
    if ~exist(runDir,'dir'), mkdir(runDir); end
    logFile = fullfile(runDir,'run.log');
    if exist(logFile,'file'), delete(logFile); end
    diary(logFile);
    logCleanup = onCleanup(@() diary('off')); %#ok<NASGU>
    fprintf('============================================================\n');
    fprintf('SFRBE-RaNN Burgers3D run log\n');
    fprintf('problem     : Burgers2D_space_time\n');
    fprintf('seed        : %d\n',cfg.seed);
    fprintf('run dir     : %s\n',runDir);
    fprintf('started     : %s\n',datestr(now));
    fprintf('============================================================\n');

%% ======================== Problem definition ========================
problem = ProblemFunction3D('setup');
problem.eps_burgers = cfg.problem.eps_burgers;
problem = ProblemFunction3D('complete',[],problem);

dom = problem.domain;
x0 = dom(1); x1 = dom(2);
y0 = dom(3); y1 = dom(4);
t0 = dom(5); t1 = dom(6);

fprintf('Problem: %s\n',problem.name);
fprintf('PDE: u_t + u*(u_x+u_y) - %.4e*(u_xx+u_yy) = 0\n',problem.eps_burgers);
fprintf('Exact: u=1/(1+exp((x+y-t)/(2*eps))), eps=%.4e\n',problem.eps_burgers);

%% ======================== Main hyperparameters ========================
% 3D space-time input: X=[x,y,t].
% For a first run, keep these moderate. Increase N/neuron/neuron_add after
% verifying the pipeline.
N = cfg.grid.N;              % [Nx,Ny,Nt]
neuron = cfg.model.neuron;   % initial global basis size
neuron_add = cfg.model.neuron_add;  % RBE candidate pool size per accepted family
GPU = 0; QR = 1; %#ok<NASGU>

% Three independent distribution scales: [r_x; r_y; r_t].
r = cfg.model.r0(:);

rng(cfg.seed);
w_basis = 2*rand(3,neuron)-1;
b_basis = [(x1-x0)*rand(1,neuron)+x0;
           (y1-y0)*rand(1,neuron)+y0;
           (t1-t0)*rand(1,neuron)+t0];

%% ======================== 3D grid / boundary / initial points ========================
x_grid = linspace(x0,x1,N(1));
y_grid = linspace(y0,y1,N(2));
t_grid = linspace(t0,t1,N(3));

[XX,YY,TT] = ndgrid(x_grid,y_grid,t_grid);
XT_grid = [XX(:),YY(:),TT(:)];

% PDE interior: x,y exclude spatial boundary; t excludes initial time.
[Xi,Yi,Ti] = ndgrid(x_grid(2:end-1),y_grid(2:end-1),t_grid(2:end));
XT_int = [Xi(:),Yi(:),Ti(:)];

% Dirichlet boundary on all four spatial faces for t>0.
[Yb,Tb] = ndgrid(y_grid,t_grid(2:end));
X_x0 = [x0*ones(numel(Yb),1), Yb(:), Tb(:)];
X_x1 = [x1*ones(numel(Yb),1), Yb(:), Tb(:)];

[Xb,Tb2] = ndgrid(x_grid,t_grid(2:end));
X_y0 = [Xb(:), y0*ones(numel(Xb),1), Tb2(:)];
X_y1 = [Xb(:), y1*ones(numel(Xb),1), Tb2(:)];

X_bd = [X_x0; X_x1; X_y0; X_y1];
g_bd = ProblemFunction3D('boundary',X_bd,problem);

% Initial face t=0 on the full spatial grid.
[Xic,Yic] = ndgrid(x_grid,y_grid);
X_ic = [Xic(:),Yic(:),t0*ones(numel(Xic),1)];
u_ic = ProblemFunction3D('initial',X_ic,problem);

% Initial background: extend h0(x,y) constantly along t.
X_guess = [XT_grid(:,1),XT_grid(:,2),t0*ones(size(XT_grid,1),1)];
u_guess = ProblemFunction3D('initial',X_guess,problem);

fprintf('Grid: full=%d, interior=%d, boundary=%d, initial=%d\n', ...
    size(XT_grid,1),size(XT_int,1),size(X_bd,1),size(X_ic,1));

%% ======================== Initial global tanh basis ========================
w = r.*w_basis;
b = -sum(b_basis.*w,1);
term0 = struct('type','tanh','w',w,'b',b);
terms = {term0};

svd_solve_cfg = struct();
svd_solve_cfg.linear_solver = 'svd';
svd_solve_cfg.solve_lambda = 0;
svd_solve_cfg.svd_mode = 'rel_tol';
svd_solve_cfg.svd_rel_tol = 1e-10;
svd_solve_cfg.svd_energy_ratio = 0.99999999;
svd_solve_cfg.svd_fixed_rank = 1200;

coef = Burgers2D_Newton_module3D('fit_values_svd',terms,XT_grid,u_guess,svd_solve_cfg);

%% ======================== Evaluation points ========================
ppN = cfg.output.pp;
[xp,yp,tp] = deal(linspace(x0,x1,ppN(1)),linspace(y0,y1,ppN(2)),linspace(t0,t1,ppN(3)));
[XP,YP,TP] = ndgrid(xp,yp,tp);
pp_points = [XP(:),YP(:),TP(:)];
pred_out_ex = ProblemFunction3D('exact',pp_points,problem);

pred_initial = Burgers2D_Newton_module3D('eval',terms,coef,pp_points,0,0);
fprintf('Initial fitted-solution reference relative L2 error: %g\n', ...
    norm(pred_initial-pred_out_ex,'fro')/(norm(pred_out_ex,'fro')+1e-12));

%% ======================== Context ========================
SAB_eps = 1e-12;

orth_cfg.mode = 'rel_tol';
orth_cfg.rel_tol = 1e-10;
orth_cfg.fixed_rank = 1200;
orth_cfg.energy_ratio = 0.99999999;

grid3d = struct();
grid3d.x_grid = x_grid;
grid3d.y_grid = y_grid;
grid3d.t_grid = t_grid;
grid3d.XT_grid = XT_grid;
grid3d.XT_int = XT_int;
grid3d.X_bd = X_bd;
grid3d.g_bd = g_bd;
grid3d.X_ic = X_ic;
grid3d.u_ic = u_ic;
grid3d.Nx = N(1);
grid3d.Ny = N(2);
grid3d.Nt = N(3);

ctx = struct();
ctx.problem = problem;
ctx.grid = grid3d;
ctx.N = N;
ctx.neuron = neuron;
ctx.neuron_add = neuron_add;
ctx.GPU = GPU;
ctx.QR = QR;
ctx.r = r(:);
ctx.w = w;
ctx.b = b;
ctx.b_basis = b_basis;
ctx.terms = terms;
ctx.coef = coef;
ctx.weight = coef;
ctx.current_centers = b_basis;
ctx.pp_points = pp_points;
ctx.xp = xp; ctx.yp = yp; ctx.tp = tp;
ctx.SAB_eps = SAB_eps;
ctx.orth_cfg = orth_cfg;

%% ======================== Full Newton ========================
newton_cfg = struct();
newton_cfg.num_newton = 8;
newton_cfg.tau_newton = 1e-8;
newton_cfg.lambda = 1e-6;
newton_cfg.solve_lambda = 0;
newton_cfg.line_search = true;
newton_cfg.max_line_search = 8;
newton_cfg.eta_bc = 300;
newton_cfg.eta_ic = 300;
newton_cfg.verbose = true;

newton_cfg.tol_newton_improve = 5e-3;
newton_cfg.stall_patience = 2;
newton_cfg.min_newton_iter = 2;

newton_cfg.linear_solver = 'svd';
newton_cfg.svd_mode = 'rel_tol';
newton_cfg.svd_rel_tol = 1e-10;
newton_cfg.svd_energy_ratio = 0.99999999;
newton_cfg.svd_fixed_rank = 1200;
newton_cfg.fixed_operator = cfg.fixed_operator;

if isfield(cfg,'fixed_operator') && cfg.fixed_operator.enabled
    fprintf('\n========== Initial stabilized fixed-operator solve ==========\n');
else
    fprintf('\n========== Initial full SVD-Newton ==========\n');
end
ctx = Burgers2D_Newton_module3D('relax_full',ctx,newton_cfg);
fprintf('rel_res after the initial full Newton solve = %.6e\n',ctx.newton_info.rel_res);

pred_after_newton = Burgers2D_Newton_module3D('eval',ctx.terms,ctx.coef,pp_points,0,0);
fprintf('Reference relative L2 error after the first global Newton solve: %g\n', ...
    norm(pred_after_newton-pred_out_ex,'fro')/(norm(pred_out_ex,'fro')+1e-12));

%% ======================== Initial global-r DDAD after Newton ========================
init_ddad_cfg = struct();
init_ddad_cfg.enable = true;
init_ddad_cfg.max_outer = 2;
init_ddad_cfg.family = 'tanh';

init_ddad_cfg.neuron_train = min(1200,neuron);
init_ddad_cfg.max_train_points = min(size(XT_grid,1),12000);
init_ddad_cfg.lambda_reg = 1e-6;

init_ddad_cfg.newton_steps = 8;
init_ddad_cfg.stop_r_change = 0.03;
init_ddad_cfg.stop_res_improve = 0.01;
init_ddad_cfg.accept_if_no_improve = true;

init_ddad_cfg.adam_opts.eta = 0.5;
init_ddad_cfg.adam_opts.eps_adam = 1e-8;
init_ddad_cfg.adam_opts.maxit = 35;
init_ddad_cfg.adam_opts.clip_min = [1;1;1];
init_ddad_cfg.adam_opts.clip_max = [300;300;300];
init_ddad_cfg.adam_opts.minit = 5;
init_ddad_cfg.adam_opts.grad_tol = 0;
init_ddad_cfg.adam_opts.verbose = true;
init_ddad_cfg.adam_opts.use_rel_grad = true;
init_ddad_cfg.adam_opts.mse_tol = -inf;
init_ddad_cfg.adam_opts.rel_mse_tol = 0;
init_ddad_cfg.adam_opts.stall_patience = inf;

if init_ddad_cfg.enable
    fprintf('\n========== Initial global-r DDAD after first Newton ==========\n');

    rel_prev_ddad = ctx.newton_info.rel_res;
    r_prev_ddad = ctx.r(:);

    for outer = 1:init_ddad_cfg.max_outer
        fprintf('\n----- Initial global-r DDAD outer %d / %d -----\n', ...
            outer,init_ddad_cfg.max_outer);

        U_current = Burgers2D_Newton_module3D('grid_solution',ctx);
        y_ddad = U_current(:);

        [r_new,init_r_info] = Burgers2D_train_initial_r_DDAD3D( ...
            XT_grid,y_ddad,w_basis,b_basis,r_prev_ddad,init_ddad_cfg); %#ok<NASGU>

        w_new = r_new(:).*w_basis;
        b_new = -sum(b_basis.*w_new,1);
        term_new = struct('type','tanh','w',w_new,'b',b_new);
        terms_new = {term_new};

        [coef_new,init_info] = Burgers2D_Newton_module3D('best_burgers2d_init_svd', ...
            ctx,terms_new,XT_grid,y_ddad,newton_cfg);
        fprintf('DDAD init compare: value rel_res=%.6e, linearized rel_res=%.6e, use=%s\n', ...
            init_info.rel_value,init_info.rel_linearized,init_info.method);

        ctx_try = ctx;
        ctx_try.r = r_new(:);
        ctx_try.w = w_new;
        ctx_try.b = b_new;
        ctx_try.b_basis = b_basis;
        ctx_try.terms = terms_new;
        ctx_try.coef = coef_new;
        ctx_try.weight = coef_new;
        % DDAD rebuilds the whole basis, so the old fixed-operator SVD cache
        % no longer corresponds to these columns.  The next fixed solve
        % performs one fresh initialization and then resumes block updates.
        if isfield(ctx_try,'fixed_operator_svd_state')
            ctx_try.fixed_operator_svd_state = [];
        end
        if isfield(ctx_try,'fixed_operator_basis_cache')
            ctx_try.fixed_operator_basis_cache = [];
        end

        ddad_newton_cfg = newton_cfg;
        ddad_newton_cfg.num_newton = init_ddad_cfg.newton_steps;
        ctx_try = Burgers2D_Newton_module3D('relax_full',ctx_try,ddad_newton_cfg);

        rel_new_ddad = ctx_try.newton_info.rel_res;
        r_change = norm(r_new(:)-r_prev_ddad)/(norm(r_prev_ddad)+eps);
        res_improve = (rel_prev_ddad - rel_new_ddad)/(rel_prev_ddad + eps);

        fprintf('DDAD outer %d result: rel_res %.6e -> %.6e, improve=%.4e, r_change=%.4e\n', ...
            outer,rel_prev_ddad,rel_new_ddad,res_improve,r_change);

        accept = (rel_new_ddad <= rel_prev_ddad) || init_ddad_cfg.accept_if_no_improve;
        if accept
            ctx = ctx_try;
            rel_prev_ddad = rel_new_ddad;
            r_prev_ddad = r_new(:);
        else
            fprintf('DDAD outer %d rejected: residual did not improve.\n',outer);
            break;
        end

        if r_change < init_ddad_cfg.stop_r_change || res_improve < init_ddad_cfg.stop_res_improve
            fprintf('DDAD stop: r_change or residual improvement below threshold.\n');
            break;
        end
    end
end

pred_after_ddad = Burgers2D_Newton_module3D('eval',ctx.terms,ctx.coef,pp_points,0,0);
fprintf('Reference relative L2 error after DDAD: %g\n', ...
    norm(pred_after_ddad-pred_out_ex,'fro')/(norm(pred_out_ex,'fro')+1e-12));
    initial_ddad_time = toc(t_total);
    fprintf('Timing: Initial+DDAD = %.3f s\n',initial_ddad_time);
%% ======================== Repeated SFR diagnose and refresh ========================
t_sfr_stage = tic;
sab_cfg = struct();
sab_cfg.enable = true;
sab_cfg.SAB_eps = SAB_eps;
sab_cfg.diag_N = cfg.sfr.diag_N;
sab_cfg.theta_res = cfg.sfr.theta_res;
sab_cfg.theta_grad = cfg.sfr.theta_grad;
sab_cfg.theta_dorfler = cfg.sfr.theta_dorfler;
sab_cfg.delta_floor = cfg.sfr.delta_floor;
sab_cfg.exclude_boundary_layers = cfg.sfr.exclude_boundary_layers; % cell centers exclude boundary/initial faces
sab_cfg.min_region_size = cfg.sfr.min_region_size;
sab_cfg.tau_eff = cfg.sfr.tau_eff;

sab_num_rounds = cfg.sfr.num_rounds;   % number of adaptive-bias rounds

if sab_cfg.enable
    for sab_round = 1:sab_num_rounds
        fprintf('\n========== SFR adaptive-bias round %d / %d ==========\n', ...
            sab_round,sab_num_rounds);

        diagS = SFR_module3D('diagnose',ctx,sab_cfg);

        if isfield(diagS,'G_SAB') && ~diagS.G_SAB
            fprintf('SAB round %d: G_SAB=0; skip the refresh in this round.\n',sab_round);
            break;
        end

        sab_ref_cfg = struct();
        sab_ref_cfg.family = 'tanh';

        % Start each later round from the r learned by the previous SFR round
        % instead of always reusing the initial ctx.r.
        if isfield(ctx,'last_r_opt') && ~isempty(ctx.last_r_opt)
            sab_ref_cfg.r0 = ctx.last_r_opt(:);
        elseif isfield(ctx,'r_opt_region') && ~isempty(ctx.r_opt_region)
            sab_ref_cfg.r0 = mean(ctx.r_opt_region,2);
        else
            sab_ref_cfg.r0 = ctx.r(:);
        end

        %% ======================== SFR neuron allocation ratio ========================
        % Priority:
        %   1. diagS.mass_ratio
        %   2. diagS.alpha_S
        %   3. sum(diagS.p_S over structure region)
        %   4. default 0.60

        alpha_source = 'default 0.60';
        alpha_tmp = 0.60;

        if isfield(diagS,'mass_ratio') && ~isempty(diagS.mass_ratio)
            alpha_tmp = diagS.mass_ratio;
            alpha_source = 'diagS.mass_ratio';

        elseif isfield(diagS,'alpha_S') && ~isempty(diagS.alpha_S)
            alpha_tmp = diagS.alpha_S;
            alpha_source = 'diagS.alpha_S';

        elseif isfield(diagS,'p_S') && isfield(diagS,'structure_mask') ...
                && ~isempty(diagS.p_S) && ~isempty(diagS.structure_mask)

            p_tmp = diagS.p_S(:);
            mask_tmp = diagS.structure_mask(:);

            if numel(p_tmp) == numel(mask_tmp)
                alpha_tmp = sum(p_tmp(mask_tmp > 0));
                alpha_source = 'sum(diagS.p_S(structure_mask))';
            end

        elseif isfield(diagS,'p_S') && isfield(diagS,'component_id') ...
                && ~isempty(diagS.p_S) && ~isempty(diagS.component_id)

            p_tmp = diagS.p_S(:);
            comp_tmp = diagS.component_id(:);

            if numel(p_tmp) == numel(comp_tmp)
                alpha_tmp = sum(p_tmp(comp_tmp > 0));
                alpha_source = 'sum(diagS.p_S(component_id>0))';
            end
        end

        alpha_tmp = alpha_tmp(:);
        alpha_tmp = sum(alpha_tmp(isfinite(alpha_tmp)));

        if ~isfinite(alpha_tmp) || alpha_tmp <= 0
            alpha_tmp = 0.60;
            alpha_source = 'fallback 0.60';
        end

        sab_ref_cfg.alpha_use = min(max(alpha_tmp,0),1);

        mS_preview = round(ctx.neuron * sab_ref_cfg.alpha_use);
        mB_preview = ctx.neuron - mS_preview;

        fprintf('SFR alpha_use = %.6f, source = %s, predicted m_S=%d, m_B=%d\n', ...
            sab_ref_cfg.alpha_use, alpha_source, mS_preview, mB_preview);
        sab_ref_cfg.lambda_reg = 1e-6;
        sab_ref_cfg.gamma_AD = 1.0;
        sab_ref_cfg.max_train_points_S = [];
        sab_ref_cfg.max_train_points_B = [];
        sab_ref_cfg.sample_within_cell = cfg.sfr.sample_within_cell;

        sab_ref_cfg.newton_cfg = newton_cfg;
        sab_ref_cfg.newton_cfg.num_newton = 6;

        sab_ref_cfg.adam_opts = init_ddad_cfg.adam_opts;
        sab_ref_cfg.adam_opts.maxit = 40;
        sab_ref_cfg.adam_opts.verbose = false;

        % Accept whenever the nonlinear residual decreases; otherwise
        % SFR_module3D restores the previous state internally.
        sab_ref_cfg.eta_accept = 0.0;

        rel_before_sab = ctx.newton_info.rel_res;

        ctx = SFR_module3D('refresh',ctx,diagS,sab_ref_cfg);

        if isfield(ctx,'newton_info') && isfield(ctx.newton_info,'rel_res')
            rel_after_sab = ctx.newton_info.rel_res;
        else
            [~,~,parts_tmp] = Burgers2D_Newton_module3D('assemble',ctx,newton_cfg);
            rel_after_sab = parts_tmp.rel_res;
        end

        pred_sab_tmp = Burgers2D_Newton_module3D('eval',ctx.terms,ctx.coef,pp_points,0,0);
        l2_sab_tmp = norm(pred_sab_tmp-pred_out_ex,'fro')/(norm(pred_out_ex,'fro')+1e-12);

        fprintf('SFR round %d result: rel_res %.6e -> %.6e, ref_L2=%.6e\n', ...
            sab_round,rel_before_sab,rel_after_sab,l2_sab_tmp);

        %% ======================== Plot SFR structure/background bias centers ========================
        if cfg.output.make_figs && isfield(ctx,'current_centers') && ~isempty(ctx.current_centers)
            C = ctx.current_centers;   % 3 x M

            if isfield(ctx,'sab_refresh_info') && isfield(ctx.sab_refresh_info,'mS_total')
                mS = ctx.sab_refresh_info.mS_total;
            else
                mS = round(size(C,2) * 0.5);
            end

            C_S = C(:,1:mS);           % structure centers
            C_B = C(:,mS+1:end);       % background centers

            figure;
            scatter3(C_B(1,:), C_B(2,:), C_B(3,:), 6, [0.65 0.65 0.65], 'filled'); hold on;
            scatter3(C_S(1,:), C_S(2,:), C_S(3,:), 12, 'r', 'filled');

            xlabel('x'); ylabel('y'); zlabel('t');
            title(sprintf('SAB round %d adaptive bias centers',sab_round));
            legend('background bias centers','structure bias centers');
            grid on; box on;
            axis([0 1 0 1 0 1]);
            view(45,25);
        end
    end
end

pred_after_sab = Burgers2D_Newton_module3D('eval',ctx.terms,ctx.coef,pp_points,0,0);
fprintf('Reference relative L2 error after SFR: %g\n', ...
    norm(pred_after_sab-pred_out_ex,'fro')/(norm(pred_out_ex,'fro')+1e-12));
    sfr_time = toc(t_sfr_stage);
    fprintf('Timing: SFR = %.3f s\n',sfr_time);

%% ======================== Freeze J_ref and RBE family enrichment ========================
t_rbe_stage = tic;
fprintf('\n========== Freeze J_ref before RBE ==========\n');
ctx = Burgers2D_Newton_module3D('freeze_jacobian',ctx,newton_cfg);

score_cfg = struct();
score_cfg.families = {'tanh','ridge_gauss','sin','radial_gauss','mexican_hat'};
if isfield(cfg,'rbe'), rbe_cfg = cfg.rbe; else, rbe_cfg = struct(); end
if ~isfield(rbe_cfg,'max_family_rounds'), rbe_cfg.max_family_rounds = 6; end
if ~isfield(rbe_cfg,'family_repeat_mode'), rbe_cfg.family_repeat_mode = 'exclude_last'; end
if ~isfield(rbe_cfg,'screen_ratio'), rbe_cfg.screen_ratio = 0.30; end
if ~isfield(rbe_cfg,'screen_max'), rbe_cfg.screen_max = 500; end
if ~isfield(rbe_cfg,'prefix_score_ratio'), rbe_cfg.prefix_score_ratio = 0.80; end
if ~isfield(rbe_cfg,'tau_stop_score'), rbe_cfg.tau_stop_score = 0.25; end
if ~isfield(rbe_cfg,'tau_cap'), rbe_cfg.tau_cap = 1e-4; end
if ~isfield(rbe_cfg,'kappa_map_max'), rbe_cfg.kappa_map_max = 1e6; end
if ~isfield(rbe_cfg,'delta'), rbe_cfg.delta = 3.0e-3; end
if ~isfield(rbe_cfg,'q'), rbe_cfg.q = 2; end
if ~isfield(rbe_cfg,'rho'), rbe_cfg.rho = 0.85; end
if ~isfield(rbe_cfg,'lambda'), rbe_cfg.lambda = 1e-6; end
if ~isfield(rbe_cfg,'gamma_AD'), rbe_cfg.gamma_AD = 1.0; end
if ~isfield(rbe_cfg,'max_train_points'), rbe_cfg.max_train_points = []; end
if ~isfield(rbe_cfg,'tol_improve'), rbe_cfg.tol_improve = 0.1; end
if ~isfield(rbe_cfg,'stop_when_small_improve'), rbe_cfg.stop_when_small_improve = true; end
if ~isfield(rbe_cfg,'stop_when_reject'), rbe_cfg.stop_when_reject = true; end
score_cfg.max_family_rounds = rbe_cfg.max_family_rounds;
% Use the common RBE screening rule: m_scr=min{500,ceil(0.3*m_full)}.
score_cfg.neuron_score = min(rbe_cfg.screen_max,ceil(rbe_cfg.screen_ratio*neuron_add));
score_cfg.neuron_add = neuron_add;
score_cfg.m_frac_list = [0.2,0.4,0.5,0.6,0.7,0.8,0.9,1.0];
score_cfg.prefix_score_ratio = rbe_cfg.prefix_score_ratio;
score_cfg.tau_stop_score = rbe_cfg.tau_stop_score;
score_cfg.family_repeat_mode = rbe_cfg.family_repeat_mode;
score_cfg.delta = rbe_cfg.delta;
score_cfg.q = rbe_cfg.q;
score_cfg.rho = rbe_cfg.rho;
score_cfg.lambda = rbe_cfg.lambda;
% Use the paper C.1 logarithmic training-point budget.  A positive
% max_train_points may still be supplied as an additional hard cap.
score_cfg.gamma_AD = rbe_cfg.gamma_AD;
score_cfg.max_train_points = rbe_cfg.max_train_points;
score_cfg.tau_cap = rbe_cfg.tau_cap;
score_cfg.kappa_map_max = rbe_cfg.kappa_map_max;
score_cfg.family_tol_accept = 1e-4;
score_cfg.tol_improve = rbe_cfg.tol_improve;
score_cfg.stop_when_reject = rbe_cfg.stop_when_reject;
score_cfg.stop_when_small_improve = rbe_cfg.stop_when_small_improve;
score_cfg.tau_block_start = 0;
score_cfg.r0 = ctx.r(:);
score_cfg.newton_cfg = newton_cfg;
score_cfg.jref_cfg = newton_cfg;
score_cfg.jref_cfg.orth_cfg = orth_cfg;
score_cfg.family_full_newton_cfg = newton_cfg;
score_cfg.family_full_newton_cfg.num_newton = 2;
score_cfg.incremental_init = true;
score_cfg.screen_adam_opts = init_ddad_cfg.adam_opts;
score_cfg.screen_adam_opts.maxit = 35;
score_cfg.block_adam_opts = score_cfg.screen_adam_opts;
score_cfg.screen_adam_opts.verbose = false;
score_cfg.block_adam_opts.verbose = false;
score_cfg.retrain_full_block = false;
ctx = RBE_FamilyScore_module3D(ctx,score_cfg);

%% ======================== Final nonlinear polish ========================
if isfield(cfg,'fixed_operator') && cfg.fixed_operator.enabled
    fprintf('\n========== Final stabilized fixed-operator polish ==========\n');
else
    fprintf('\n========== Final full Newton polishing ==========\n');
end
final_cfg = newton_cfg;
final_cfg.num_newton = 0;
ctx = Burgers2D_Newton_module3D('relax_full',ctx,final_cfg);

pred_final = Burgers2D_Newton_module3D('eval',ctx.terms,ctx.coef,pp_points,0,0);
rel_l2 = norm(pred_final-pred_out_ex,'fro')/(norm(pred_out_ex,'fro')+1e-12);
fprintf('\nFinal relative L2 error = %.6e\n',rel_l2);
fprintf('Final nonlinear relative residual = %.6e\n',ctx.newton_info.rel_res);
rbe_time = toc(t_rbe_stage);
solver_time = toc(t_total);
fprintf('Timing: RBE+final polish = %.3f s\n',rbe_time);
fprintf('Timing: Total solver (excluding data/figure saving) = %.3f s\n',solver_time);

%% ======================== Save dense 50x50x50 time-slice data ========================
t_post_stage = tic;
% Ported from the legacy main_score_copy.m.  Only saves dense post-processing
% data (all_slices_50x50x50.mat) for the cube-surface plot; does not change
% the solver / basis / SFR / RBE.
if isfield(cfg,'output') && isfield(cfg.output,'save_dense') && cfg.output.save_dense
    fprintf('\n========== Save dense 50x50x50 time-slice data ==========\n');
    dense_save_cfg = struct();
    dense_save_cfg.N = cfg.output.dense_N;          % dense grid: Nx x Ny x Nt
    dense_save_cfg.chunk_size = cfg.output.dense_chunk;
    dense_save_cfg.root_dir = fullfile(cfg.output.results_root, cfg.output.run_name, 'dense');
    dense_save_cfg.save_each_slice = true;
    dense_save_cfg.save_all_3d = true;
    if ~exist(dense_save_cfg.root_dir,'dir'), mkdir(dense_save_cfg.root_dir); end
    run_tag = datestr(now,'yyyymmdd_HHMMSS');
    dense_out_dir = fullfile(dense_save_cfg.root_dir,['run_',run_tag]);
    if ~exist(dense_out_dir,'dir'), mkdir(dense_out_dir); end

    Nx_dense = dense_save_cfg.N(1);
    Ny_dense = dense_save_cfg.N(2);
    Nt_dense = dense_save_cfg.N(3);
    x_vec = linspace(x0,x1,Nx_dense);
    y_vec = linspace(y0,y1,Ny_dense);
    t_values = linspace(t0,t1,Nt_dense);
    [X_grid_dense,Y_grid_dense] = ndgrid(x_vec,y_vec);
    n_xy_dense = numel(X_grid_dense);

    U_pred = zeros(Nx_dense,Ny_dense,Nt_dense);
    U_exact = zeros(Nx_dense,Ny_dense,Nt_dense);
    U_error = zeros(Nx_dense,Ny_dense,Nt_dense);
    rel_L2_time = zeros(Nt_dense,1);
    max_error_time = zeros(Nt_dense,1);

    for kt_dense = 1:Nt_dense
        t_value = t_values(kt_dense);
        X_slice_dense = [X_grid_dense(:), Y_grid_dense(:), t_value*ones(n_xy_dense,1)];
        exact_vec = ProblemFunction3D('exact',X_slice_dense,problem);
        pred_vec = zeros(size(exact_vec));
        for i0_dense = 1:dense_save_cfg.chunk_size:n_xy_dense
            i1_dense = min(i0_dense + dense_save_cfg.chunk_size - 1,n_xy_dense);
            Xi_dense = X_slice_dense(i0_dense:i1_dense,:);
            pred_tmp = Burgers2D_Newton_module3D('eval',ctx.terms,ctx.coef,Xi_dense,0,0);
            pred_vec(i0_dense:i1_dense) = pred_tmp(:);
        end
        U_pred_slice = reshape(pred_vec,Nx_dense,Ny_dense);
        U_exact_slice = reshape(exact_vec,Nx_dense,Ny_dense);
        U_error_slice = U_pred_slice - U_exact_slice;
        U_pred(:,:,kt_dense) = U_pred_slice;
        U_exact(:,:,kt_dense) = U_exact_slice;
        U_error(:,:,kt_dense) = U_error_slice;
        rel_L2_time(kt_dense) = norm(U_error_slice(:),2)/(norm(U_exact_slice(:),2)+1e-12);
        max_error_time(kt_dense) = max(abs(U_error_slice(:)));

        if dense_save_cfg.save_each_slice
            U_pred_xy = U_pred_slice; U_exact_xy = U_exact_slice; U_error_xy = U_error_slice;
            U_pred_yx = U_pred_slice.'; U_exact_yx = U_exact_slice.'; U_error_yx = U_error_slice.';
            t_str = strrep(sprintf('%.6f',t_value),'.','p');
            slice_file = fullfile(dense_out_dir, sprintf('slice_%03d_t_%s.mat',kt_dense,t_str));
            save(slice_file, ...
                'x_vec','y_vec','t_values','t_value','kt_dense', ...
                'X_grid_dense','Y_grid_dense', ...
                'U_pred_xy','U_exact_xy','U_error_xy', ...
                'U_pred_yx','U_exact_yx','U_error_yx', ...
                'rel_L2_time','max_error_time','dense_save_cfg','-v7.3');
        end
        fprintf('Saved dense slice %03d / %03d, t = %.6f, rel_L2(t) = %.6e, max_err(t) = %.6e\n', ...
            kt_dense,Nt_dense,t_value,rel_L2_time(kt_dense),max_error_time(kt_dense));
    end

    U_pred_yx = permute(U_pred,[2,1,3]);
    U_exact_yx = permute(U_exact,[2,1,3]);
    U_error_yx = permute(U_error,[2,1,3]);
    rel_L2_dense_all = norm(U_error(:),2)/(norm(U_exact(:),2)+1e-12);
    max_error_dense_all = max(abs(U_error(:)));

    dense_info = struct();
    dense_info.problem_name = problem.name;
    dense_info.eps_burgers = problem.eps_burgers;
    dense_info.domain = problem.domain;
    dense_info.N_dense = dense_save_cfg.N;
    dense_info.grid_order_native = 'U_pred(ix,iy,it), generated by ndgrid(x_vec,y_vec,t_values)';
    dense_info.grid_order_plot = 'U_pred_yx(iy,ix,it), direct for imagesc(x_vec,y_vec,U_pred_yx(:,:,it))';
    dense_info.rel_L2_dense_all = rel_L2_dense_all;
    dense_info.max_error_dense_all = max_error_dense_all;
    dense_info.rel_L2_final_pp_grid = rel_l2;
    dense_info.final_nonlinear_rel_res = ctx.newton_info.rel_res;
    dense_info.created_time = datestr(now);

    save(fullfile(dense_out_dir,'metadata_50x50x50.mat'), ...
        'x_vec','y_vec','t_values','rel_L2_time','max_error_time', ...
        'rel_L2_dense_all','max_error_dense_all','dense_info','dense_save_cfg','-v7.3');
    if dense_save_cfg.save_all_3d
        save(fullfile(dense_out_dir,'all_slices_50x50x50.mat'), ...
            'x_vec','y_vec','t_values', ...
            'U_pred','U_exact','U_error', ...
            'U_pred_yx','U_exact_yx','U_error_yx', ...
            'rel_L2_time','max_error_time', ...
            'rel_L2_dense_all','max_error_dense_all', ...
            'dense_info','dense_save_cfg','-v7.3');
    end
    fprintf('Dense 50x50x50 data saved to folder:\n  %s\n',dense_out_dir);
    fprintf('Dense-grid relative L2 error = %.6e, max error = %.6e\n', ...
        rel_L2_dense_all,max_error_dense_all);
    fprintf('Cube plot: plot_burgers_cube(''%s'')\n', dense_out_dir);
end

%% ======================== Plot one y-slice for quick inspection ========================
if cfg.output.make_figs
    try
    y_mid = 0.5;
    [XS,TS] = ndgrid(xp,tp);
    YS = y_mid*ones(numel(XS),1);
    X_slice = [XS(:),YS(:),TS(:)];
    pred_slice = reshape(Burgers2D_Newton_module3D('eval',ctx.terms,ctx.coef,X_slice,0,0),numel(xp),numel(tp));
    exact_slice = reshape(ProblemFunction3D('exact',X_slice,problem),numel(xp),numel(tp));

    figure;
    imagesc(tp,xp,abs(pred_slice-exact_slice));
    set(gca,'YDir','normal');
    colorbar;
    xlabel('t'); ylabel('x');
    title('|error| at y=0.5');
    saveas(gcf,'burgers2d_error_slice_y05.png');
    catch ME
        fprintf('Plot skipped: %s\n',ME.message);
    end
end

    if ~exist(runDir,'dir'), mkdir(runDir); end
    % ctx contains the full fitted basis/state and can exceed MATLAB's
    % legacy MAT-file size limit; use -v7.3 so the summary is actually saved.
    % The fixed-operator response cache is transient and is not part of the
    % reproducible solver state; omit it from the summary to avoid duplicating
    % the large response matrices on disk.
    if isfield(ctx,'fixed_operator_basis_cache')
        ctx.fixed_operator_basis_cache = [];
    end
    save(fullfile(runDir,'99_burgers3d_summary.mat'), ...
        'ctx','problem','rel_l2','N','neuron','neuron_add','cfg','-v7.3');
    fprintf('Saved: %s\n', fullfile(runDir,'99_burgers3d_summary.mat'));

    % Save SFR bias centers (structure/background) for plot_burgers_bias.
    if isfield(ctx,'current_centers') && ~isempty(ctx.current_centers)
        C = ctx.current_centers;   % 3 x M
        if isfield(ctx,'sab_refresh_info') && isfield(ctx.sab_refresh_info,'mS_total')
            mS = ctx.sab_refresh_info.mS_total;
        else
            mS = round(size(C,2) * 0.5);
        end
        sab_bias = struct();
        sab_bias.centers = C;
        sab_bias.mS_total = mS;
        sab_bias.centersS = C(:,1:mS);      % structure
        sab_bias.centersB = C(:,mS+1:end);  % background
        sab_bias.run_dir = runDir;
    save(fullfile(runDir,'sab_bias.mat'),'sab_bias');
        fprintf('Saved SFR bias centers: %s\n', fullfile(runDir,'sab_bias.mat'));
    end
postprocess_time = toc(t_post_stage);
total_wall_time = toc(t_total);
fprintf('Timing: Postprocessing/data/figures = %.3f s\n',postprocess_time);
fprintf('Timing: Total wall time (including postprocessing) = %.3f s\n',total_wall_time);
end
