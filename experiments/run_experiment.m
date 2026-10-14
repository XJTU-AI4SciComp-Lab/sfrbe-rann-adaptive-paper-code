function run_experiment(cfg)
%RUN_EXPERIMENT unified pipeline: Initial -> PDAD(AD) -> SFR -> RBE -> evaluate/save/plot.
%
%   All random stages use the scalar cfg.seed, reset before every random stage
%   for reproducibility.
%   AD / SFR / RBE scale training keeps the iterate with the smallest residual
%   MSE (best selection).
%   Errors (Rel./Abs. L2 and H1) are reported at every stage.
%   Outputs:
%     - run.log            plain-text run log (selected family and r per round;
%                          per-stage errors and timings; cat/tail friendly)
%     - *.mat              only plotting data (U_pred/U_exact/U_error/anchors)
%     - 99_summary.mat     minimal summary (including summary.rbe.log)
    tStart = tic;

    warning('off', 'MATLAB:nearlySingularMatrix');
    warning('off', 'MATLAB:singularMatrix');
    warning('off', 'MATLAB:illConditionedMatrix');

    rng(cfg.seed);

    %% ==================== output directory ====================
    if isfield(cfg.output, 'run_name') && ~isempty(cfg.output.run_name)
        runTag = cfg.output.run_name;
    else
        runTag = cfg.problem;   % default: one fixed result dir per problem, overwritten each run
    end
    runDir = fullfile(cfg.output.results_root, runTag);
    if exist(runDir, 'dir'), rmdir(runDir, 's'); end
    mkdir(runDir);
    fprintf('\n[SFRBE] %s  seed=%d  ->  %s\n', cfg.paper.table, cfg.seed, runDir);

    appendRunLog(runDir, sprintf([ ...
        '============================================================\n' ...
        'SFRBE-RaNN run log\n' ...
        'paper table : %s\n' ...
        'problem     : %s\n' ...
        'seed        : %d\n' ...
        'run dir     : %s\n' ...
        'started     : %s\n' ...
        '============================================================'], ...
        cfg.paper.table, cfg.problem, cfg.seed, runDir, datestr(now)));

    %% ==================== problem definition ====================
    projRoot = fileparts(fileparts(mfilename('fullpath')));
    addpath(fullfile(projRoot, 'problems', cfg.problem), '-begin');
    problem = ProblemFunction('setup');
    problem = ProblemFunction('complete', [], problem);
    op = problem.op;
    dom = problem.domain;
    a1 = dom(1); a2 = dom(2); a3 = dom(3); a4 = dom(4);

    %% ==================== grid / boundary / initial basis ====================
    N = cfg.grid.N;
    neuron = cfg.model.neuron;
    b_point = cfg.grid.b_point;
    neuron_add = cfg.model.neuron_add;
    GPU = cfg.solver.GPU;
    QR = cfg.solver.QR;
    eta0 = cfg.model.eta0;
    r0 = cfg.model.r0(:);
    family = cfg.model.family;

    rng(cfg.seed);
    w_basis = 2 * rand(2, neuron) - 1;
    b_basis = [(a2 - a1) * rand(1, neuron) + a1;
               (a4 - a3) * rand(1, neuron) + a3];

    gridInfo = tryProblemAction(problem, 'grid', N);
    if isempty(gridInfo)
        % default rectangular domain
        x_point = YY_utils('get_inner_point', a1, a2, a3, a4, N);
        x_line = linspace(a1, a2, b_point)';
        y_line = linspace(a3, a4, b_point)';
        x_0 = [x_line, a3 * ones(size(x_line));
               x_line, a4 * ones(size(x_line));
               a1 * ones(b_point, 1), y_line;
               a2 * ones(b_point, 1), y_line];
    else
        % problem-defined (possibly non-rectangular) interior/boundary points
        x_point = gridInfo.x_point;
        x_0 = gridInfo.x_0;
    end
    right = ProblemFunction('rhs', x_point, problem);
    y0_exact = ProblemFunction('boundary', x_0, problem);
    neumann_info = tryProblemAction(problem, 'neumann', []);
    eta_neumann = getfield_default_local(cfg.model, 'eta_neumann', eta0);

    %% ==================== PDAD: initial scale training (best selection) ====================
    % PDAD is a linear-equation-oriented scale warm-up (Poisson-type).  For
    % nonlinear problems (cfg.run_pdad = false) it is skipped and the initial
    % scale r0 is used directly, so the very first stage is Newton from the
    % zero state, followed by the DDAD self-data refit.
    if isfield(cfg, 'run_pdad') && ~cfg.run_pdad
        r = r0(:);
        fprintf('  [PDAD] skipped (run_pdad=false); use r0=[%s]\n', ...
            num2str(r0.', '%.6g '));
        appendRunLog(runDir, sprintf('[PDAD] skipped (run_pdad=false); use r0=[%s]', ...
            num2str(r0.', '%.6g ')));
    else
        pdad = cfg.pdad;
        rng(cfg.seed);
        idxPde = randperm(size(x_point, 1), min(pdad.n_pde_rtrain, size(x_point, 1)));
        idxBd  = randperm(size(x_0, 1), min(pdad.n_bd_rtrain, size(x_0, 1)));
        idxNeu = randperm(size(w_basis, 2), min(pdad.n_neuron_rtrain, size(w_basis, 2)));

        [r_opt, ~] = PDAD_train_initial_r_analytic_poisson( ...
            x_point(idxPde, :), right(idxPde, :), ...
            x_0(idxBd, :), y0_exact(idxBd, :), ...
            w_basis(:, idxNeu), b_basis(:, idxNeu), ...
            r0, op, pdad.opts);
        r = r_opt(:);
        fprintf('  [PDAD] r0=[%s] -> best r=[%s]\n', ...
            num2str(r0.', '%.6g '), num2str(r.', '%.6g '));
        appendRunLog(runDir, sprintf('[PDAD] r0=[%s] best r=[%s]', ...
            num2str(r0.', '%.6g '), num2str(r.', '%.6g ')));
    end

    %% ==================== initial global RaNN solve ====================
    w = r .* w_basis;
    b = -sum(b_basis .* w, 1);
    term0 = struct('w', w, 'b', b);

    use_value_fit_start = (isfield(cfg, 'run_newton') && cfg.run_newton) || ...
        (is_fixed_operator_enabled(cfg) && getfield_default_local(cfg.fixed_operator, 'initial_value_fit', false));
    if use_value_fit_start
        % Initialization WITHOUT any known solution values: value-fit the
        % basis to the initial data extended along time (u(x,0) at every
        % point).  Only the initial condition (a problem input) is used;
        % the Newton iterations below (run_newton_initial) refine the full
        % nonlinear residual, then the DDAD stage re-fits the basis to the
        % self-computed solution.  The exact/reference solution is used only
        % for the final accuracy evaluation, never for training.
        Xbg = [x_point; x_0];
        u_background = tryProblemAction(problem, 'initial', [Xbg(:,1), a3*ones(size(Xbg,1),1)]);
        if isempty(u_background)
            error('run_experiment:noInitial', ...
                'Nonlinear initial solve requires ProblemFunction(''initial'',...). Add it to the problem definition.');
        end
        Phi_all = BasisFunction(family, term0, Xbg, 0, 0);
        weight = YY_utils('solve_af', Phi_all, u_background, GPU, QR);
    else
        A = BasisOperator(family, term0, x_point, op);
        Phi_0 = BasisFunction(family, term0, x_0, 0, 0);
        A_n = neumann_flux_matrix_local(family,term0,neumann_info,eta_neumann);
        matrix_a = [A; eta0 * Phi_0; A_n];
        matrix_b = [right; eta0 * y0_exact; zeros(size(A_n,1),1)];
        weight = YY_utils('solve_af', matrix_a, matrix_b, GPU, QR);
    end

    if is_fixed_operator_enabled(cfg)
        consN = ProblemFunction('constraints', [], problem);
        basisEval0 = @(ord, dir, X) BasisFunction(family, term0, X, ord, dir);
        [weight, fi0] = FixedOperatorSolver(problem, basisEval0, x_point, consN, weight, ...
            fixed_cfg_with_rhs(cfg.fixed_operator, op, right, eta0));
        fprintf('  [FixedOp] initial refine: loss=%.4e, rel_res=%.4e, iter=%d\n', fi0.loss, fi0.rel_res, fi0.iter);
        appendRunLog(runDir, sprintf('[FixedOp] initial refine: loss=%.4e rel_res=%.4e iter=%d', fi0.loss, fi0.rel_res, fi0.iter));
    end

    % Optional nonlinear refinement (e.g., KdV): damped Newton on term0.
    if isfield(cfg, 'run_newton_initial') && cfg.run_newton_initial
        consN = tryProblemAction(problem, 'constraints', []);
        if ~isempty(consN)
            basisEval0 = @(ord, dir, X) BasisFunction(family, term0, X, ord, dir);
            [weight, nw0] = NewtonSolver(problem, basisEval0, x_point, consN, weight, cfg.newton);
            fprintf('  [Newton] initial refine: rel_res=%.4e, iter=%d\n', nw0.rel_res, nw0.iter);
            appendRunLog(runDir, sprintf('[Newton] initial refine: rel_res=%.4e, iter=%d', nw0.rel_res, nw0.iter));
        end
    end

    % Optional DDAD r-training after the initial Newton (nonlinear problems):
    % short Newton -> train r on the current solution values ->
    % rebuild basis -> re-fit to the self-computed solution -> short Newton again.
    if isfield(cfg, 'run_ddad_after_newton') && cfg.run_ddad_after_newton
        consN = tryProblemAction(problem, 'constraints', []);
        if ~isempty(consN)
            % (a) short Newton from the zero/Newton start.  Newton only
            %     accepts non-increasing residual steps, so no exact/reference
            %     values are needed for the accept/reject decision.
            nAllD = size(x_point, 1);
            stepD = max(1, floor(nAllD / 4000));
            basisEval0 = @(ord, dir, X) BasisFunction(family, term0, X, ord, dir);
            if is_fixed_operator_enabled(cfg)
                [weight, fiD] = FixedOperatorSolver(problem, basisEval0, x_point, consN, weight, ...
                    fixed_cfg_with_rhs(cfg.fixed_operator, op, right, eta0));
                fprintf('  [DDAD] short fixed-operator: loss=%.4e, rel_res=%.4e, iter=%d\n', fiD.loss, fiD.rel_res, fiD.iter);
            else
                [weight, nwD] = NewtonSolver(problem, basisEval0, x_point, consN, weight, cfg.newton_ddad);
                fprintf('  [DDAD] short Newton: rel_res=%.4e, iter=%d\n', nwD.rel_res, nwD.iter);
            end

            % (b) train r with the current solution values as target (subsample)
            idxTr = 1:stepD:nAllD;
            X_tr = x_point(idxTr, :);
            u_target = BasisFunction(family, term0, X_tr, 0, 0) * weight;
            rawD = struct('family', family, 'w_basis', w_basis, 'b_basis', b_basis, 'centers', b_basis);
            [r_new, mseD, ~] = ADRaNN_train_r_analytic('value', X_tr, u_target, rawD, r, cfg.ddad_opts, 1e-8);
            fprintf('  [DDAD] r=[%s] -> r=[%s] (MSE=%.4e)\n', ...
                num2str(r.', '%.6g '), num2str(r_new.', '%.6g '), mseD);
            appendRunLog(runDir, sprintf('[DDAD] r=[%s] -> r=[%s] MSE=%.4e', ...
                num2str(r.', '%.6g '), num2str(r_new.', '%.6g '), mseD));

            % (c) rebuild basis with the new r and warm-start the
            %     coefficients via bestInitSVD (value projection vs
            %     Newton-linearized solve, pick the smaller residual)
            r = r_new(:);
            w = r .* w_basis;
            b = -sum(b_basis .* w, 1);
            term_self = term0;                       % old basis carries the current solution
            term0 = struct('w', w, 'b', b);
            basisEvalNewD = @(ord, dir, X) BasisFunction(family, term0, X, ord, dir);
            solEvalOldD  = @(ord, dir, X) BasisFunction(family, term_self, X, ord, dir) * weight;
            consD = tryProblemAction(problem, 'constraints', []);
            if is_fixed_operator_enabled(cfg)
                % DDAD is a value/ridge fit of the current numerical solution;
                % use it directly as the warm start in the rebuilt basis.
                X_fit_D = [X_tr; x_0];
                y_fit_D = [solEvalOldD(0, 0, X_tr); solEvalOldD(0, 0, x_0)];
                Phi_fit_D = BasisFunction(family, term0, X_fit_D, 0, 0);
                weight = YY_utils('solve_af', Phi_fit_D, y_fit_D, GPU, QR);
                fprintf('  [DDAD] re-init: direct value/ridge fit in rebuilt basis\n');
                appendRunLog(runDir, '[DDAD] re-init: direct value/ridge fit in rebuilt basis');
            else
                etaD = getfield_default_local(cfg.newton_ddad, 'eta_bc', eta0);
                [weight, initInfoD] = bestInitSVD(problem, basisEvalNewD, solEvalOldD, consD, ...
                    x_point, etaD, cfg.newton_ddad, GPU, QR);
                fprintf('  [DDAD] re-init: use=%s, rel_value=%.4e, rel_linearized=%.4e\n', ...
                    initInfoD.method, initInfoD.rel_value, initInfoD.rel_linearized);
                appendRunLog(runDir, sprintf('[DDAD] re-init: use=%s, rel_value=%.4e, rel_linearized=%.4e', ...
                    initInfoD.method, initInfoD.rel_value, initInfoD.rel_linearized));
            end

            % (d) short Newton again on the rebuilt basis
            basisEval0 = @(ord, dir, X) BasisFunction(family, term0, X, ord, dir);
            if is_fixed_operator_enabled(cfg)
                [weight, fiD2] = FixedOperatorSolver(problem, basisEval0, x_point, consN, weight, ...
                    fixed_cfg_with_rhs(cfg.fixed_operator, op, right, eta0));
                fprintf('  [DDAD] re-fit + fixed-operator: loss=%.4e, rel_res=%.4e, iter=%d\n', fiD2.loss, fiD2.rel_res, fiD2.iter);
                appendRunLog(runDir, sprintf('[DDAD] re-fit + fixed-operator: loss=%.4e rel_res=%.4e iter=%d', fiD2.loss, fiD2.rel_res, fiD2.iter));
            else
                [weight, nwD2] = NewtonSolver(problem, basisEval0, x_point, consN, weight, cfg.newton_ddad);
                fprintf('  [DDAD] re-fit + Newton: rel_res=%.4e, iter=%d\n', nwD2.rel_res, nwD2.iter);
                appendRunLog(runDir, sprintf('[DDAD] re-fit + Newton: rel_res=%.4e, iter=%d', nwD2.rel_res, nwD2.iter));
            end

            % Optional second DDAD outer update. The initial r0 is not
            % counted here: ddad_max_outer=2 means r0 -> r1 -> r2.
            if is_fixed_operator_enabled(cfg) && ...
                    getfield_default_local(cfg.fixed_operator, 'ddad_max_outer', 1) >= 2
                term_self2 = term0;
                weight_self2 = weight;
                u_target2 = BasisFunction(family, term_self2, X_tr, 0, 0) * weight_self2;
                [r_new2, mseD2, ~] = ADRaNN_train_r_analytic('value', X_tr, u_target2, rawD, r, cfg.ddad_opts, 1e-8);
                fprintf('  [DDAD-2] r=[%s] -> r=[%s] (MSE=%.4e)\n', ...
                    num2str(r.', '%.6g '), num2str(r_new2.', '%.6g '), mseD2);
                appendRunLog(runDir, sprintf('[DDAD-2] r=[%s] -> r=[%s] MSE=%.4e', ...
                    num2str(r.', '%.6g '), num2str(r_new2.', '%.6g '), mseD2));

                r = r_new2(:);
                w = r .* w_basis;
                b = -sum(b_basis .* w, 1);
                term0 = struct('w', w, 'b', b);
                basisEval0 = @(ord, dir, X) BasisFunction(family, term0, X, ord, dir);
                solEvalOld2 = @(ord, dir, X) BasisFunction(family, term_self2, X, ord, dir) * weight_self2;
                X_fit_D2 = [X_tr; x_0];
                y_fit_D2 = [solEvalOld2(0, 0, X_tr); solEvalOld2(0, 0, x_0)];
                Phi_fit_D2 = BasisFunction(family, term0, X_fit_D2, 0, 0);
                weight = YY_utils('solve_af', Phi_fit_D2, y_fit_D2, GPU, QR);
                [weight, fiD3] = FixedOperatorSolver(problem, basisEval0, x_point, consN, weight, ...
                    fixed_cfg_with_rhs(cfg.fixed_operator, op, right, eta0));
                fprintf('  [DDAD-2] re-fit + fixed-operator: loss=%.4e, rel_res=%.4e, iter=%d\n', ...
                    fiD3.loss, fiD3.rel_res, fiD3.iter);
                appendRunLog(runDir, sprintf('[DDAD-2] re-fit + fixed-operator: loss=%.4e rel_res=%.4e iter=%d', ...
                    fiD3.loss, fiD3.rel_res, fiD3.iter));
            end
        end
    end

    % error evaluation points (plotting grid)
    rng(cfg.seed);   % eval grid stream; also sets the continuing stream for SFR/RBE
    pp = cfg.output.pp;
    evalInfo = tryProblemAction(problem, 'eval_grid', pp);
    if isempty(evalInfo)
        xp1 = a1:(a2 - a1)/pp:a2;
        xp2 = a3:(a4 - a3)/pp:a4;
        [Xg, Yg] = meshgrid(xp1, xp2);
        pp_points = [Xg(:), Yg(:)];
        evalMask = [];
    else
        xp1 = evalInfo.x;
        xp2 = evalInfo.y;
        pp_points = evalInfo.X;      % only in-domain points
        evalMask = evalInfo.mask;
    end
    ex = ProblemFunction('derivatives', pp_points, problem);

    pred_u0  = BasisFunction(family, term0, pp_points, 0, 0) * weight;
    pred_ux0 = BasisFunction(family, term0, pp_points, 1, 1) * weight;
    pred_uy0 = BasisFunction(family, term0, pp_points, 1, 2) * weight;
    sol_initial = struct('u', pred_u0, 'ux', pred_ux0, 'uy', pred_uy0);
    m_initial = computeErrors(pred_u0, pred_ux0, pred_uy0, ex, xp1, xp2);
    fprintf('  [Initial] Rel.L2=%.4e Abs.L2=%.4e Rel.H1=%.4e Abs.H1=%.4e\n', ...
        m_initial.rel_L2, m_initial.abs_L2, m_initial.rel_H1, m_initial.abs_H1);
    appendRunLog(runDir, sprintf('[Initial] Rel.L2=%.4e Abs.L2=%.4e Rel.H1=%.4e Abs.H1=%.4e', ...
        m_initial.rel_L2, m_initial.abs_L2, m_initial.rel_H1, m_initial.abs_H1));

    if cfg.output.save_data
        saveSnapshot(runDir, '01_initial_global_RaNN', xp1, xp2, ...
            pred_u0, ex.u, ...
            struct('stage', 'initial global RaNN', 'r', r, 'neuron', neuron), ...
            m_initial, evalMask);
    end
    tAfterInitial = toc(tStart);   % elapsed up to end of PDAD + Initial (no figures)

    %% ==================== shared context ====================
    SAB_eps = 1e-12;
    orth_cfg.mode = 'rel_tol';
    orth_cfg.rel_tol = cfg.solver.orth_rel_tol;   % paper tau_svd = 1e-10; set 0 to disable truncation
    orth_cfg.fixed_rank = 1200;
    orth_cfg.energy_ratio = 0.99999999;

    ctx = struct();
    ctx.problem = problem;
    ctx.a1 = a1; ctx.a2 = a2; ctx.a3 = a3; ctx.a4 = a4;
    ctx.N = N; ctx.neuron = neuron; ctx.neuron_add = neuron_add;
    ctx.GPU = GPU; ctx.QR = QR; ctx.r = r;
    ctx.x_point = x_point; ctx.x_0 = x_0;
    ctx.right = right; ctx.y0_exact = y0_exact; ctx.eta0 = eta0;
    ctx.neumann_info = neumann_info;
    ctx.eta_neumann = eta_neumann;
    ctx.w = w; ctx.b = b; ctx.b_basis = b_basis; ctx.weight = weight;
    % Keep the evaluation reference only for a posteriori diagnostics and
    % optional plotting; SFR/RBE decisions use residuals from the PDE data.
    ctx.pp_points = pp_points; ctx.pred_out_ex = ex.u;
    ctx.xp1 = xp1; ctx.xp2 = xp2;
    ctx.SAB_eps = SAB_eps; ctx.orth_cfg = orth_cfg;
    if isfield(cfg,'output') && isfield(cfg.output,'detailed_residuals') && ...
            cfg.output.detailed_residuals
        ctx.detail_round_callback = @(ctx_cb,e_cb) ...
            appendDetailedRBERound_local(runDir,ctx_cb,e_cb,ex,xp1,xp2);
    end
    insideProbe = tryProblemAction(problem, 'inside', [-0.5, 0.5]);
    if ~isempty(insideProbe)
        ctx.inside_fn = @(X) ProblemFunction('inside', X, problem);
    end

    if isfield(cfg,'output') && isfield(cfg.output,'detailed_residuals') && ...
            cfg.output.detailed_residuals
        appendDetailedStage_local(runDir,'Initial',ctx,m_initial);
    end

    diagS = struct('G_SAB', 0, 'num_region', 0);   % default when SFR skipped
    if isfield(cfg, 'run_sfr') && ~cfg.run_sfr
        % ---- SFR skipped: final = after initial AD solve (AD only variant) ----
        sol_sfr = sol_initial;
        m_sfr = m_initial;
        tAfterSFR = tAfterInitial;   % SFR takes no time
        fprintf('  [SFR] skipped (run_sfr=false); AD only\n');
        appendRunLog(runDir, '[SFR] skipped (run_sfr=false); AD only');
    else
    %% ==================== SFR (structure-aware redistribution, best scale selection) ====================
    sab_cfg = cfg.sfr.cfg;
    if cfg.rng_reset_each_stage, rng(cfg.seed); end
    diagS = SFR_module('diagnose', ctx, sab_cfg);

    if diagS.G_SAB
        sab_ref_cfg = cfg.sfr.ref;
        if ~isfield(sab_ref_cfg,'jitter_seed_base')
            sab_ref_cfg.jitter_seed_base = cfg.seed;
        end
        if cfg.rng_reset_each_stage, rng(cfg.seed); end
        ctx = SFR_module('refresh', ctx, diagS, sab_ref_cfg);

        % Optional fixed-operator or Newton refinement after the SFR refresh.
        if is_fixed_operator_enabled(cfg) && isfield(ctx, 'use_orth_basis') && ctx.use_orth_basis == 1
            consN = ProblemFunction('constraints', [], problem);
            termS = struct('w', ctx.w, 'b', ctx.b);
            basisEvalS = @(ord, dir, X) BasisFunction('tanh', termS, X, ord, dir) * ctx.T_current;
            [ctx.gamma_current, fiS] = FixedOperatorSolver(problem, basisEvalS, x_point, consN, ctx.gamma_current, ...
                fixed_cfg_with_rhs(cfg.fixed_operator, op, right, eta0));
            fprintf('  [FixedOp] after SFR: loss=%.4e rel_res=%.4e iter=%d\n', fiS.loss, fiS.rel_res, fiS.iter);
            appendRunLog(runDir, sprintf('[FixedOp] after SFR: loss=%.4e rel_res=%.4e iter=%d', fiS.loss, fiS.rel_res, fiS.iter));
        elseif isfield(cfg, 'run_newton_sfr') && cfg.run_newton_sfr && ...
                isfield(ctx, 'use_orth_basis') && ctx.use_orth_basis == 1
            consN = tryProblemAction(problem, 'constraints', []);
            if ~isempty(consN)
                termS = struct('w', ctx.w, 'b', ctx.b);
                basisEvalS = @(ord, dir, X) BasisFunction('tanh', termS, X, ord, dir) * ctx.T_current;
                [ctx.gamma_current, nwS] = NewtonSolver( ...
                    problem, basisEvalS, x_point, consN, ctx.gamma_current, cfg.newton);
                fprintf('  [Newton] after SFR: rel_res=%.4e, iter=%d\n', nwS.rel_res, nwS.iter);
                appendRunLog(runDir, sprintf('[Newton] after SFR: rel_res=%.4e, iter=%d', nwS.rel_res, nwS.iter));
            end
        end
    end
    sol_sfr = evalCurrentSolution(ctx, pp_points);
    m_sfr = computeErrors(sol_sfr.u, sol_sfr.ux, sol_sfr.uy, ex, xp1, xp2);
    fprintf('  [SFR] G_SAB=%d accept=%d Rel.L2=%.4e Abs.L2=%.4e Rel.H1=%.4e Abs.H1=%.4e\n', ...
        diagS.G_SAB, getfield_default_local(ctx, 'SAB_accept', 0), ...
        m_sfr.rel_L2, m_sfr.abs_L2, m_sfr.rel_H1, m_sfr.abs_H1);

    sfrLines = {sprintf('[SFR] G_SAB=%d accept=%d Rel.L2=%.4e Abs.L2=%.4e Rel.H1=%.4e Abs.H1=%.4e', ...
        diagS.G_SAB, getfield_default_local(ctx, 'SAB_accept', 0), ...
        m_sfr.rel_L2, m_sfr.abs_L2, m_sfr.rel_H1, m_sfr.abs_H1)};
    if isfield(ctx,'SAB_E_pred_old') && isfield(ctx,'SAB_E_pred_new')
        sfrLines{end+1} = sprintf('  safeguard: old=%.4e new=%.4e threshold=%.4e margin=%.4e accepted=%d', ...
            ctx.SAB_E_pred_old,ctx.SAB_E_pred_new, ...
            getfield_default_local(ctx,'SAB_accept_threshold',NaN), ...
            getfield_default_local(ctx,'SAB_accept_margin',NaN), ...
            getfield_default_local(ctx,'SAB_accept',0)); %#ok<AGROW>
    end
    if isfield(diagS, 'num_region')
        sfrLines{end+1} = sprintf('  regions=%d m_S=%g m_B=%g', ...
            diagS.num_region, getfield_default_local(diagS, 'm_S_target', NaN), ...
            getfield_default_local(diagS, 'm_B_target', NaN)); %#ok<AGROW>
    end
    if isfield(diagS, 'partition_G')
        sfrLines{end+1} = sprintf('  global SFR G=[%s] target=[%s] m_S=[%s] m_B=[%s]', ...
            num2str(double(diagS.partition_G(:).'), '%d '), ...
            num2str(diagS.partition_target(:).', '%d '), ...
            num2str(diagS.partition_m_S(:).', '%d '), ...
            num2str(diagS.partition_m_B(:).', '%d ')); %#ok<AGROW>
    end
    if isfield(diagS, 'multi_structure_enabled')
        sfrLines{end+1} = sprintf('  multiple_global_components=%d', ...
            diagS.multi_structure_enabled); %#ok<AGROW>
    end
    if isfield(ctx, 'r_opt_region') && ~isempty(ctx.r_opt_region)
        MSEr = getfield_default_local(ctx, 'MSE_region', []);
        for rr = 1:size(ctx.r_opt_region, 2)
            if numel(MSEr) >= rr
                msestr = sprintf(' MSE=%.4e', MSEr(rr));
            else
                msestr = '';
            end
            sfrLines{end+1} = sprintf('  region %d r=[%s]%s', rr, ...
                num2str(ctx.r_opt_region(:, rr).', '%.6g '), msestr); %#ok<AGROW>
        end
    end
    if isfield(ctx, 'r_opt_B') && ~isempty(ctx.r_opt_B)
        sfrLines{end+1} = sprintf('  background r=[%s]', ...
            num2str(ctx.r_opt_B(:).', '%.6g ')); %#ok<AGROW>
    end
    appendRunLog(runDir, strjoin(sfrLines, newline));
    if isfield(cfg,'output') && isfield(cfg.output,'detailed_residuals') && ...
            cfg.output.detailed_residuals
        appendDetailedStage_local(runDir,'SFR',ctx,m_sfr);
    end

    if cfg.output.save_data
        saveSnapshot(runDir, '02_after_SFR', xp1, xp2, ...
            sol_sfr.u, ex.u, ...
            struct('stage', 'after SFR', 'G_SAB', diagS.G_SAB, ...
                   'SAB_accept', getfield_default_local(ctx, 'SAB_accept', 0)), ...
            m_sfr, evalMask);
        saveSFRData(runDir, ctx, diagS);
    end
    tAfterSFR = toc(tStart);      % elapsed up to end of SFR (no figures)
    end

    if isfield(cfg, 'run_rbe') && ~cfg.run_rbe
        % ---- RBE skipped: final solution = after SFR (AD+SFR variant) ----
        sol_final = sol_sfr;
        m_final = m_sfr;
        fprintf('  [RBE] skipped (run_rbe=false); final = after SFR\n');
        appendRunLog(runDir, '[RBE] skipped (run_rbe=false); final = after SFR');
        if cfg.output.save_data
            saveSnapshot(runDir, '03_after_RBE', xp1, xp2, ...
                sol_final.u, ex.u, ...
                struct('stage', 'final (RBE skipped, AD+SFR only)', ...
                       'orb_block_count', 0), m_final, evalMask);
            saveSummary(runDir, cfg, m_initial, m_sfr, m_final, diagS, ctx);
        end
        tAfterRBE = toc(tStart);      % elapsed up to end of SFR (no figures)
    else
    %% ==================== RBE (residual-driven block enrichment, best scale selection) ====================
    score_cfg = cfg.rbe;
    if is_fixed_operator_enabled(cfg)
        % Let the RBE acceptance test refine each trial augmented space with
        % the same fixed operator used by the final fixed-operator stage.
        % The RBE module only receives cfg.rbe, so pass the operator, RHS,
        % and boundary weight explicitly.
        score_cfg.fixed_operator = fixed_cfg_with_rhs(cfg.fixed_operator, op, right, eta0);
    end
    if isfield(ctx, 'last_r_opt') && ~isempty(ctx.last_r_opt)
        score_cfg.r0 = ctx.last_r_opt;
    elseif isfield(ctx, 'r_opt_region') && ~isempty(ctx.r_opt_region)
        score_cfg.r0 = mean(ctx.r_opt_region, 2);
    else
        score_cfg.r0 = r;
    end
    if cfg.rng_reset_each_stage, rng(cfg.seed); end
    ctx = RBE_FamilyScore_module(ctx, score_cfg);

    % Optional nonlinear refinement of the final RBE coefficients.  NewtonSolver
    % applies its own line search using the original nonlinear residual.
    if is_fixed_operator_enabled(cfg) && isfield(ctx, 'orb_initialized') && ctx.orb_initialized == 1
        consN = ProblemFunction('constraints', [], problem);
        basisEvalOrb = @(ord, dir, X) RBE_Eval('terms', ctx.orb_terms, X, ord, dir);
        fixed_cfg_final = fixed_cfg_with_rhs(cfg.fixed_operator, op, right, eta0);
        if isfield(ctx,'fixed_operator_svd_state') && ...
                ~isempty(ctx.fixed_operator_svd_state)
            fixed_cfg_final.svd_state = ctx.fixed_operator_svd_state;
        end
        [ctx.gamma_orb_current, fiF] = FixedOperatorSolver(problem, basisEvalOrb, x_point, consN, ctx.gamma_orb_current, ...
            fixed_cfg_final);
        ctx.fixed_operator_svd_state = fiF.svd_state;
        fprintf('  [FixedOp] final refine: loss=%.4e rel_res=%.4e iter=%d factor=%s\n', ...
            fiF.loss, fiF.rel_res, fiF.iter, fiF.factorization);
        appendRunLog(runDir, sprintf('[FixedOp] final refine: loss=%.4e rel_res=%.4e iter=%d factor=%s', ...
            fiF.loss, fiF.rel_res, fiF.iter, fiF.factorization));
    elseif isfield(cfg, 'run_newton') && cfg.run_newton && ...
            isfield(ctx, 'orb_initialized') && ctx.orb_initialized == 1
        consN = tryProblemAction(problem, 'constraints', []);
        if ~isempty(consN)
            basisEvalOrb = @(ord, dir, X) RBE_Eval('terms', ctx.orb_terms, X, ord, dir);
            [ctx.gamma_orb_current, nwF] = NewtonSolver( ...
                problem, basisEvalOrb, x_point, consN, ctx.gamma_orb_current, cfg.newton);
            fprintf('  [Newton] final refine: rel_res=%.4e, iter=%d\n', ...
                nwF.rel_res, nwF.iter);
            appendRunLog(runDir, sprintf('[Newton] final refine: rel_res=%.4e, iter=%d', ...
                nwF.rel_res, nwF.iter));
        end
    end

    sol_final = evalCurrentSolution(ctx, pp_points);
    m_final = computeErrors(sol_final.u, sol_final.ux, sol_final.uy, ex, xp1, xp2);
    fprintf('  [RBE] blocks=%d Rel.L2=%.4e Abs.L2=%.4e Rel.H1=%.4e Abs.H1=%.4e\n', ...
        getfield_default_local(ctx, 'orb_block_count', 0), ...
        m_final.rel_L2, m_final.abs_L2, m_final.rel_H1, m_final.abs_H1);

    rbeLines = {sprintf('[RBE] blocks=%d Rel.L2=%.4e Abs.L2=%.4e Rel.H1=%.4e Abs.H1=%.4e', ...
        getfield_default_local(ctx, 'orb_block_count', 0), ...
        m_final.rel_L2, m_final.abs_L2, m_final.rel_H1, m_final.abs_H1)};
    if isfield(ctx, 'rbe_log') && ~isempty(ctx.rbe_log)
        for kk = 1:numel(ctx.rbe_log)
            e = ctx.rbe_log(kk);
            if e.accept
                rbeLines{end+1} = sprintf(['  round %d: %s m=%d r=[%s] ', ...
                    'cap=%.4e rel_res=%.4e->%.4e ', ...
                    'kappa_map=%.3e<=%.3e'], ...
                    e.round, e.family, e.m, num2str(e.r, '%.6g '), ...
                    e.cap_score, e.rel_before, e.rel_after, e.kappa, ...
                    cfg.rbe.kappa_map_max); %#ok<AGROW>
            else
                rbeLines{end+1} = sprintf(['  round %d: %s rejected ', ...
                    'rel_res=%.4e kappa_map=%.3e<=%.3e (%s)'], ...
                    e.round, e.family, e.rel_before, e.kappa, ...
                    cfg.rbe.kappa_map_max, e.reason); %#ok<AGROW>
            end
        end
    end
    appendRunLog(runDir, strjoin(rbeLines, newline));
    if isfield(cfg,'output') && isfield(cfg.output,'detailed_residuals') && ...
            cfg.output.detailed_residuals
        appendDetailedStage_local(runDir,'RBE-final',ctx,m_final);
    end

    if cfg.output.save_data
        saveSnapshot(runDir, '03_after_RBE', xp1, xp2, ...
            sol_final.u, ex.u, ...
            struct('stage', 'after RBE', ...
                   'orb_block_count', getfield_default_local(ctx, 'orb_block_count', 0)), ...
            m_final, evalMask);
        saveSummary(runDir, cfg, m_initial, m_sfr, m_final, diagS, ctx);
    end
    tAfterRBE = toc(tStart);      % elapsed up to end of RBE (no figures)
    end

    %% ==================== paper figures ====================
    if cfg.output.make_figs
        pd = struct();
        pd.xp1 = xp1; pd.xp2 = xp2;
        pd.exact = ex.u;
        pd.pred_sfr = sol_sfr.u;
        pd.pred_final = sol_final.u;
        pd.centers = getfield_default_local(ctx, 'b_basis', []);
        pd.diagS = diagS;
        pd.pred_initial = pred_u0;
        pd.mask = evalMask;
        makePaperFigures(cfg, pd);
    end
    tAfterFigs = toc(tStart);     % elapsed including figure export

    fprintf('  [Timing] PDAD+Initial=%.3fs | SFR=%.3fs | RBE=%.3fs | Figures=%.3fs | Total=%.3fs\n', ...
        tAfterInitial, tAfterSFR - tAfterInitial, tAfterRBE - tAfterSFR, ...
        tAfterFigs - tAfterRBE, tAfterFigs);
    appendRunLog(runDir, sprintf('Timing: PDAD+Initial=%.3fs SFR=%.3fs RBE=%.3fs Figures=%.3fs Total=%.3fs', ...
        tAfterInitial, tAfterSFR - tAfterInitial, tAfterRBE - tAfterSFR, ...
        tAfterFigs - tAfterRBE, tAfterFigs));
    fprintf('  [Done] Rel.L2=%.4e Abs.L2=%.4e Rel.H1=%.4e Abs.H1=%.4e   (%s, compute %.3f s, figures %.3f s)\n', ...
        m_final.rel_L2, m_final.abs_L2, m_final.rel_H1, m_final.abs_H1, ...
        runDir, tAfterRBE, tAfterFigs - tAfterRBE);
    appendRunLog(runDir, sprintf('[Done] Rel.L2=%.4e Abs.L2=%.4e Rel.H1=%.4e Abs.H1=%.4e (compute %.3f s, figures %.3f s)', ...
        m_final.rel_L2, m_final.abs_L2, m_final.rel_H1, m_final.abs_H1, ...
        tAfterRBE, tAfterFigs - tAfterRBE));
end

%% ============================================================
function out = tryProblemAction(problem, action, arg)
    % Try a generic ProblemFunction action; return [] when unsupported.
    out = [];
    try
        out = ProblemFunction(action, arg, problem);
    catch
        out = [];
    end
end

%% ============================================================
function sol = evalCurrentSolution(ctx, X)
    sol = struct();
    sol.u  = evalCurrentSolutionComponent(ctx, X, 0, 0);
    sol.ux = evalCurrentSolutionComponent(ctx, X, 1, 1);
    sol.uy = evalCurrentSolutionComponent(ctx, X, 1, 2);
end

function v = evalCurrentSolutionComponent(ctx, X, ord, dir)
    if isfield(ctx, 'orb_initialized') && ctx.orb_initialized == 1
        v = RBE_Eval('solution', ctx.orb_terms, X, ord, dir, ctx.gamma_orb_current);
    elseif isfield(ctx, 'use_orth_basis') && ctx.use_orth_basis == 1
        term_tmp = struct('w', ctx.w, 'b', ctx.b);
        v = BasisFunction('tanh', term_tmp, X, ord, dir) * ctx.T_current * ctx.gamma_current;
    else
        term_tmp = struct('w', ctx.w, 'b', ctx.b);
        v = BasisFunction('tanh', term_tmp, X, ord, dir) * ctx.weight;
    end
    v = gather(v);
end

function saveSFRData(runDir, ctx, diagS)
    sfr_data = struct();
    sfr_data.centers = gather(getfield_default_local(ctx, 'b_basis', []));
    sfr_data.r_opt_region = getfield_default_local(ctx, 'r_opt_region', []);
    sfr_data.r_opt_B = getfield_default_local(ctx, 'r_opt_B', []);
    sfr_data.MSE_region = getfield_default_local(ctx, 'MSE_region', []);
    sfr_data.MSE_B = getfield_default_local(ctx, 'MSE_B', []);
    sfr_data.SAB_accept = getfield_default_local(ctx, 'SAB_accept', 0);
    sfr_data.SAB_accept_reason = getfield_default_local(ctx, 'SAB_accept_reason', 'not evaluated');
    sfr_data.SAB_E_pred_old = getfield_default_local(ctx, 'SAB_E_pred_old', NaN);
    sfr_data.SAB_E_pred_new = getfield_default_local(ctx, 'SAB_E_pred_new', NaN);
    sfr_data.SAB_accept_threshold = getfield_default_local(ctx, 'SAB_accept_threshold', NaN);
    sfr_data.SAB_accept_margin = getfield_default_local(ctx, 'SAB_accept_margin', NaN);
    sfr_data.SAB_validation_count = getfield_default_local(ctx, 'SAB_validation_count', 0);
    sfr_data.SAB_validation_same_samples = getfield_default_local(ctx, 'SAB_validation_same_samples', false);
    d = struct();
    d.G_SAB = diagS.G_SAB;
    d.num_region = getfield_default_local(diagS, 'num_region', 0);
    d.alpha_use = getfield_default_local(diagS, 'alpha_use', []);
    d.m_S_target = getfield_default_local(diagS, 'm_S_target', []);
    d.m_B_target = getfield_default_local(diagS, 'm_B_target', []);
    if isfield(diagS, 'xx1'), d.xx1 = diagS.xx1; end
    if isfield(diagS, 'xx2'), d.xx2 = diagS.xx2; end
    if isfield(diagS, 'diag_N'), d.diag_N = diagS.diag_N; end
    if isfield(diagS, 'h'), d.h = diagS.h; end
    if isfield(diagS, 'bounds'), d.bounds = diagS.bounds; end
    if isfield(diagS, 'Nvalid'), d.Nvalid = diagS.Nvalid; end
    if isfield(diagS, 'valid_mask'), d.valid_mask = diagS.valid_mask; end
    if isfield(diagS, 'domain_labels'), d.domain_labels = diagS.domain_labels; end
    if isfield(diagS, 'quad_points'), d.quad_points = diagS.quad_points; end
    if isfield(diagS, 'quad_valid'), d.quad_valid = diagS.quad_valid; end
    if isfield(diagS, 'n_quad'), d.n_quad = diagS.n_quad; end
    if isfield(diagS, 'quad_rule'), d.quad_rule = diagS.quad_rule; end
    if isfield(diagS, 'theta_res'), d.theta_res = diagS.theta_res; end
    if isfield(diagS, 'theta_grad'), d.theta_grad = diagS.theta_grad; end
    if isfield(diagS, 'etaR'), d.etaR = diagS.etaR; end
    if isfield(diagS, 'etaG'), d.etaG = diagS.etaG; end
    if isfield(diagS, 's_res_hat'), d.s_res_hat = diagS.s_res_hat; end
    if isfield(diagS, 's_grad_hat'), d.s_grad_hat = diagS.s_grad_hat; end
    if isfield(diagS, 'structure_score'), d.structure_score = diagS.structure_score; end
    if isfield(diagS, 'eta2'), d.eta2 = diagS.eta2; end
    if isfield(diagS, 'global_dorfler_lambda'), d.global_dorfler_lambda = diagS.global_dorfler_lambda; end
    if isfield(diagS, 'global_dorfler_ok'), d.global_dorfler_ok = diagS.global_dorfler_ok; end
    if isfield(diagS, 'global_marked_count'), d.global_marked_count = diagS.global_marked_count; end
    % SFR diagnostics.  The default configuration has one global region; the
    % partition fields are retained only for backward-compatible result files.
    if isfield(diagS, 'partition_grid'), d.partition_grid = diagS.partition_grid; end
    if isfield(diagS, 'partition_id'), d.partition_id = diagS.partition_id; end
    if isfield(diagS, 'partition_raw_id'), d.partition_raw_id = diagS.partition_raw_id; end
    if isfield(diagS, 'num_partition'), d.num_partition = diagS.num_partition; end
    if isfield(diagS, 'partition_valid_count'), d.partition_valid_count = diagS.partition_valid_count; end
    if isfield(diagS, 'partition_energy'), d.partition_energy = diagS.partition_energy; end
    if isfield(diagS, 'partition_marked_energy'), d.partition_marked_energy = diagS.partition_marked_energy; end
    if isfield(diagS, 'partition_lambda'), d.partition_lambda = diagS.partition_lambda; end
    if isfield(diagS, 'partition_dorfler_ok'), d.partition_dorfler_ok = diagS.partition_dorfler_ok; end
    if isfield(diagS, 'partition_target'), d.partition_target = diagS.partition_target; end
    if isfield(diagS, 'partition_m_S'), d.partition_m_S = diagS.partition_m_S; end
    if isfield(diagS, 'partition_m_B'), d.partition_m_B = diagS.partition_m_B; end
    if isfield(diagS, 'partition_alpha_S'), d.partition_alpha_S = diagS.partition_alpha_S; end
    if isfield(diagS, 'partition_G'), d.partition_G = diagS.partition_G; end
    if isfield(diagS, 'partition_chi'), d.partition_chi = diagS.partition_chi; end
    if isfield(diagS, 'partition_cover'), d.partition_cover = diagS.partition_cover; end
    if isfield(diagS, 'partition_num_components'), d.partition_num_components = diagS.partition_num_components; end
    if isfield(diagS, 'component_energy'), d.component_energy = diagS.component_energy; end
    if isfield(diagS, 'component_res_ratio'), d.component_res_ratio = diagS.component_res_ratio; end
    if isfield(diagS, 'component_chi'), d.component_chi = diagS.component_chi; end
    if isfield(diagS, 'component_cover'), d.component_cover = diagS.component_cover; end
    if isfield(diagS, 'component_G'), d.component_G = diagS.component_G; end
    if isfield(diagS, 'multi_structure_enabled'), d.multi_structure_enabled = diagS.multi_structure_enabled; end
    sfr_data.diag = d;
    save(fullfile(runDir, '02_SFR_anchors.mat'), 'sfr_data');
    fprintf('  [save] %s\n', fullfile(runDir, '02_SFR_anchors.mat'));
end

function val = getfield_default_local(s, name, default_val)
    if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
        val = s.(name);
    else
        val = default_val;
    end
end

function tf = is_fixed_operator_enabled(cfg)
    tf = isfield(cfg,'fixed_operator') && isstruct(cfg.fixed_operator) && ...
         isfield(cfg.fixed_operator,'enabled') && cfg.fixed_operator.enabled;
end

function fc = fixed_cfg_with_rhs(fc, op, rhs, eta)
    fc.op = op; fc.rhs_int = rhs;
    if ~isfield(fc,'eta_bc') || isempty(fc.eta_bc), fc.eta_bc = eta; end
end

function appendDetailedStage_local(runDir,label,ctx,metrics)
    d = splitResidual_local(ctx);
    line = sprintf(['[%s] Rel.L2=%.4e Abs.L2=%.4e Rel.H1=%.4e Abs.H1=%.4e ', ...
        'interior_L2=%.4e boundary_L2=%.4e'], ...
        label,metrics.rel_L2,metrics.abs_L2,metrics.rel_H1,metrics.abs_H1, ...
        d.operator_L2,d.boundary_L2);
    fprintf('  %s\n',line);
    appendRunLog(runDir,line);
end

function appendDetailedRBERound_local(runDir,ctx,e,ex,xp1,xp2)
    try
        sol = evalCurrentSolution(ctx,ctx.pp_points);
        m = computeErrors(sol.u,sol.ux,sol.uy,ex,xp1,xp2);
        d = splitResidual_local(ctx);
        status = 'accepted';
        if ~e.accept, status = ['rejected:' e.reason]; end
        line = sprintf(['  round %d: %s m=%d %s rel_res=%.4e->%.4e ', ...
            'Rel.L2=%.4e Abs.L2=%.4e Rel.H1=%.4e Abs.H1=%.4e ', ...
            'interior_L2=%.4e boundary_L2=%.4e'], ...
            e.round,e.family,e.m,status,e.rel_before,e.rel_after, ...
            m.rel_L2,m.abs_L2,m.rel_H1,m.abs_H1,d.operator_L2,d.boundary_L2);
        fprintf('%s\n',line);
        appendRunLog(runDir,line);
    catch err
        warning('run_experiment:detailedLog', ...
            'Detailed RBE logging skipped for round %d: %s',e.round,err.message);
    end
end

function d = splitResidual_local(ctx)
    X = gather(ctx.x_point);
    p = ctx.problem;
    op = p.op;
    u = evalCurrentSolutionComponent(ctx,X,0,0);
    ux = evalCurrentSolutionComponent(ctx,X,1,1);
    uy = evalCurrentSolutionComponent(ctx,X,1,2);
    uxx = evalCurrentSolutionComponent(ctx,X,2,1);
    uyy = evalCurrentSolutionComponent(ctx,X,2,2);
    uxy = evalCurrentSolutionComponent(ctx,X,2,3);
    Lu = op_value_local(op.alpha0,X,u) + op_value_local(op.alpha1,X,ux) + ...
        op_value_local(op.alpha2,X,uxx) + op_value_local(op.alpha3,X,uy) + ...
        op_value_local(op.alpha4,X,uyy) + op_value_local(op.alpha5,X,uxy);
    r_op = gather(Lu(:) - ctx.right(:));

    Xb = gather(ctx.x_0);
    ub = evalCurrentSolutionComponent(ctx,Xb,0,0);
    r_bc = gather(ub(:) - ctx.y0_exact(:));
    n_op = max(1,numel(r_op));
    n_bc = max(1,numel(r_bc));
    d.operator_L2 = norm(r_op,2)/sqrt(n_op);
    d.boundary_L2 = norm(r_bc,2)/sqrt(n_bc);
end

function z = op_value_local(c,X,v)
    if isa(c,'function_handle')
        c = c(X);
    end
    c = gather(c);
    if isscalar(c)
        z = c.*v;
    else
        z = c(:).*v;
    end
end

function B = neumann_flux_matrix_local(family,term,info,eta)
%NEUMANN_FLUX_MATRIX_LOCAL Weighted natural-boundary flux rows.
    if isempty(info) || ~isstruct(info) || ~isfield(info,'X') || isempty(info.X)
        B = zeros(0,size(term.w,2));
        return;
    end
    X = info.X;
    phix = BasisFunction(family,term,X,1,1);
    phiy = BasisFunction(family,term,X,1,2);
    B = eta * info.coefficient(:) .* ...
        (info.normal(:,1).*phix + info.normal(:,2).*phiy);
end
