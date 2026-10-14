function ctx = RBE_FamilyScore_module3D(ctx,cfg)
    %RBE_FAMILYSCORE_MODULE_COMPAT  Burgers3D residual-driven RBE family enrichment.
    %
    % This is the Burgers2D version.  It does NOT use ctx.x_point / ctx.right /
    % ctx.y0_exact.  All residuals, Jacobian actions, and Newton refinements
    % are routed through Burgers2D_Newton_module3D.
    %
    % family_repeat_mode:
    %   'exclude_last' : Default. Exclude only the family selected in the
    %                    previous round, then restore it in the next round.
    %                    For example, if round 1 selects tanh, round 2 excludes
    %                    tanh. If round 2 selects ridge_gauss, round 3 restores
    %                    tanh and excludes only ridge_gauss.
    %
    %   'all'          : Exclude nothing; screen every family in every round.
    %
    % Main logic:
    %   1. Use current nonlinear residual R(u) and frozen J_ref space.
    %   2. For each candidate family, train r by the Burgers2D Newton residual equation
    %        J_ref Phi(r) alpha ~= -F_int.
    %   3. Select the best family by frozen-Jref residual-capture score.
    %   4. For the selected family, build one full candidate pool of neuron_add.
    %   5. Sort candidate columns by frozen-Jref residual capture and choose prefix.
    %   6. Append this family once and initialize by value/linearized-Burgers2D warm start.
    %   7. Run full SVD-Newton over the whole coefficient vector.
    %   8. Accept/reject and stop are decided by family-level residual improvement.
    
        if nargin < 2
            cfg = struct();
        end
        cfg = fill_score_cfg_local(cfg,ctx);
    
        if ~isfield(ctx,'jref') || isempty(ctx.jref) || ~isfield(ctx.jref,'Q')
            fprintf('\nORB: ctx.jref not found, freeze J_ref first.\n');
            ctx = Burgers2D_Newton_module3D('freeze_jacobian',ctx,cfg.jref_cfg);
        end
    
        fprintf('\n========== Burgers2D all-SVD prefix-size Family Blocks ==========\n');
        fprintf('Use existing frozen J_ref SVD for family screening and block scoring.\n');
        fprintf('family_repeat_mode = %s\n',cfg.family_repeat_mode);
    
        used_family = {};
        r_global = cfg.r0(:);
    
        % Record only the family selected in the previous round. In
        % mode='exclude_last', temporarily exclude it from this round. It is
        % restored automatically once a different family is selected.
        last_selected_family = '';
    
        for family_round = 1:cfg.max_family_rounds
    
            [R,~,parts] = Burgers2D_Newton_module3D('assemble',ctx,cfg.newton_cfg);
            rel_old = parts.rel_res;
            fprintf('\n========== Block family round %d / %d ==========\n', ...
                family_round,cfg.max_family_rounds);
            fprintf('Current rel_res=%.6e\n',rel_old);
    
            if rel_old < cfg.tau_block_start
                fprintf('Stop family search: current rel_res %.6e < tau_block_start %.6e.\n', ...
                    rel_old,cfg.tau_block_start);
                break;
            end
    
            candidate_families = choose_candidate_families_local(cfg,last_selected_family);
    
            if isempty(candidate_families)
                fprintf('No candidate family remains in this round; stop ORB.\n');
                break;
            end
    
            if strcmpi(cfg.family_repeat_mode,'exclude_last')
                if isempty(last_selected_family)
                    fprintf('Candidate mode: exclude_last; no family was selected in the previous round, so use the full candidate set.\n');
                else
                    fprintf('Candidate mode: exclude_last; temporarily exclude the previously selected family: %s\n', ...
                        last_selected_family);
                end
            elseif strcmpi(cfg.family_repeat_mode,'all')
                fprintf('Candidate mode: all; screen every family in every round.\n');
            end
    
            fprintf('Candidate families: ');
            for ii = 1:numel(candidate_families)
                fprintf('%s ',candidate_families{ii});
            end
            fprintf('\n');
    
            % current interior residual for sampling and r-training
            F_int = parts.F_int(:);
            p_int = residual_sampling_prob_local(F_int,cfg,ctx.SAB_eps);
            X_int = ctx.grid.XT_int;
    
            % Frozen numerical state for Burgers2D residual-training.  This is
            % the current iterate at which the Jacobian is frozen, not an
            % exact/reference solution.
            if isfield(ctx,'jref') && isfield(ctx.jref,'u_ref_int') && ~isempty(ctx.jref.u_ref_int)
                u_ref_all  = ctx.jref.u_ref_int(:);
                ux_ref_all = ctx.jref.ux_ref_int(:);
                uy_ref_all = ctx.jref.uy_ref_int(:);
            else
                u_ref_all  = parts.u_int(:);
                ux_ref_all = parts.ux_int(:);
                uy_ref_all = parts.uy_int(:);
            end
    
            %% ------------------------------------------------------------
            % 1. Screening over candidate families with neuron_score neurons
            %% ------------------------------------------------------------
            best_id = 0;
            best_score = -inf;
            screen = struct([]);
    
            for ff = 1:numel(candidate_families)
    
                family = candidate_families{ff};
                rng(7000 + 100*family_round + ff);
    
                m_scr = cfg.neuron_score;
                idx_center = YY_utils3D('sample_from_prob',p_int,m_scr);
                centers = gather(X_int(idx_center,:).');
    
                raw = Burgers2D_Newton_module3D('make_raw_family',family,m_scr,centers,cfg);
    
                % Paper rule (C.1): N_R^max = min{N_eq,R,
                % ceil(m_R*log(1+m_R))}.  During family screening the
                % low-dimensional block has m_R = m_scr.
                n_train = rbe_training_cap_3d_local( ...
                    m_scr,numel(p_int),cfg);
                idx_train = YY_utils3D('sample_from_prob',p_int,n_train);
    
                X_train  = gather(X_int(idx_train,:));
                target   = gather(-F_int(idx_train));
                u_ref    = gather(u_ref_all(idx_train));
                ux_ref   = gather(ux_ref_all(idx_train));
                uy_ref   = gather(uy_ref_all(idx_train));
    
                [r_scr,mse_scr,term_scr] = Burgers2D_Newton_module3D( ...
                    'train_r_residual', ...
                    X_train,target,u_ref,ux_ref,uy_ref, ...
                    raw,r_global,cfg.screen_adam_opts,ctx.problem,family,cfg.lambda);
    
                [score_scr,rank_scr,kappa_scr,valid_scr] = score_term_local(ctx,term_scr,R,cfg);
    
                screen(ff).family = family; %#ok<AGROW>
                screen(ff).score  = score_scr;
                screen(ff).rank   = rank_scr;
                screen(ff).kappa  = kappa_scr;
                screen(ff).valid  = valid_scr;
                screen(ff).r      = r_scr(:);
                screen(ff).MSE    = mse_scr;
    
                fprintf('screen %s: score=%.4e, rank=%d, kappa=%.3e, r=[%s ], MSE=%.4e\n', ...
                    family,score_scr,rank_scr,kappa_scr,sprintf(' %.4f',r_scr(:)),mse_scr);
    
                if valid_scr && score_scr > best_score
                    best_score = score_scr;
                    best_id = ff;
                end
            end
    
            if best_id == 0
                fprintf('No candidate family passed screening; stop ORB.\n');
                break;
            end
    
            best_family = screen(best_id).family;
            r_opt = screen(best_id).r(:);
    
            % Record the family selected in this round. In exclude_last mode,
            % the next round temporarily excludes it, and the following round
            % restores it automatically.
            last_selected_family = best_family;
    
            fprintf('Selected family = %s, screen_score=%.4e, r0=[%s ]\n', ...
                best_family,best_score,sprintf(' %.4f',r_opt(:)));
    
            %% ------------------------------------------------------------
            % 2. Build one full candidate pool; reuse screening r
            %% ------------------------------------------------------------
            m_full = cfg.neuron_add;
            rng(9000 + 1000*family_round);
            idx_center = YY_utils3D('sample_from_prob',p_int,m_full);
            centers = gather(X_int(idx_center,:).');
    
            raw_full = Burgers2D_Newton_module3D('make_raw_family',best_family,m_full,centers,cfg);
    
            % Important: r has been determined in the screening stage.
            % Do not run Adam again here unless explicitly requested.
            if getfield_default_local(cfg,'retrain_full_block',false)
    
                % If full-block scale retraining is explicitly enabled,
                % the trained block has m_R = m_full.
                n_train = rbe_training_cap_3d_local( ...
                    m_full,numel(p_int),cfg);
                idx_train = YY_utils3D('sample_from_prob',p_int,n_train);
    
                X_train  = gather(X_int(idx_train,:));
                target   = gather(-F_int(idx_train));
                u_ref    = gather(u_ref_all(idx_train));
                ux_ref   = gather(ux_ref_all(idx_train));
                uy_ref   = gather(uy_ref_all(idx_train));
    
                [r_opt,~,term_full] = Burgers2D_Newton_module3D( ...
                    'train_r_residual', ...
                    X_train,target,u_ref,ux_ref,uy_ref, ...
                    raw_full,r_opt,cfg.block_adam_opts,ctx.problem,best_family,cfg.lambda);
            else
                term_full = Burgers2D_Newton_module3D('build_term_from_r',best_family,raw_full,r_opt);
            end
    
            % ============================================================
            % Rank the shared neuron_add candidate pool by the frozen-Jref
            % residual-capture score of each column, then take a prefix.
            %
            % score_j = ((c_j_perp' * R)^2)/(||c_j_perp||^2 + eps)
            % c_j_perp = c_j - Q_old*(Q_old'*c_j)
            % ============================================================
            [term_full,sort_info] = sort_term_by_capture_local(ctx,term_full,R,cfg);
    
            fprintf('Candidate pool ranked by frozen-Jref per-column residual capture: best=%.4e, median=%.4e, worst=%.4e\n', ...
                sort_info.best,sort_info.median,sort_info.worst);
    
            %% ------------------------------------------------------------
            % 3. Prefix-size selection
            %% ------------------------------------------------------------
            [term_selected,prefix_info] = select_prefix_local(ctx,term_full,R,cfg);
    
            if ~prefix_info.valid
                fprintf('%s family reject before Newton: prefix score invalid.\n',best_family);
                if cfg.stop_when_reject
                    break;
                else
                    continue;
                end
            end
    
            fprintf('prefix selected m=%d, cap_score=%.4e, full_score=%.4e, rank=%d, kappa=%.3e\n', ...
                prefix_info.m,prefix_info.score,prefix_info.full_score,prefix_info.rank,prefix_info.kappa);

            if prefix_info.score < cfg.tau_stop_score
                fprintf('Stopping family selection: final prefix cap_score=%.4e < tau_stop_score=%.4e; the block is not added.\n', ...
                    prefix_info.score,cfg.tau_stop_score);
                if cfg.stop_when_reject
                    break;
                else
                    continue;
                end
            end

            %% ------------------------------------------------------------
            % 4. Append family once, incremental warm-start for fixed operator,
            % or legacy full SVD initialization for the Newton baseline.
            %% ------------------------------------------------------------
            ctx_before = ctx;
            rel_before = rel_old;
    
            terms_aug = ctx.terms;
            terms_aug{end+1} = term_selected;
            m_new = Burgers2D_Newton_module3D('term_num_cols',term_selected); %#ok<NASGU>
    
            if isfield(cfg,'incremental_init') && cfg.incremental_init
                % Incremental frozen-Jref path: solve the small augmented
                % SVD problem for both the old coefficient correction and
                % the new block.  This is the coefficient-space analogue of
                % the orthogonal residual projection used by the capture
                % score; no full augmented SVD is formed.
                [coef_aug,init_info] = Burgers2D_Newton_module3D( ...
                    'jref_augmented_init',ctx,term_selected,R,cfg.jref_cfg);
                fprintf('Family incremental init: block=%d, rel_res %.6e -> %.6e\n', ...
                    init_info.block_size,init_info.rel_before,init_info.rel_after);
            else
                % Legacy Newton path: compare value and linearized global
                % initializations on the complete augmented basis.
                U_current = Burgers2D_Newton_module3D('grid_solution',ctx);
                [coef_aug,init_info] = Burgers2D_Newton_module3D('best_burgers2d_init_svd', ...
                    ctx,terms_aug,ctx.grid.XT_grid,U_current(:),cfg.family_full_newton_cfg);

                fprintf('Family init compare: value rel_res=%.6e, linearized rel_res=%.6e, use=%s\n', ...
                    init_info.rel_value,init_info.rel_linearized,init_info.method);
            end
    
            ctx_try = ctx;
            ctx_try.terms = terms_aug;
            ctx_try.coef = coef_aug;
            ctx_try.weight = coef_aug;
    
            % Small SVD update of frozen-Jref space. This is for subsequent
            % scoring/orthogonalization and avoids a new big SVD.
            ctx_try = Burgers2D_Newton_module3D('append_jref_block',ctx_try,term_selected,cfg.jref_cfg);
    
            if isfield(ctx_try,'jref_append') && ~ctx_try.jref_append.valid
                fprintf('%s family reject: jref append rank=0.\n',best_family);
                if cfg.stop_when_reject
                    break;
                else
                    continue;
                end
            end
    
            fprintf('\n--- Family %s full Newton refine: before rel_res = %.6e ---\n', ...
                best_family,rel_before);
    
            ctx_try = Burgers2D_Newton_module3D('relax_full',ctx_try,cfg.family_full_newton_cfg);
    
            rel_after = ctx_try.newton_info.rel_res;
            family_improve = (rel_before - rel_after)/(rel_before + eps);
            fprintf('--- Family %s refine done: after rel_res = %.6e, family_improve=%.4e ---\n', ...
                best_family,rel_after,family_improve);
    
            %% ------------------------------------------------------------
            % 5. Family-level accept/reject and stopping
            %% ------------------------------------------------------------
            if rel_after < rel_before && family_improve >= cfg.family_tol_accept
    
                ctx = ctx_try;
                r_global = r_opt;
                ctx.last_r_opt = r_opt;
                used_family{end+1} = best_family; %#ok<AGROW>
    
                fprintf('%s family accept: rel %.6e -> %.6e, family_improve=%.4e\n', ...
                    best_family,rel_before,rel_after,family_improve);
    
            else
    
                ctx = ctx_before;
                fprintf('%s family reject: rel %.6e -> %.6e, family_improve=%.4e < %.4e\n', ...
                    best_family,rel_before,rel_after,family_improve,cfg.family_tol_accept);
                if cfg.stop_when_reject
                    break;
                else
                    continue;
                end
            end
    
            if cfg.stop_when_small_improve && family_improve < cfg.tol_improve
                fprintf('Total family improvement is too small: family_improve=%.4e < tol_improve=%.4e; stop subsequent families.\n', ...
                    family_improve,cfg.tol_improve);
                break;
            end
        end
    
        fprintf('\nSelected family order: ');
        for k = 1:numel(used_family)
            fprintf('%s ',used_family{k});
        end
        fprintf('\n');
    end

    %% ============================================================
    % defaults
    %% ============================================================
    function cfg = fill_score_cfg_local(cfg,ctx)
        if ~isfield(cfg,'families')
            cfg.families = {'tanh','ridge_gauss','sin','radial_gauss','mexican_hat'};
        end
    
        % family_repeat_mode:
        %   'exclude_last' : Default. Exclude only the family selected in the
        %                    previous round and restore it in the next round.
        %   'all'          : Exclude nothing; include every family each round.
        %
        % Backward compatibility with the legacy remove_used_family field:
        %   remove_used_family = false -> all
        %   remove_used_family = true  -> exclude_last
        if ~isfield(cfg,'family_repeat_mode') || isempty(cfg.family_repeat_mode)
            if isfield(cfg,'remove_used_family')
                if cfg.remove_used_family
                    cfg.family_repeat_mode = 'exclude_last';
                else
                    cfg.family_repeat_mode = 'all';
                end
            else
                cfg.family_repeat_mode = 'exclude_last';
            end
        end
    
        valid_modes = {'exclude_last','all'};
        if ~any(strcmpi(cfg.family_repeat_mode,valid_modes))
            error('cfg.family_repeat_mode must be ''exclude_last'' or ''all''. Current value: %s', ...
                cfg.family_repeat_mode);
        end
    
        if ~isfield(cfg,'max_family_rounds'), cfg.max_family_rounds = 3; end
        if ~isfield(cfg,'neuron_score'), cfg.neuron_score = 300; end
        if ~isfield(cfg,'neuron_add'), cfg.neuron_add = ctx.neuron_add; end
        if ~isfield(cfg,'m_block_max'), cfg.m_block_max = cfg.neuron_add; end
        if ~isfield(cfg,'m_frac_list'), cfg.m_frac_list = [0.2 0.4 0.6 0.8 1.0]; end
        if ~isfield(cfg,'prefix_score_ratio'), cfg.prefix_score_ratio = 0.80; end
        if ~isfield(cfg,'tau_stop_score'), cfg.tau_stop_score = 0.25; end
        if ~isfield(cfg,'lambda'), cfg.lambda = 1e-6; end
        if ~isfield(cfg,'delta'), cfg.delta = 0.003; end
        if ~isfield(cfg,'q'), cfg.q = 2; end
        if ~isfield(cfg,'rho'), cfg.rho = 0.85; end
        if ~isfield(cfg,'tau_cap'), cfg.tau_cap = 1e-5; end
        if ~isfield(cfg,'kappa_map_max'), cfg.kappa_map_max = 1e6; end
        if ~isfield(cfg,'family_tol_accept'), cfg.family_tol_accept = 1e-4; end
        if ~isfield(cfg,'tol_improve'), cfg.tol_improve = 0.02; end
        if ~isfield(cfg,'stop_when_reject'), cfg.stop_when_reject = true; end
        if ~isfield(cfg,'stop_when_small_improve'), cfg.stop_when_small_improve = true; end
        % [] means use the paper-derived cap ceil(gamma_AD*m_R*log(1+m_R)).
        % A positive value remains available as an optional hard upper cap.
        if ~isfield(cfg,'max_train_points'), cfg.max_train_points = []; end
        if ~isfield(cfg,'gamma_AD') || isempty(cfg.gamma_AD), cfg.gamma_AD = 1.0; end
        if ~isfield(cfg,'tau_block_start'), cfg.tau_block_start = 0; end
        if ~isfield(cfg,'plot_figures'), cfg.plot_figures = false; end
        if ~isfield(cfg,'r0'), cfg.r0 = ctx.r; end
    
        if ~isfield(cfg,'newton_cfg')
            cfg.newton_cfg = struct();
        end

        if ~isfield(cfg,'jref_cfg')
            cfg.jref_cfg = cfg.newton_cfg;
        end
    
        if ~isfield(cfg,'family_full_newton_cfg')
            cfg.family_full_newton_cfg = cfg.newton_cfg;
            cfg.family_full_newton_cfg.num_newton = getfield_default_local(cfg,'family_newton_steps',3);
        end
    
        if ~isfield(cfg,'screen_adam_opts')
            cfg.screen_adam_opts = struct();
        end
        cfg.screen_adam_opts = fill_adam_opts_local(cfg.screen_adam_opts);
    
        if ~isfield(cfg,'block_adam_opts')
            cfg.block_adam_opts = cfg.screen_adam_opts;
        end
        cfg.block_adam_opts = fill_adam_opts_local(cfg.block_adam_opts);
    end

    function n_train = rbe_training_cap_3d_local(m_features,n_available,cfg)
    %RBE_TRAINING_CAP_3D_LOCAL Paper (C.1) training-point budget for RBE.
    %   N_R^max = min{N_eq,R, ceil(gamma_AD*m_R*log(1+m_R))}.
    % gamma_AD defaults to one; max_train_points is only an optional extra cap.
        if n_available <= 0
            n_train = 0;
            return;
        end
        m_features = max(1,double(m_features));
        if ~isfield(cfg,'gamma_AD') || isempty(cfg.gamma_AD)
            gamma_ad = 1.0;
        else
            gamma_ad = double(cfg.gamma_AD);
        end
        cap = ceil(gamma_ad*m_features*log(1+m_features));
        if isfield(cfg,'max_train_points') && ~isempty(cfg.max_train_points) && ...
                isnumeric(cfg.max_train_points) && isfinite(cfg.max_train_points) && ...
                cfg.max_train_points > 0
            cap = min(cap,double(cfg.max_train_points));
        end
        n_train = max(1,min(double(n_available),cap));
    end
    
    function opts = fill_adam_opts_local(opts)
        if ~isfield(opts,'eta'), opts.eta = 0.25; end
        if ~isfield(opts,'maxit'), opts.maxit = 40; end
        if ~isfield(opts,'eps_adam'), opts.eps_adam = 1e-8; end
        if ~isfield(opts,'clip_min'), opts.clip_min = [1;1;1]; end
        if ~isfield(opts,'clip_max'), opts.clip_max = [300;300;300]; end
        if ~isfield(opts,'minit'), opts.minit = 5; end
        if ~isfield(opts,'grad_tol'), opts.grad_tol = 0; end
        if ~isfield(opts,'fd_step_rel'), opts.fd_step_rel = 1e-3; end
        if ~isfield(opts,'verbose'), opts.verbose = false; end
        if ~isfield(opts,'use_rel_grad'), opts.use_rel_grad = true; end
    end
    
    function candidate_families = choose_candidate_families_local(cfg,last_selected_family)
        candidate_families = cfg.families;
    
        switch lower(cfg.family_repeat_mode)
            case 'all'
                % Exclude nothing.
    
            case 'exclude_last'
                % Exclude only the family selected in the previous round. It
                % returns automatically after a different family is selected.
                if ~isempty(last_selected_family)
                    keep = true(size(candidate_families));
                    for kk = 1:numel(candidate_families)
                        keep(kk) = ~strcmpi(candidate_families{kk},last_selected_family);
                    end
                    candidate_families = candidate_families(keep);
                end
    
            otherwise
                error('Unknown cfg.family_repeat_mode = %s',cfg.family_repeat_mode);
        end
    end
    
    %% ============================================================
    % sampling and scoring
    %% ============================================================
    function p = residual_sampling_prob_local(F_int,cfg,SAB_eps)
        R_abs = abs(F_int(:));
        if isempty(R_abs)
            p = zeros(0,1);
            return;
        end
        R_hat = R_abs/(max(R_abs)+SAB_eps);
        score = cfg.delta + R_hat.^cfg.q;
        p_adapt = score/(sum(score)+SAB_eps);
        p_uniform = ones(size(score))/numel(score);
        p = (1-cfg.rho)*p_uniform + cfg.rho*p_adapt;
        p = p/(sum(p)+SAB_eps);
    end
    
    function [score,rank_keep,kappa_map,valid] = score_term_local(ctx,term,R,cfg)
        SAB_eps = ctx.SAB_eps;
    
        C = Burgers2D_Newton_module3D('jref_matrix_for_term',ctx,term,cfg.jref_cfg);
        C = gather(C);
        R = gather(R(:));
    
        C = project_off_jref_parts_local(ctx,C);
    
        [U,S,V] = svd(C,'econ'); %#ok<ASGLU>
        s = diag(S);
    
        if isempty(s) || s(1) == 0
            score = 0;
            rank_keep = 0;
            kappa_map = inf;
            valid = false;
            return;
        end
    
        rel_tol = get_orth_rel_tol_local(ctx,cfg);
        rank_keep = sum(s >= rel_tol*s(1));
        rank_keep = max(0,rank_keep);
    
        if rank_keep < 1
            score = 0;
            kappa_map = inf;
            valid = false;
            return;
        end
    
        clear C S
        if rank_keep < size(U,2)
            U = U(:,1:rank_keep);
            V = V(:,1:rank_keep);
        end
        s = s(1:rank_keep);
    
        T_block = bsxfun(@rdivide,V,s.');
        kappa_map = norm(T_block,2);
    
        score = norm(U.'*R)^2/(norm(R)^2 + SAB_eps);
        valid = (rank_keep >= 1) && (score >= cfg.tau_cap) && (kappa_map <= cfg.kappa_map_max);
    end
    
    function [term_sorted,info] = sort_term_by_capture_local(ctx,term,R,cfg)
    %SORT_TERM_BY_CAPTURE_LOCAL
    % Sort columns in a sampled candidate pool by one-column frozen-Jref residual capture.
    
        C = Burgers2D_Newton_module3D('jref_matrix_for_term',ctx,term,cfg.jref_cfg);
        C = gather(C);
        R = gather(R(:));
    
        C = project_off_jref_parts_local(ctx,C);
    
        col_norm2 = sum(C.^2,1).' + ctx.SAB_eps;
        col_corr  = C.' * R;
        score_col = (col_corr.^2) ./ col_norm2;
        score_col(~isfinite(score_col)) = 0;
    
        [~,ord] = sort(score_col,'descend');
    
        term_sorted = select_term_cols_local(term,ord);
    
        info = struct();
        if isempty(score_col)
            info.best = 0;
            info.median = 0;
            info.worst = 0;
        else
            score_sorted = score_col(ord);
            info.best = score_sorted(1);
            info.median = median(score_sorted);
            info.worst = score_sorted(end);
        end
    end
    
    function [term_sel,info] = select_prefix_local(ctx,term_full,R,cfg)
        m_full = Burgers2D_Newton_module3D('term_num_cols',term_full);
    
        frac_list = cfg.m_frac_list(:).';
        frac_list = unique(min(max(frac_list,0),1));
        frac_list(frac_list <= 0) = [];
    
        if isempty(frac_list) || frac_list(end) < 1
            frac_list = unique([frac_list,1.0]);
        end
    
        full_score = 0;
        full_rank = 0;
        full_kappa = inf;
        full_valid = false;
    
        scores = zeros(numel(frac_list),1);
        ranks  = zeros(numel(frac_list),1);
        kappas = inf(numel(frac_list),1);
        valids = false(numel(frac_list),1);
        ms     = zeros(numel(frac_list),1);
    
        for ii = 1:numel(frac_list)
    
            m = max(1,ceil(frac_list(ii)*m_full));
            m = min(m,m_full);
            ms(ii) = m;
    
            term_i = select_term_cols_local(term_full,1:m);
            [score_i,rank_i,kappa_i,valid_i] = score_term_local(ctx,term_i,R,cfg);
    
            scores(ii) = score_i;
            ranks(ii) = rank_i;
            kappas(ii) = kappa_i;
            valids(ii) = valid_i;
    
            fprintf('  prefix m=%d: cap_score=%.4e, rank=%d, kappa=%.3e\n', ...
                m,score_i,rank_i,kappa_i);
    
            if m == m_full
                full_score = score_i;
                full_rank = rank_i;
                full_kappa = kappa_i;
                full_valid = valid_i;
            end
        end
    
        if full_score <= 0 || ~full_valid
            [best_score,best_id] = max(scores .* valids);
    
            if best_score <= 0
                term_sel = term_full;
                info = struct('valid',false,'m',m_full,'score',0, ...
                    'full_score',full_score,'rank',0,'kappa',inf);
                return;
            end
    
        else
            threshold = cfg.prefix_score_ratio * full_score;
            best_id = [];
    
            for ii = 1:numel(frac_list)
                if valids(ii) && scores(ii) >= threshold
                    best_id = ii;
                    break;
                end
            end
    
            if isempty(best_id)
                best_id = numel(frac_list);
            end
        end
    
        m_sel = ms(best_id);
        term_sel = select_term_cols_local(term_full,1:m_sel);
    
        info = struct();
        info.valid = valids(best_id);
        info.m = m_sel;
        info.score = scores(best_id);
        info.full_score = full_score;
        info.full_rank = full_rank;
        info.full_kappa = full_kappa;
        info.rank = ranks(best_id);
        info.kappa = kappas(best_id);
    end
    
    %% ============================================================
    % term column utilities
    %% ============================================================
    function term2 = select_term_cols_local(term,cols)
        cols = cols(:).';
        term2 = term;
    
        switch lower(term.type)
    
            case {'tanh','gauss','sin'}
                term2.w = term.w(:,cols);
                term2.b = term.b(:,cols);
    
            case {'radial_gauss','mexican_hat'}
                term2.A = term.A(:,cols);
                term2.r = term.r;
    
            otherwise
                error('Unknown term.type = %s',term.type);
        end
    end
    
    function rel_tol = get_orth_rel_tol_local(ctx,cfg)
        rel_tol = 1e-10;
    
        if isfield(cfg,'jref_cfg') && ...
           isfield(cfg.jref_cfg,'orth_cfg') && ...
           isfield(cfg.jref_cfg.orth_cfg,'rel_tol')
            rel_tol = cfg.jref_cfg.orth_cfg.rel_tol;
    
        elseif isfield(ctx,'orth_cfg') && isfield(ctx.orth_cfg,'rel_tol')
            rel_tol = ctx.orth_cfg.rel_tol;
        end
    end

    function Cperp = project_off_jref_parts_local(ctx,C)
        Cperp = C;
        row_chunk = 4096;
        if isfield(ctx,'jref') && isfield(ctx.jref,'svd_state') && ...
                ~isempty(ctx.jref.svd_state)
            Q = gather(ctx.jref.svd_state.U);
            H = Q.'*C;
            for row0 = 1:row_chunk:size(C,1)
                rows = row0:min(row0+row_chunk-1,size(C,1));
                Cperp(rows,:) = Cperp(rows,:) - Q(rows,:)*H;
            end
            return;
        end
        if ~isfield(ctx,'jref') || ~isfield(ctx.jref,'parts')
            return;
        end
        for kk = 1:numel(ctx.jref.parts)
            Qk = gather(ctx.jref.parts{kk}.Q);
            Hk = Qk.'*C;
            for row0 = 1:row_chunk:size(C,1)
                rows = row0:min(row0+row_chunk-1,size(C,1));
                Cperp(rows,:) = Cperp(rows,:) - Qk(rows,:)*Hk;
            end
            clear Qk Hk
        end
    end
    
    function val = getfield_default_local(s,name,default_val)
        if isstruct(s) && isfield(s,name)
            val = s.(name);
        else
            val = default_val;
        end
    end
