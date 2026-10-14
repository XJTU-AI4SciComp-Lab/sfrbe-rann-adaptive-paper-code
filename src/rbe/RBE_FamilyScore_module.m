function ctx = RBE_FamilyScore_module(ctx,cfg)
%RBE_FAMILYSCORE_MODULE_COMPAT
% Residual-score family enrichment for the generic 2D ProblemFunction PDE.
%
% This version is for the double-Gaussian Poisson / generic 2D linear
% ProblemFunction project. The supported families are the ones implemented
% in BasisFunction:
%   tanh, ridge_gauss, sin, radial_gauss, mexican_hat.
%
% Logic:
%   1. Initialize the RBE residual space from the current global tanh solve.
%   2. At each round, screen all candidate families except the last selected
%      one when cfg.family_repeat_mode='exclude_last'.
%   3. Each family keeps its own latest trained r as the next initial value.
%      If a family is skipped/excluded, its memory is not changed.
%   4. Rank all screening-admissible families by residual-capture score.
%      Test them in descending score order and select the first family that
%      admits a stable prefix satisfying the capture and mapping bounds.
%   5. Add the smallest admissible prefix reaching the prescribed fraction
%      of the corresponding full-pool score.
%   6. Stop only after cfg.stall_patience consecutive reject / insufficient
%      improvement rounds.

    if ~isfield(cfg,'op') || isempty(cfg.op)
        cfg.op = get_problem_op_local(ctx);
    end

    cfg = fill_score_cfg_local(cfg,ctx);
    ctx = init_orb_state_local(ctx,cfg);
    orb_cache_cleanup = [];
    if isfield(ctx,'orb_basis_cache_dir') && ~isempty(ctx.orb_basis_cache_dir)
        cache_dir_for_cleanup = ctx.orb_basis_cache_dir;
        orb_cache_cleanup = onCleanup(@() cleanup_orb_basis_cache_local(cache_dir_for_cleanup)); %#ok<NASGU>
    end
    ctx.rbe_log = struct([]);   % per-round selected family / m / r / scores

    fprintf('\n========== Residual-Score Family Selection: one family per round ==========%s',newline);
    fprintf('family_repeat_mode = %s\n',cfg.family_repeat_mode);

    SAB_eps = ctx.SAB_eps;
    n_int = size(ctx.x_point,1);

    current_rel = norm(ctx.r_orb_current_W(:),2) / ...
                  (norm(ctx.y_orb_current_W(:),2) + SAB_eps);

    fprintf('Initial ScoreFamily: rank=%d, rel_res=%.4e\n', ...
        orb_basis_column_count_local(ctx),current_rel);

    r_memory = struct();
    used_family = {};
    total_attempt = 0;
    bad_round_count = 0;
    last_selected_family = '';

    for family_round = 1:cfg.max_family_rounds

        current_rel = norm(ctx.r_orb_current_W(:),2) / ...
                      (norm(ctx.y_orb_current_W(:),2) + SAB_eps);


        families_round = choose_candidate_families_local(cfg,last_selected_family);

        if isempty(families_round)
            fprintf('No candidate family this round; stopping RBE family enrichment.\n');
            break;
        end

        fprintf('\n========== Family round %d / %d ==========%s', ...
            family_round,cfg.max_family_rounds,newline);

        if strcmpi(cfg.family_repeat_mode,'exclude_last')
            if isempty(last_selected_family)
                fprintf('family_repeat_mode = exclude_last: no family selected last round, using the full candidate set.\n');
            else
                fprintf('family_repeat_mode = exclude_last: temporarily excluding last round family: %s\n', ...
                    last_selected_family);
            end
        elseif strcmpi(cfg.family_repeat_mode,'all')
            fprintf('family_repeat_mode = all: every family is screened each round.\n');
        end

        fprintf('candidates: ');
        for ii = 1:numel(families_round)
            fprintf('%s ',families_round{ii});
        end
        fprintf('\n');

        %% ========== screening: score each family with a proportional pool ==========
        screen_result = struct([]);

        % The screening pool is tied to the full candidate block.  This
        % preserves m_scr=300 for the paper setting m_full=1000, while
        % avoiding an unnecessarily large screening problem for small
        % smoke blocks.  The cap is a hard computational safeguard.
        m_full_screen = full_candidate_block_size_local(cfg);
        m_scr = screening_block_size_local(cfg,m_full_screen);
        fprintf('screening block: m_full=%d, m_scr=%d (ratio=%.2f, cap=%d)\n', ...
            m_full_screen,m_scr,cfg.screen_ratio,cfg.screen_max);

        % The residual sampling distribution is common to all families in
        % this round; compute it once and keep family-specific draws below.
        r_int = ctx.r_orb_current_W(1:n_int);
        p = residual_sampling_prob_local(r_int,cfg,SAB_eps);

        for ff = 1:numel(families_round)

            family = families_round{ff};
            % Paper legacy: no fixed screening seed (keep the continuing random stream).

            idx_center = YY_utils('sample_from_prob',p,m_scr);
            centers = gather(ctx.x_point(idx_center,:).');

            raw = build_raw_family_local(family,m_scr,centers,cfg);

            n_train = rbe_training_cap_local( ...
                m_scr,length(p),cfg);
            idx_train = YY_utils('sample_from_prob',p,n_train);

            X_train = gather(ctx.x_point(idx_train,:));
            target = gather(-r_int(idx_train));

            r0_family = get_family_r0_local(r_memory,family,cfg.r0);
            scrOpts = cfg.screen_adam_opts;
            old_mem = get_family_memory_value_local(r_memory,family);

            [r_scr,MSE_scr,~,term_scr] = train_r_generic_local( ...
                ctx,X_train,target,raw,r0_family,scrOpts,cfg.op,family,cfg.lambda,cfg);

            % Whenever a family scale training finishes this round, store its latest r.
            % If the family later fails the full-pool/prefix test, restore the old memory.
            r_memory = set_family_r_memory_local(r_memory,family,r_scr);

            [score_scr,rank_scr,kappa_scr,valid_scr] = score_candidate_block_local( ...
                ctx,term_scr,family,cfg);

            screen_result(ff).family = family; %#ok<AGROW>
            screen_result(ff).score = score_scr;
            screen_result(ff).rank = rank_scr;
            screen_result(ff).kappa = kappa_scr;
            screen_result(ff).valid = valid_scr;
            screen_result(ff).r = r_scr;
            screen_result(ff).r0 = r0_family;
            screen_result(ff).old_memory = old_mem;
            screen_result(ff).MSE = MSE_scr;

            fprintf(['screen %s: score=%.4e, rank=%d, ', ...
                     'kappa_map=%.3e <= %.3e (valid=%d), ', ...
                     'r0=[%.4f, %.4f], r=[%.4f, %.4f], MSE=%.4e\n'], ...
                family,score_scr,rank_scr,kappa_scr,cfg.kappa_map_max,valid_scr, ...
                r0_family(1),r0_family(2),r_scr(1),r_scr(2),MSE_scr);
        end

        %% ============================================================
        % sort screening-admissible families by score in descending order.
        % Build each family large candidate pool in turn; if the current family has no valid prefix, 
        % try the next high-score family within the same round.
        %% ============================================================
        valid_idx = find([screen_result.valid]);

        if isempty(valid_idx)
            bad_round_count = bad_round_count + 1;
            fprintf('No family passed screening this round, bad_round=%d/%d.\n', ...
                bad_round_count,cfg.stall_patience);

            if bad_round_count >= cfg.stall_patience
                fprintf('Stopping family selection: %d consecutive rounds without valid candidates.\n', ...
                    cfg.stall_patience);
                break;
            else
                last_selected_family = '';
                continue;
            end
        end

        valid_scores = [screen_result(valid_idx).score];
        [~,score_order] = sort(valid_scores,'descend');
        candidate_ids = valid_idx(score_order);

        old_res_norm = norm(ctx.r_orb_current_W(:),2);
        old_rel = old_res_norm / ...
                  (norm(ctx.y_orb_current_W(:),2) + SAB_eps);

        r_int = ctx.r_orb_current_W(1:n_int);
        p = residual_sampling_prob_local(r_int,cfg,SAB_eps);

        m_full = full_candidate_block_size_local(cfg);

        m_list = unique(max(1,round(cfg.m_frac_list(:).' * m_full)),'stable');
        m_list(m_list > m_full) = m_full;
        if isempty(m_list) || m_list(end) ~= m_full
            m_list = [m_list,m_full];
        end

        family_found = false;
        selected_family = '';
        selected_r = [];
        selected_old_memory = [];
        selected_m = [];
        selected_term = [];
        selected_score = -inf;
        selected_rank = 0;
        selected_kappa = inf;
        full_score = 0;
        screening_score = -inf;

        for kk = 1:numel(candidate_ids)

            candidate_id = candidate_ids(kk);
            candidate_family = screen_result(candidate_id).family;
            candidate_r = screen_result(candidate_id).r(:);
            candidate_old_memory = screen_result(candidate_id).old_memory;
            candidate_screen_score = screen_result(candidate_id).score;

            total_attempt = total_attempt + 1;

            fprintf(['\nTrying family %d/%d: %s, screening score=%.4e, ', ...
                     'r=[%.4f, %.4f]\n'], ...
                kk,numel(candidate_ids),candidate_family, ...
                candidate_screen_score,candidate_r(1),candidate_r(2));

            % Make each family full pool deterministic for a given round, 
            % while avoiding identical random sequences across families.
            rng(9000 + 1000*family_round + ...
                10*candidate_id + ctx.orb_block_count);

            idx_center = YY_utils('sample_from_prob',p,m_full);
            centers = gather(ctx.x_point(idx_center,:).');
            raw_full = build_raw_family_local( ...
                candidate_family,m_full,centers,cfg);

            % candidate_r is already trained during screening; 
            % The large candidate pool reuses this scale directly.
            term_full_unsorted = build_term_from_r_local( ...
                candidate_family,raw_full,candidate_r);
            C_full_unsorted = candidate_residual_matrix_local( ...
                ctx,term_full_unsorted,candidate_family,cfg);

            [order,cap_info,C_perp_unsorted] = rank_candidate_matrix_local( ...
                ctx,C_full_unsorted);

            fprintf(['Candidate pool ranked by per-column residual capture: ', ...
                     'best=%.4e, median=%.4e, worst=%.4e\n'], ...
                cap_info.best,cap_info.median,cap_info.worst);

            raw_full = reorder_raw_family_local(raw_full,order);
            % Reuse the already residualized and reordered block below.  The
            % old implementation projected the same candidate repeatedly for
            % the full block and every prefix.
            C_perp = C_perp_unsorted(:,order);
            clear C_perp_unsorted C_full_unsorted

            [candidate_full_score,~,~,~] = ...
                score_projected_candidate_matrix_local(ctx,C_perp,cfg);

            found_prefix = false;

            for mm = m_list
                raw_prefix = subset_raw_family_local(raw_full,mm);
                term_prefix = build_term_from_r_local( ...
                    candidate_family,raw_prefix,candidate_r);
                [score_m,rank_m,kappa_m,valid_m] = ...
                    score_projected_candidate_matrix_local(ctx,C_perp(:,1:mm),cfg);

                fprintf(['  prefix m=%d: cap_score=%.4e, ', ...
                         'rank=%d, kappa_map=%.3e <= %.3e (valid=%d)\n'], ...
                    mm,score_m,rank_m,kappa_m,cfg.kappa_map_max,valid_m);

                reaches_target = ...
                    score_m >= cfg.prefix_score_ratio * candidate_full_score;

                if valid_m && reaches_target
                    selected_family = candidate_family;
                    selected_r = candidate_r;
                    selected_old_memory = candidate_old_memory;
                    screening_score = candidate_screen_score;

                    selected_m = mm;
                    selected_term = term_prefix;
                    selected_score = score_m;
                    selected_rank = rank_m;
                    selected_kappa = kappa_m;
                    full_score = candidate_full_score;

                    found_prefix = true;
                    family_found = true;
                    break;
                end
            end

            if family_found
                break;
            end

            fprintf(['Family %s does not satisfy both the capture-ratio and ', ...
                     'amplification constraints for a prefix; trying the next high-score family.\n'], ...
                candidate_family);

            % The large candidate pool of this family failed; restore its r-memory from before screening.
            r_memory = restore_family_memory_local( ...
                r_memory,candidate_family,candidate_old_memory);
        end

        %% ============================================================
        % None of the screening-admissible families has a valid prefix.
        %% ============================================================
        if ~family_found
            bad_round_count = bad_round_count + 1;
            fprintf(['None of the screening-admissible families has a valid prefix, ', ...
                     'bad_round=%d/%d.\n'], ...
                bad_round_count,cfg.stall_patience);

            last_selected_family = '';

            if bad_round_count >= cfg.stall_patience
                fprintf(['Stopping family selection: %d consecutive rounds without any family ', ...
                         'having a valid prefix.\n'],cfg.stall_patience);
                break;
            else
                continue;
            end
        end

        % Record the selected family only after a valid prefix is actually found.
        last_selected_family = selected_family;

        fprintf(['\nSelected family: %s, screening score=%.4e, ', ...
                 'r=[%.4f, %.4f]\n'], ...
            selected_family,screening_score,selected_r(1),selected_r(2));

        fprintf(['prefix selected m=%d, cap_score=%.4e, full_score=%.4e, ', ...
                 'rank=%d, kappa_map=%.3e <= %.3e\n'], ...
            selected_m,selected_score,full_score,selected_rank, ...
            selected_kappa,cfg.kappa_map_max);

    % ============================================================
    % absolute capture-score stopping condition
    % Use the finally selected prefix score, not the screening score.
    % If selected_score < cfg.tau_stop_score, do not add the block.
    % ============================================================
    if selected_score < cfg.tau_stop_score
        fprintf(['Stopping family selection: final prefix cap_score=%.4e ', ...
                '< tau_stop_score=%.4e, the block is not added.\n'], ...
                selected_score,cfg.tau_stop_score);
        ctx.rbe_log = append_rbe_log_local(ctx.rbe_log, family_round, selected_family, ...
            selected_m, selected_r, screening_score, selected_score, full_score, ...
            selected_rank, selected_kappa, false, old_rel, old_rel, NaN, ...
            'cap_below_tau_stop', screen_result);
        invoke_detail_round_callback_local(ctx,ctx.rbe_log(end));
        fprintf('  >> RBE round %d: %s cap=%.3e < tau_stop, not added\n', ...
            family_round, selected_family, selected_score);
        break;
    end

    % Rebuild the accepted raw block only once inside the append routine.
    clear C_perp
    [ctx,accept,info] = append_family_block_local( ...
        ctx,selected_term,selected_family,cfg);

        if accept
            ctx.last_r_opt = selected_r;
            used_family{end+1} = selected_family; %#ok<AGROW>
            fprintf(['%s family accept: m=%d, rel_res %.4e -> %.4e, ', ...
                     'rho=%.4e, kappa_map=%.3e <= %.3e\n'], ...
                selected_family,selected_m,old_rel,info.rel_new,info.rho_tr, ...
                info.kappa,cfg.kappa_map_max);
            ctx.rbe_log = append_rbe_log_local(ctx.rbe_log, family_round, selected_family, ...
                selected_m, selected_r, screening_score, selected_score, full_score, ...
                selected_rank, selected_kappa, true, old_rel, info.rel_new, NaN, ...
                '', screen_result);
            fprintf(['  >> RBE round %d: %s m=%d r=[%.4f %.4f] ', ...
                     'cap=%.3e rel_res=%.4e->%.4e ', ...
                     'kappa_map=%.3e<=%.3e\n'], ...
                family_round, selected_family, selected_m, selected_r(1), selected_r(2), ...
                selected_score, old_rel, info.rel_new, info.kappa, ...
                cfg.kappa_map_max);

            % Optional full Newton after each accepted block.
            % For the frozen-Jacobian (nonlinear) path Newton already ran
            % inside append_family_block_local before the accept decision.
            if isfield(cfg, 'run_newton_each_block') && cfg.run_newton_each_block && ...
                    isfield(cfg, 'newton') && ...
                    ~(isfield(cfg,'use_frozen_jacobian') && cfg.use_frozen_jacobian)
                consN = get_problem_constraints_local(ctx);
                if ~isempty(consN)
                    basisEvalOrb = @(ord, dir, X) RBE_Eval('terms', ctx.orb_terms, X, ord, dir);
                    [ctx.gamma_orb_current, nwB] = NewtonSolver( ...
                        ctx.problem, basisEvalOrb, ctx.x_point, consN, ...
                        ctx.gamma_orb_current, cfg.newton);
                    fprintf('  [Newton] after block %d: rel_res=%.4e, iter=%d\n', ...
                        family_round, nwB.rel_res, nwB.iter);
                end
            end

            invoke_detail_round_callback_local(ctx,ctx.rbe_log(end));



            if info.rho_tr < cfg.tol_improve
                bad_round_count = bad_round_count + 1;
                fprintf('Current family improvement too small: rho=%.4e < %.4e, bad_round=%d/%d.\n', ...
                    info.rho_tr,cfg.tol_improve,bad_round_count,cfg.stall_patience);
            else
                bad_round_count = 0;
            end

            if bad_round_count >= cfg.stall_patience
                fprintf('Stopping family selection: %d consecutive rounds with insufficient improvement.\n',cfg.stall_patience);
                break;
            end
        else
            bad_round_count = bad_round_count + 1;
        
            % Restore the family original r-memory.
            % append_family_block_local does not modify the current trial space when accept=false.
            r_memory = restore_family_memory_local( ...
                r_memory,selected_family,selected_old_memory);
        
            fprintf(['%s family reject: m=%d, rel_res %.4e -> %.4e, ', ...
                     'kappa_map=%.3e <= %.3e, reason=%s\n'], ...
                selected_family,selected_m,old_rel,info.rel_new,info.kappa, ...
                cfg.kappa_map_max,info.reason);
            ctx.rbe_log = append_rbe_log_local(ctx.rbe_log, family_round, selected_family, ...
                selected_m, selected_r, screening_score, selected_score, full_score, ...
                selected_rank, selected_kappa, false, old_rel, info.rel_new, NaN, ...
                info.reason, screen_result);
            invoke_detail_round_callback_local(ctx,ctx.rbe_log(end));
            fprintf(['  >> RBE round %d: %s m=%d r=[%.4f %.4f] ', ...
                     'rel_res=%.4e->%.4e kappa_map=%.3e<=%.3e rejected (%s)\n'], ...
                family_round, selected_family, selected_m, selected_r(1), selected_r(2), ...
                old_rel, info.rel_new, info.kappa, cfg.kappa_map_max, info.reason);
        
            % If the new block does not reduce the residual (or increases it), stop the whole RBE immediately.
            if strcmp(info.reason,'res_not_down') || ...
               strcmp(info.reason,'res_worse_rollback')
        
                fprintf(['Stopping family selection: residual did not decrease after adding the candidate block, ', ...
                         'the block was rejected and the old solution kept.\n']);
                break;
            end
        
            % Other failure reasons still use the original consecutive-failure rule.
            if cfg.stop_when_reject && bad_round_count >= cfg.stall_patience
                fprintf('Stopping family selection: %d consecutive rejects/insufficient improvements.\n', ...
                    cfg.stall_patience);
                break;
            end
        end
    end

    fprintf('\n========== Residual-Score Family Selection summary ==========%s',newline);
    fprintf('number of selected families = %d\n',numel(used_family));
    fprintf('selection order: ');
    for kk = 1:numel(used_family)
        fprintf('%s ',used_family{kk});
    end
    fprintf('\n');
    fprintf('total families tried = %d\n',total_attempt);
    fprintf('current full stable rank = %d\n',orb_basis_column_count_local(ctx));
    fprintf('current full weighted residual norm = %.6e\n',norm(ctx.r_orb_current_W));
    fprintf('current relative residual = %.6e\n', ...
        norm(ctx.r_orb_current_W)/(norm(ctx.y_orb_current_W)+SAB_eps));

    if ~isempty(orb_cache_cleanup)
        ctx.Q_orb_blocks = {};
        ctx.Q_orb_block_files = {};
        ctx.Q_orb_block_sizes = zeros(0,2);
        ctx.orb_basis_cache_dir = '';
        clear orb_cache_cleanup
    end
end

%% ============================================================
% configuration defaults
%% ============================================================
%% ============================================================
% generic problem helpers
%% ============================================================
function cons = get_problem_constraints_local(ctx)
    cons = [];
    try
        cons = ProblemFunction('constraints', [], ctx.problem);
    catch
        cons = [];
    end
end

%% ============================================================
% RBE log helpers
%% ============================================================
function log = append_rbe_log_local(log, rnd, family, m, r, scrScore, capScore, ...
        fullScore, rankKeep, kappaKeep, accepted, relBefore, relAfter, l2After, reason, screen)
    entry = struct('round', rnd, 'family', family, 'm', m, 'r', r(:).', ...
        'screen_score', scrScore, 'cap_score', capScore, 'full_score', fullScore, ...
        'rank', rankKeep, 'kappa', kappaKeep, 'accept', accepted, ...
        'rel_before', relBefore, 'rel_after', relAfter, 'l2_after', l2After, ...
        'reason', reason, 'screen', screen);
    if isempty(log)
        log = entry;
    else
        log(end+1) = entry; %#ok<AGROW>
    end
end

function invoke_detail_round_callback_local(ctx,entry)
    if isfield(ctx,'detail_round_callback') && ...
            isa(ctx.detail_round_callback,'function_handle')
        try
            ctx.detail_round_callback(ctx,entry);
        catch err
            warning('RBE:detailedLog', ...
                'Detailed round callback failed: %s',err.message);
        end
    end
end

function cfg = fill_score_cfg_local(cfg,ctx)
    if ~isfield(cfg,'max_family_rounds'), cfg.max_family_rounds = 2; end
    if ~isfield(cfg,'screen_ratio') || isempty(cfg.screen_ratio)
        cfg.screen_ratio = 0.30;
    end
    if ~isfield(cfg,'screen_max') || isempty(cfg.screen_max)
        cfg.screen_max = 500;
    end

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
        error('cfg.family_repeat_mode must be ''exclude_last'' or ''all'', got %s.', ...
            cfg.family_repeat_mode);
    end

    if ~isfield(cfg,'neuron_add')
        if isfield(ctx,'neuron_add')
            cfg.neuron_add = ctx.neuron_add;
        elseif isfield(cfg,'m_block0')
            cfg.neuron_add = cfg.m_block0;
        else
            cfg.neuron_add = 500;
        end
    end

    if ~isfield(cfg,'m_block0'), cfg.m_block0 = cfg.neuron_add; end
    if ~isfield(cfg,'m_block_min'), cfg.m_block_min = 1; end
    if ~isfield(cfg,'m_block_max'), cfg.m_block_max = cfg.neuron_add; end
    if ~isfield(cfg,'m_frac_list'), cfg.m_frac_list = [0.2,0.4,0.6,0.8,1.0]; end
    if ~isfield(cfg,'prefix_score_ratio'), cfg.prefix_score_ratio = 0.80; end
    if ~isfield(cfg,'tau_stop_score'), cfg.tau_stop_score = 0.25; end
    if ~isfield(cfg,'tol_res'), cfg.tol_res = 1e-6; end
    if ~isfield(cfg,'tol_improve'), cfg.tol_improve = 0.01; end
    if ~isfield(cfg,'stall_patience') || isempty(cfg.stall_patience), cfg.stall_patience = 2; end
    if ~isfield(cfg,'stop_when_reject'), cfg.stop_when_reject = true; end
    if ~isfield(cfg,'eta_accept'), cfg.eta_accept = 0; end
    if ~isfield(cfg,'rollback_worse_tol'), cfg.rollback_worse_tol = 0.05; end
    if ~isfield(cfg,'tau_cap'), cfg.tau_cap = 1e-4; end
    if ~isfield(cfg,'kappa_map_max'), cfg.kappa_map_max = inf; end
    if ~isfield(cfg,'max_train_points'), cfg.max_train_points = []; end
    if ~isfield(cfg,'gamma_AD') || isempty(cfg.gamma_AD), cfg.gamma_AD = 1.0; end
    if ~isfield(cfg,'delta'), cfg.delta = 0.003; end
    if ~isfield(cfg,'q'), cfg.q = 2; end
    if ~isfield(cfg,'rho'), cfg.rho = 0.85; end
    if ~isfield(cfg,'lambda'), cfg.lambda = 1e-6; end
    if ~isfield(cfg,'plot_figures'), cfg.plot_figures = false; end
    if ~isfield(cfg,'r0'), cfg.r0 = ctx.r; end
    if ~isfield(cfg,'families'), cfg.families = {'tanh','ridge_gauss','sin','radial_gauss','mexican_hat'}; end
    if ~isfield(cfg,'use_frozen_jacobian'), cfg.use_frozen_jacobian = false; end

end

function m_full = full_candidate_block_size_local(cfg)
%FULL_CANDIDATE_BLOCK_SIZE_LOCAL apply one definition of m_full everywhere.
    m_full = double(cfg.neuron_add);
    m_full = max(double(cfg.m_block_min),m_full);
    m_full = min(double(cfg.m_block_max),m_full);
    m_full = max(1,round(m_full));
end

function m_scr = screening_block_size_local(cfg,m_full)
%SCREENING_BLOCK_SIZE_LOCAL m_scr=min{screen_max,ceil(screen_ratio*m_full)}.
    m_scr = ceil(double(cfg.screen_ratio)*double(m_full));
    m_scr = min(double(cfg.screen_max),m_scr);
    m_scr = max(1,round(m_scr));
end

%% ============================================================
% family-wise r-memory
%% ============================================================
function r0 = get_family_r0_local(r_memory,family,r_default)
    key = family_key_local(family);
    if isfield(r_memory,key) && ~isempty(r_memory.(key))
        r0 = r_memory.(key);
    else
        r0 = r_default(:);
    end
end

function val = get_family_memory_value_local(r_memory,family)
    key = family_key_local(family);
    if isfield(r_memory,key)
        val = r_memory.(key);
    else
        val = [];
    end
end

function r_memory = set_family_r_memory_local(r_memory,family,r_value)
    key = family_key_local(family);
    if all(isfinite(r_value(:)))
        r_memory.(key) = r_value(:);
    end
end

function r_memory = restore_family_memory_local(r_memory,family,old_value)
    key = family_key_local(family);
    if isempty(old_value)
        if isfield(r_memory,key)
            r_memory = rmfield(r_memory,key);
        end
    else
        r_memory.(key) = old_value(:);
    end
end

function key = family_key_local(family)
    key = lower(strtrim(family));
    key = regexprep(key,'[^a-zA-Z0-9_]','_');
end

%% ============================================================
% candidate family selection logic
%% ============================================================
function families_round = choose_candidate_families_local(cfg,last_selected_family)
    families_round = cfg.families;

    switch lower(cfg.family_repeat_mode)
        case 'all'
            % No exclusion at all.

        case 'exclude_last'
            % Only the family selected last round is excluded; it returns once another family is chosen.
            if ~isempty(last_selected_family)
                keep = true(size(families_round));
                for kk = 1:numel(families_round)
                    keep(kk) = ~strcmpi(families_round{kk},last_selected_family);
                end
                families_round = families_round(keep);
            end

        otherwise
            error('Unknown cfg.family_repeat_mode = %s',cfg.family_repeat_mode);
    end
end

%% ============================================================
% RBE initialization
%% ============================================================
function ctx = init_orb_state_local(ctx,cfg)
    if isfield(ctx,'orb_initialized') && ctx.orb_initialized == 1
        return;
    end

    yW = gather([ctx.right; bc_target_local(ctx, cfg)]);
    term0 = struct('w',ctx.w,'b',ctx.b);

    B_int = BasisOperator('tanh',term0,ctx.x_point,cfg.op);
    B_bd = bc_matrix_local(ctx, term0, 'tanh', cfg);
    B_full = gather([B_int; B_bd]);

    if isfield(ctx,'use_orth_basis') && ctx.use_orth_basis == 1 && ...
       isfield(ctx,'Q_current') && isfield(ctx,'T_current') && isfield(ctx,'gamma_current')
        Q = gather(ctx.Q_current);
        gamma = gather(ctx.gamma_current);
        yW = gather(ctx.y_current_W);
        rW = yW - Q * gamma;
        T = ctx.T_current;
    else
        [U_r,V_r,s_r] = YY_utils('svd_truncate',B_full,ctx.orth_cfg);
        T = bsxfun(@rdivide,V_r,s_r.');
        Q = U_r;
        gamma = Q' * yW;
        rW = yW - Q * gamma;
    end

    if cfg.use_frozen_jacobian
        % Nonlinear (frozen Newton-Jacobian) RBE: keep the current solution in
        % raw terms + raw coefficients so the initialized trial space
        % reproduces the current nonlinear solve exactly.  Q and rW are the
        % frozen-Jacobian scoring basis and the true weighted residual.
        [Q, R_frozen, yW] = frozen_jacobian_state_local(ctx, term0, cfg);
        c = current_raw_coef_local(ctx);
        T = eye(numel(c));
        gamma = c;
        rW = R_frozen;
    end

    % Store orthogonal response directions by accepted block.  This is
    % algebraically identical to [Q_0,U_1,...], but avoids repeatedly
    % allocating an ever larger contiguous matrix during enrichment.
    ctx.Q_orb_current = [];
    ctx.Q_orb_blocks = {Q};
    ctx.Q_orb_block_files = {''};
    ctx.Q_orb_block_sizes = [size(Q,1),size(Q,2)];
    ctx.orb_basis_cache_dir = '';

    % The multiscale Helmholtz run has a much larger response basis than
    % the other 2-D tests.  Keep the initial Q block in memory by default:
    % the projection is then reused without repeatedly loading a multi-GB
    % block from disk.  Disk streaming remains available as an explicit
    % fallback through cfg.use_disk_basis_cache.
    if isfield(ctx,'problem') && isfield(ctx.problem,'name') && ...
            strcmpi(ctx.problem.name,'linear_multiscale_sine_2d') && ...
            getfield_default_local(cfg,'use_disk_basis_cache',false)
        ctx.orb_basis_cache_dir = tempname;
        try
            mkdir(ctx.orb_basis_cache_dir);
            block_file = fullfile(ctx.orb_basis_cache_dir,'q_block_0001.mat');
            Q_block = Q; %#ok<NASGU>
            save(block_file,'Q_block','-v7.3');
            ctx.Q_orb_blocks = {[]};
            ctx.Q_orb_block_files = {block_file};
            if isfield(ctx,'Q_current')
                ctx.Q_current = [];
            end
        catch err
            % Disk cache unavailable (e.g., temp filesystem full).  Keep this
            % basis block in memory so RBE can still run.
            warning('ORB:cacheFallback', ...
                'Disk basis cache unavailable (%s); keeping block in memory.', err.message);
            ctx.orb_basis_cache_dir = '';
            ctx.Q_orb_blocks = {Q};
            ctx.Q_orb_block_files = {''};
        end
    end
    ctx.gamma_orb_current = gamma;
    ctx.y_orb_current_W = yW;
    ctx.r_orb_current_W = rW;
    ctx.r_orb0_norm = norm(rW);

    ctx.orb_terms = {};
    ctx.orb_terms{1}.type = 'tanh';
    ctx.orb_terms{1}.w = ctx.w;
    ctx.orb_terms{1}.b = ctx.b;
    ctx.orb_terms{1}.T = T;

    ctx.orb_initialized = 1;
    ctx.orb_block_count = 0;
end

%% ============================================================
% residual sampling probability
%% ============================================================
function p = residual_sampling_prob_local(r_int,cfg,SAB_eps)
    R_for_sampling = abs(r_int);
    R_hat = R_for_sampling ./ (max(R_for_sampling) + SAB_eps);
    score = cfg.delta + R_hat.^cfg.q;
    p_adapt = score ./ (sum(score) + SAB_eps);
    p_uniform = ones(size(p_adapt)) / length(p_adapt);
    p = (1 - cfg.rho) * p_uniform + cfg.rho * p_adapt;
    p = p ./ (sum(p) + SAB_eps);
end

function n_train = rbe_training_cap_local(m_features,n_available,cfg)
%RBE_TRAINING_CAP_LOCAL independent N log(1+N) scale-training budget.
% The screening pool is trained separately from every other RBE/SFR pool.
    if n_available <= 0
        n_train = 0;
        return;
    end
    m_features = max(1,double(m_features));
    gamma_ad = getfield_default_local(cfg,'gamma_AD',1.0);
    cap = ceil(gamma_ad*m_features*log(1+m_features));
    if isfield(cfg,'max_train_points') && ~isempty(cfg.max_train_points) && ...
            isnumeric(cfg.max_train_points) && isfinite(cfg.max_train_points) && ...
            cfg.max_train_points > 0
        cap = min(cap,double(cfg.max_train_points));
    end
    n_train = max(1,min(double(n_available),cap));
end

function v = getfield_default_local(s, name, default_val)
    if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
        v = s.(name);
    else
        v = default_val;
    end
end

%% ============================================================
% family raw/term utilities
%% ============================================================
function raw = build_raw_family_local(family,m,centers,cfg)
    raw = struct();
    raw.family = family;
    raw.m = m;
    raw.centers = centers;

    switch lower(family)
        case {'tanh','ridge_gauss','gauss','sin'}
            raw.w_basis = 2*rand(2,m) - 1;

        case {'radial_gauss','mexican_hat'}
            raw.w_basis = [];

        otherwise
            error('Unknown candidate family = %s',family);
    end
end

function raw2 = subset_raw_family_local(raw,m)
    idx = 1:m;
    raw2 = raw;
    raw2.m = m;
    raw2.centers = raw.centers(:,idx);
    if isfield(raw,'w_basis') && ~isempty(raw.w_basis)
        raw2.w_basis = raw.w_basis(:,idx);
    end
    if isfield(raw,'z_alpha')
        raw2.z_alpha = raw.z_alpha(idx);
        raw2.z_mu = raw.z_mu(idx);
        raw2.z_beta = raw.z_beta(idx);
        raw2.beta_shift = raw.beta_shift(idx);
    end
end

function raw2 = reorder_raw_family_local(raw,order)
    raw2 = raw;
    raw2.centers = raw.centers(:,order);
    if isfield(raw,'w_basis') && ~isempty(raw.w_basis)
        raw2.w_basis = raw.w_basis(:,order);
    end
    if isfield(raw,'z_alpha')
        raw2.z_alpha = raw.z_alpha(order);
        raw2.z_mu = raw.z_mu(order);
        raw2.z_beta = raw.z_beta(order);
        raw2.beta_shift = raw.beta_shift(order);
    end
end

function term = build_term_from_r_local(family,raw,r)
    r = r(:);
    centers = raw.centers;

    switch lower(family)
        case 'tanh'
            W = r .* raw.w_basis;
            term.type = 'tanh';
            term.w = W;
            term.b = -sum(centers .* W,1);

        case {'ridge_gauss','gauss'}
            W = r .* raw.w_basis;
            term.type = 'gauss';
            term.w = W;
            term.b = -sum(centers .* W,1);

        case 'sin'
            W = r .* raw.w_basis;
            term.type = 'sin';
            term.w = W;
            term.b = -sum(centers .* W,1);

        case 'radial_gauss'
            term.type = 'radial_gauss';
            term.A = centers;
            term.r = r;

        case 'mexican_hat'
            term.type = 'mexican_hat';
            term.A = centers;
            term.r = r;

        otherwise
            error('Unknown family = %s',family);
    end
end

function raw_m = get_raw_m_local(term)
    switch lower(term.type)
        case {'tanh','gauss','sin'}
            raw_m = size(term.w,2);

        case {'radial_gauss','mexican_hat'}
            raw_m = size(term.A,2);

        otherwise
            error('Unknown term.type = %s',term.type);
    end
end

%% ============================================================
% scoring / append
%% ============================================================

function C = candidate_residual_matrix_local(ctx,term,family,cfg)
    C_int = int_residual_response_local(ctx,term,family,cfg);
    C_bd  = bc_matrix_local(ctx, term, family, cfg);
    C = gather([C_int; C_bd]);
end

function [order,info,C_perp] = rank_candidate_matrix_local(ctx,C)
%RANK_CANDIDATE_MATRIX_LOCAL
% Input the already assembled weighted residual matrix C to avoid recomputing candidate features.
    SAB_eps = ctx.SAB_eps;

    [~,C_perp] = project_from_orb_basis_local(ctx,C);

    rW = ctx.r_orb_current_W(:);

    col_norm2 = sum(C_perp.^2,1).' + SAB_eps;
    col_corr = C_perp.' * rW;
    score_col = (col_corr.^2) ./ col_norm2;
    score_col(~isfinite(score_col)) = 0;

    [~,order] = sort(score_col,'descend');

    info = struct();
    if isempty(score_col)
        info.best = 0;
        info.median = 0;
        info.worst = 0;
    else
        score_sorted = score_col(order);
        info.best = score_sorted(1);
        info.median = median(score_sorted);
        info.worst = score_sorted(end);
    end
end

function [score,rank_keep,kappa_map,valid] = ...
        score_projected_candidate_matrix_local(ctx,C_perp,cfg)
%SCORE_PROJECTED_CANDIDATE_MATRIX_LOCAL score a block after projection.
% C_perp is already orthogonalized against the retained response space.
    SAB_eps = ctx.SAB_eps;

    [U_r,V_r,s_r,rank_keep] = YY_utils('svd_truncate',C_perp,ctx.orth_cfg);

    if isempty(s_r) || max(s_r) == 0 || rank_keep < 1
        score = 0;
        kappa_map = inf;
        valid = false;
        return;
    end

    T_block = bsxfun(@rdivide,V_r,s_r.');
    kappa_map = norm(T_block,2);
    score = norm(U_r' * ctx.r_orb_current_W)^2 / ...
            (norm(ctx.r_orb_current_W)^2 + SAB_eps);

    valid = (rank_keep >= 1) && ...
            (score >= cfg.tau_cap) && ...
            (kappa_map <= cfg.kappa_map_max);
end

function [order,info] = rank_candidate_columns_local(ctx,term,family,cfg)
%RANK_CANDIDATE_COLUMNS_LOCAL
% Rank the columns of an already sampled large candidate pool.
% score_j = ((c_j_perp' * residual)^2)/(||c_j_perp||^2 + eps)

    SAB_eps = ctx.SAB_eps;

    C_int = int_residual_response_local(ctx,term,family,cfg);
    C_bd  = bc_matrix_local(ctx, term, family, cfg);
    C = gather([C_int; C_bd]);

    [~,C_perp] = project_from_orb_basis_local(ctx,C);

    rW = ctx.r_orb_current_W(:);

    col_norm2 = sum(C_perp.^2,1).' + SAB_eps;
    col_corr = C_perp.' * rW;
    score_col = (col_corr.^2) ./ col_norm2;

    score_col(~isfinite(score_col)) = 0;

    [~,order] = sort(score_col,'descend');

    info = struct();
    if isempty(score_col)
        info.best = 0;
        info.median = 0;
        info.worst = 0;
    else
        score_sorted = score_col(order);
        info.best = score_sorted(1);
        info.median = median(score_sorted);
        info.worst = score_sorted(end);
    end
end

function [score,rank_keep,kappa_map,valid] = score_candidate_block_local(ctx,term,family,cfg)
    SAB_eps = ctx.SAB_eps;

    C_int = int_residual_response_local(ctx,term,family,cfg);
    C_bd  = bc_matrix_local(ctx, term, family, cfg);
    C = gather([C_int; C_bd]);

    [~,C_perp] = project_from_orb_basis_local(ctx,C);

    [U_r,V_r,s_r,rank_keep] = YY_utils('svd_truncate',C_perp,ctx.orth_cfg);

    if isempty(s_r) || max(s_r) == 0 || rank_keep < 1
        score = 0;
        kappa_map = inf;
        valid = false;
        return;
    end

    T_block = bsxfun(@rdivide,V_r,s_r.');
    kappa_map = norm(T_block,2);
    score = norm(U_r' * ctx.r_orb_current_W)^2 / ...
            (norm(ctx.r_orb_current_W)^2 + SAB_eps);

    valid = (rank_keep >= 1) && ...
            (score >= cfg.tau_cap) && ...
            (kappa_map <= cfg.kappa_map_max);
end

function [ctx,accept,info] = append_family_block_local(ctx,term,family,cfg,C)
    if nargin < 5 || isempty(C)
        C = candidate_residual_matrix_local(ctx,term,family,cfg);
    end
    if isfield(cfg,'use_frozen_jacobian') && cfg.use_frozen_jacobian
        % Nonlinear RBE: append raw features and accept using the true
        % nonlinear residual.
        [ctx,accept,info] = append_family_block_frozen_local(ctx,term,family,cfg,C);
        return;
    end

    SAB_eps = ctx.SAB_eps;
    old_res_norm = norm(ctx.r_orb_current_W(:),2);
    old_rel = old_res_norm/(norm(ctx.y_orb_current_W(:),2)+SAB_eps); %#ok<NASGU>


    [H,C_perp] = project_from_orb_basis_local(ctx,C);
    [U_r,V_r,s_r,r_keep] = YY_utils('svd_truncate',C_perp,ctx.orth_cfg);

    info = struct();
    info.reason = '';
    info.rho_tr = 0;
    info.rel_new = old_res_norm/(norm(ctx.y_orb_current_W(:),2)+SAB_eps);
    info.l2_new = NaN;
    info.cap = 0;
    info.rank = 0;
    info.kappa = inf;

    if isempty(s_r) || max(s_r) == 0 || r_keep < 1
        accept = false;
        info.reason = 'rank_zero';
        return;
    end

    T_block = bsxfun(@rdivide,V_r,s_r.');
    Z = U_r;
    kappa_map = norm(T_block,2);
    cap = norm(Z' * ctx.r_orb_current_W)^2 / ...
          (norm(ctx.r_orb_current_W)^2 + SAB_eps);

    gamma_new = Z' * ctx.r_orb_current_W;
    gamma_trial = [ctx.gamma_orb_current; gamma_new];
    r_trial = ctx.r_orb_current_W - Z * gamma_new;

    new_res_norm = norm(r_trial);
    rho_tr = 1 - new_res_norm^2 / (old_res_norm^2 + SAB_eps);

    accept_rank = r_keep >= 1;
    accept_cap = cap >= cfg.tau_cap;
    accept_res = new_res_norm <= (1 - cfg.eta_accept) * old_res_norm;
    accept_map = kappa_map <= cfg.kappa_map_max;
    accept_not_worse = new_res_norm <= (1 + cfg.rollback_worse_tol) * old_res_norm;
    accept = accept_rank && accept_cap && accept_res && accept_map && accept_not_worse;

    info.cap = cap;
    info.rank = r_keep;
    info.kappa = kappa_map;
    info.rho_tr = rho_tr;
    info.rel_new = new_res_norm/(norm(ctx.y_orb_current_W(:),2)+SAB_eps);

    if ~accept
        if ~accept_cap
            info.reason = 'cap_too_small';
        elseif ~accept_res
            info.reason = 'res_not_down';
        elseif ~accept_not_worse
            info.reason = 'res_worse_rollback';
        elseif ~accept_map
            info.reason = 'map_kappa_large';
        else
            info.reason = 'unknown';
        end
        return;
    end

    r_old_cols = orb_basis_column_count_local(ctx);
    G_corr = H * T_block;

    for tt = 1:numel(ctx.orb_terms)
        ctx.orb_terms{tt}.T = [ctx.orb_terms{tt}.T, -ctx.orb_terms{tt}.T * G_corr];
    end

    raw_m = get_raw_m_local(term);
    term.T = [zeros(raw_m,r_old_cols),T_block];
    ctx.orb_terms{end+1} = term;

    % These work arrays are no longer needed.  Releasing them before the
    % concatenation lowers the peak memory without changing the update.
    clear C C_perp U_r V_r s_r H G_corr gamma_new T_block
    ctx = append_orb_basis_block_local(ctx,Z);
    ctx.gamma_orb_current = gamma_trial;
    ctx.r_orb_current_W = r_trial;
    ctx.orb_block_count = ctx.orb_block_count + 1;

    info.l2_new = NaN;
end

function [ctx,accept,info] = append_family_block_frozen_local(ctx,term,family,cfg,C)
%APPEND_FAMILY_BLOCK_FROZEN_LOCAL
% Residual-driven block enrichment for nonlinear PDEs.
% The block is appended as raw features; the augmented coefficients are
% warm-started by the frozen-Jacobian projection and refined by full Newton.
% The block is accepted only when the true nonlinear residual decreases;
% otherwise the trial is rolled back and RBE terminates.

    SAB_eps = ctx.SAB_eps;

    rel_before = norm(ctx.r_orb_current_W(:),2) / ...
                 (norm(ctx.y_orb_current_W(:),2) + SAB_eps);
    info = struct();
    info.reason = '';
    info.rho_tr = 0;
    info.rel_new = rel_before;
    info.l2_new = NaN;
    info.cap = 0;
    info.rank = 0;
    info.kappa = inf;

    if nargin < 5 || isempty(C)
        C = candidate_residual_matrix_local(ctx,term,family,cfg);
    end

    [~,C_perp] = project_from_orb_basis_local(ctx,C);
    [U_r,V_r,s_r,r_keep] = YY_utils('svd_truncate',C_perp,ctx.orth_cfg);

    if isempty(s_r) || max(s_r) == 0 || r_keep < 1
        accept = false;
        info.reason = 'rank_zero';
        return;
    end

    Z = U_r;
    T_block = bsxfun(@rdivide,V_r,s_r.');
    kappa_map = norm(T_block,2);
    cap = norm(Z' * ctx.r_orb_current_W)^2 / ...
          (norm(ctx.r_orb_current_W)^2 + SAB_eps);

    if cap < cfg.tau_cap
        accept = false;
        info.reason = 'cap_too_small';
        return;
    end
    if kappa_map > cfg.kappa_map_max
        accept = false;
        info.reason = 'map_kappa_large';
        return;
    end

    use_incremental_init = isfield(cfg,'incremental_init') && ...
        ~isempty(cfg.incremental_init) && logical(cfg.incremental_init);

    % Trial state: append the raw block.  The frozen-Jacobian RBE state keeps
    % raw coefficients (T=I), while Q is only a compressed scoring space;
    % therefore the incremental path must not multiply raw T by H.  It
    % solves only the new raw block using the small SVD map T_block.
    raw_m = get_raw_m_local(term);
    r_old_cols = numel(ctx.gamma_orb_current);

    ctx_try = ctx;
    for tt = 1:numel(ctx_try.orb_terms)
        mi = size(ctx_try.orb_terms{tt}.T,1);
        ctx_try.orb_terms{tt}.T = [ctx_try.orb_terms{tt}.T, zeros(mi,raw_m)];
    end
    term2 = term;
    term2.T = [zeros(raw_m,r_old_cols),eye(raw_m)];
    ctx_try.orb_terms{end+1} = term2;

    if use_incremental_init
        % Incremental coefficient initialization in the raw coefficient
        % space: retain old coefficients and solve only the new block.
        % T_block maps the new orthogonal response coefficients back to the
        % raw candidate columns, so this remains valid even when the frozen
        % scoring basis is rank-truncated relative to the raw basis.
        gamma_new = T_block * (Z' * ctx.r_orb_current_W);
        ctx_try.gamma_orb_current = [ctx.gamma_orb_current(:); gamma_new(:)];
        fprintf('  [RBE] incremental warm start: block_rank=%d, raw_block=%d, old_raw=%d\n', ...
            r_keep, raw_m, r_old_cols);
    else
        % Legacy warm start: value projection vs Newton-linearized solve on
        % the complete augmented basis, followed by optional full Newton.
        basisEvalOrbN = @(ord,dir,X) RBE_Eval('terms', ctx_try.orb_terms, X, ord, dir);
        solEvalCur = @(ord,dir,X) current_solution_local(ctx, X, ord, dir);
        consR = get_problem_constraints_local(ctx);
        etaR = ctx.eta0;
        if isfield(cfg,'newton') && isfield(cfg.newton,'eta_bc') && ~isempty(cfg.newton.eta_bc)
            etaR = cfg.newton.eta_bc;
        end
        svdCfgR = struct();
        if isfield(cfg,'newton') && ~isempty(cfg.newton), svdCfgR = cfg.newton; end
        [gamma_fit, initInfoR] = bestInitSVD(ctx.problem, basisEvalOrbN, solEvalCur, consR, ...
            ctx.x_point, etaR, svdCfgR, ctx.GPU, ctx.QR);
        ctx_try.gamma_orb_current = gamma_fit(:);
        fprintf('  [RBE] warm start: use=%s, rel_value=%.4e, rel_linearized=%.4e\n', ...
            initInfoR.method, initInfoR.rel_value, initInfoR.rel_linearized);
    end

    % For the fixed-operator path, the block initializer is only a warm
    % start.  Refine the trial augmented space before the acceptance test;
    % otherwise a locally initialized block can look worse even though the
    % enlarged space contains the old solution (new coefficients = 0).
    if use_incremental_init && isfield(cfg,'fixed_operator') && ...
            isstruct(cfg.fixed_operator) && ...
            isfield(cfg.fixed_operator,'enabled') && cfg.fixed_operator.enabled
        % The zero-extension of the old coefficients is an exact feasible
        % point in the enlarged space.  If the local SVD warm start is worse
        % than that point for the true nonlinear residual, fall back to zero
        % new coefficients before the fixed-operator refinement.
        ctx_zero = ctx_try;
        ctx_zero.gamma_orb_current = [ctx.gamma_orb_current(:); zeros(raw_m,1)];
        R_seed = true_nonlinear_residual_local(ctx_try,cfg);
        R_zero = true_nonlinear_residual_local(ctx_zero,cfg);
        if norm(R_zero(:),2) <= norm(R_seed(:),2)
            ctx_try.gamma_orb_current = ctx_zero.gamma_orb_current;
        end
        consF = get_problem_constraints_local(ctx);
        if ~isempty(consF)
            basisEvalF = @(ord,dir,X) RBE_Eval('terms',ctx_try.orb_terms,X,ord,dir);
            fixed_cfg_trial = cfg.fixed_operator;
            % Reuse the factorization committed by the last accepted RBE
            % block.  A missing state means this is the initial RBE space;
            % FixedOperatorSolver performs the one-time full factorization.
            if isfield(ctx,'fixed_operator_svd_state') && ...
                    ~isempty(ctx.fixed_operator_svd_state)
                fixed_cfg_trial.svd_state = ctx.fixed_operator_svd_state;
            elseif isfield(fixed_cfg_trial,'svd_state')
                fixed_cfg_trial = rmfield(fixed_cfg_trial,'svd_state');
            end
            % The accepted RBE term is the only new block.  Let the fixed
            % solver assemble just this block's differential/boundary rows
            % for the incremental SVD update; full-basis evaluation remains
            % available inside the Picard residual and line search.
            term_block = term;
            term_block.T = eye(raw_m);
            fixed_cfg_trial.block_basis_eval = @(ord,dir,X) ...
                RBE_Eval('terms',{term_block},X,ord,dir);
            t_fixed_trial = tic;
            [ctx_try.gamma_orb_current, fiR] = FixedOperatorSolver( ...
                ctx.problem,basisEvalF,ctx.x_point,consF, ...
                ctx_try.gamma_orb_current,fixed_cfg_trial);
            fixed_trial_seconds = toc(t_fixed_trial);
            ctx_try.fixed_operator_svd_state = fiR.svd_state;
            info.fixed_operator_iter = fiR.iter;
            info.fixed_operator_rel_res = fiR.rel_res;
            info.fixed_operator_factorization = fiR.factorization;
            info.fixed_operator_seconds = fixed_trial_seconds;
            fprintf('  [RBE] fixed-operator trial refine: loss=%.4e, rel_res=%.4e, iter=%d, factor=%s, time=%.2fs\n', ...
                fiR.loss,fiR.rel_res,fiR.iter,fiR.factorization,fixed_trial_seconds);
        end
    end

    if isfield(cfg,'run_newton_each_block') && cfg.run_newton_each_block && ...
       isfield(cfg,'newton') && ~isempty(cfg.newton)
        consN = get_problem_constraints_local(ctx);
        if ~isempty(consN)
            basisEvalOrb = @(ord,dir,X) RBE_Eval('terms', ctx_try.orb_terms, X, ord, dir);
            [ctx_try.gamma_orb_current, nwB] = NewtonSolver( ...
                ctx.problem, basisEvalOrb, ctx.x_point, consN, ...
                ctx_try.gamma_orb_current, cfg.newton);
            info.newton_rel = nwB.rel_res;
        end
    end

    R_after = true_nonlinear_residual_local(ctx_try,cfg);
    rel_after = norm(R_after(:),2) / (norm(ctx.y_orb_current_W(:),2) + SAB_eps);
    info.rel_new = rel_after;
    info.l2_new = NaN;
    info.rho_tr = (rel_before - rel_after) / (rel_before + SAB_eps);
    info.cap = cap;
    info.rank = r_keep;
    info.kappa = kappa_map;

    residual_ok = rel_after < rel_before;
    if residual_ok
        accept = true;
        ctx = ctx_try;
        % Update the scoring basis with the new residual directions and
        % recompute the true weighted residual at the refined solution.
        ctx = append_orb_basis_block_local(ctx,Z);
        ctx.r_orb_current_W = R_after;
        ctx.orb_block_count = ctx.orb_block_count + 1;
        info.reason = '';
    else
        accept = false;
        info.reason = 'res_not_down';
    end
end

%% ============================================================
% r-training wrapper
%% ============================================================
function [r_opt,MSE_opt,alpha_opt,term_opt,hist] = train_r_generic_local( ...
    ctx,X,target,raw,r0,opts,op,family,lambda_ridge,cfg)

    if isfield(cfg,'use_frozen_jacobian') && cfg.use_frozen_jacobian
        % Nonlinear case: train r against the frozen Newton-Jacobian response
        % with the ANALYTIC dM/dr and relative-gradient Adam (matches the
        % legacy ADRaNN 'kdv_residual' mode, so r keeps moving even when the
        % residual is tiny).
        solX = struct();
        solX.u    = current_solution_local(ctx, X, 0, 0);
        solX.ux   = current_solution_local(ctx, X, 1, 1);
        solX.ut   = current_solution_local(ctx, X, 1, 2);
        solX.uxx  = current_solution_local(ctx, X, 2, 1);
        solX.uxxx = current_solution_local(ctx, X, 3, 1);
        extra = struct();
        extra.problem = ctx.problem;
        extra.sol = solX;
        [r_opt,MSE_opt,alpha_opt,term_opt,hist] = ADRaNN_train_r_analytic( ...
            'frozen_jacobian',X,target,raw,r0,opts,family,lambda_ridge,extra);
    else
        extra = struct();
        extra.op = op;
        [r_opt,MSE_opt,alpha_opt,term_opt,hist] = ADRaNN_train_r_analytic( ...
            'operator',X,target,raw,r0,opts,family,lambda_ridge,extra);
    end
end

function [r_opt,MSE_opt,alpha_opt,term_opt,hist] = train_r_frozen_local( ...
    ctx,X,target,raw,r0,opts,family,lambda_ridge)
%TRAIN_R_FROZEN_LOCAL
% r-training for nonlinear PDEs against the frozen Newton-Jacobian response:
%   min_r min_a || J(u) Phi(r) a - target ||^2 + lambda ||a||^2
% The Jacobian coefficients are frozen at the current solution on the
% training points (matching the legacy KDV_Newton_module 'train_r_residual').

    X = gather(X);
    target = gather(target(:));
    r = gather(r0(:));
    opts = fill_adam_opts_local(opts,r);

    sol = struct();
    sol.u    = current_solution_local(ctx, X, 0, 0);
    sol.ux   = current_solution_local(ctx, X, 1, 1);
    sol.ut   = current_solution_local(ctx, X, 1, 2);
    sol.uxx  = current_solution_local(ctx, X, 2, 1);
    sol.uxxx = current_solution_local(ctx, X, 3, 1);
    Jc = ProblemFunction('jacobian', X, ctx.problem, sol);

    beta1 = getfield_default_local(opts,'beta1',0.9);
    beta2 = getfield_default_local(opts,'beta2',0.999);
    eps_adam = getfield_default_local(opts,'eps_adam',1e-8);

    clip_min = expand_clip_local(opts.clip_min,numel(r),-inf);
    clip_max = expand_clip_local(opts.clip_max,numel(r), inf);

    m_adam = zeros(size(r));
    v_adam = zeros(size(r));

    MSE_opt = inf;
    r_opt = r;
    alpha_opt = [];
    best_iter = 0;

    hist.MSE = nan(max(opts.maxit,1),1);
    hist.r = nan(numel(r),max(opts.maxit,1));
    hist.grad = nan(numel(r),max(opts.maxit,1));
    hist.grad_norm = nan(max(opts.maxit,1),1);

    for it = 1:opts.maxit
        [MSE,alpha] = objective_frozen_local(X,target,raw,r,Jc,family,lambda_ridge);
        g = gradient_fd_local(@(rr)objective_frozen_local(X,target,raw,rr,Jc,family,lambda_ridge), ...
            r,clip_min,clip_max,opts.fd_step_rel);

        hist.MSE(it) = MSE;
        hist.r(:,it) = r;
        hist.grad(:,it) = g;
        hist.grad_norm(it) = norm(g);

        if MSE < MSE_opt
            MSE_opt = MSE;
            r_opt = r;
            alpha_opt = alpha;
            best_iter = it;
        end

        if it >= opts.minit && norm(g) < opts.grad_tol
            break;
        end

        g_used = g;
        if opts.use_rel_grad
            g_used = g_used ./ (abs(r)+1e-12);
        end

        m_adam = beta1*m_adam + (1-beta1)*g_used;
        v_adam = beta2*v_adam + (1-beta2)*(g_used.^2);
        mh = m_adam/(1-beta1^it);
        vh = v_adam/(1-beta2^it);

        r = r - opts.eta*mh./(sqrt(vh)+eps_adam);
        r = min(max(r,clip_min),clip_max);
    end

    [MSE_final,alpha_final] = objective_frozen_local(X,target,raw,r,Jc,family,lambda_ridge);
    if isfinite(MSE_final) && MSE_final < MSE_opt
        MSE_opt = MSE_final;
        r_opt = r;
        alpha_opt = alpha_final;
        best_iter = opts.maxit;
    end

    term_opt = build_term_from_r_local(family,raw,r_opt);

    valid = isfinite(hist.MSE);
    hist.MSE = hist.MSE(valid);
    hist.r = hist.r(:,valid);
    hist.grad = hist.grad(:,valid);
    hist.grad_norm = hist.grad_norm(valid);
    hist.best_iter = best_iter;
    hist.best_MSE = MSE_opt;
    hist.best_r = r_opt(:);
end

function [MSE,alpha] = objective_frozen_local(X,target,raw,r,Jc,family,lambda)
    term = build_term_from_r_local(family,raw,r);
    Phi    = BasisFunction(family,term,X,0,0);
    Phix   = BasisFunction(family,term,X,1,1);
    Phit   = BasisFunction(family,term,X,1,2);
    Phixxx = BasisFunction(family,term,X,3,1);
    A = Jc.ut .* Phit + Jc.ux .* Phix + Jc.uxxx .* Phixxx;
    if isfield(Jc,'u')
        A = A + Jc.u .* Phi;
    end
    if isfield(Jc,'uxx')
        Phixx = BasisFunction(family,term,X,2,1);
        A = A + Jc.uxx .* Phixx;
    end
    A = gather(A);
    K = size(A,2);
    G = A.'*A + lambda*eye(K);
    rhs = A.'*target(:);
    [L,p] = chol(G,'lower');
    if p == 0
        alpha = L'\(L\rhs);
    else
        alpha = G\rhs;
    end
    res = A*alpha - target(:);
    MSE = mean(res.^2);
end

function opts = fill_adam_opts_local(opts,r)
    if ~isfield(opts,'maxit') || isempty(opts.maxit), opts.maxit = 30; end
    if ~isfield(opts,'eta') || isempty(opts.eta), opts.eta = 0.25; end
    if ~isfield(opts,'eps_adam') || isempty(opts.eps_adam), opts.eps_adam = 1e-8; end
    if ~isfield(opts,'clip_min') || isempty(opts.clip_min), opts.clip_min = ones(numel(r),1); end
    if ~isfield(opts,'clip_max') || isempty(opts.clip_max), opts.clip_max = 150*ones(numel(r),1); end
    if ~isfield(opts,'minit') || isempty(opts.minit), opts.minit = 5; end
    if ~isfield(opts,'grad_tol') || isempty(opts.grad_tol), opts.grad_tol = 1e-12; end
    if ~isfield(opts,'fd_step_rel') || isempty(opts.fd_step_rel), opts.fd_step_rel = 1e-4; end
    if ~isfield(opts,'verbose') || isempty(opts.verbose), opts.verbose = false; end
    if ~isfield(opts,'use_rel_grad') || isempty(opts.use_rel_grad), opts.use_rel_grad = false; end
end

function clip = expand_clip_local(x,n,default_val)
    if isempty(x)
        clip = default_val*ones(n,1);
        return;
    end
    x = x(:);
    if numel(x) == 1
        clip = x*ones(n,1);
    elseif numel(x) == n
        clip = x;
    else
        error('clip_min/clip_max length mismatch: got %d, expected 1 or %d.',numel(x),n);
    end
end

function g = gradient_fd_local(fun,r,clip_min,clip_max,fd_step_rel)
    g = zeros(size(r));
    for kk = 1:numel(r)
        h = fd_step_rel * max(1,abs(r(kk)));
        rp = r;
        rm = r;
        rp(kk) = min(rp(kk)+h,clip_max(kk));
        rm(kk) = max(rm(kk)-h,clip_min(kk));
        if abs(rp(kk)-rm(kk)) < eps
            g(kk) = 0;
        else
            fp = fun(rp);
            fm = fun(rm);
            g(kk) = (fp-fm)/(rp(kk)-rm(kk));
        end
    end
end

%% ============================================================
% problem op
%% ============================================================
function op = get_problem_op_local(ctx)
    if ~isfield(ctx,'problem') || isempty(ctx.problem)
        error('ctx.problem does not exist.please main set in ctx.problem = ProblemFunction(''setup'');');
    end

    problem = ProblemFunction('complete',[],ctx.problem);

    if ~isfield(problem,'op') || isempty(problem.op)
        error('ctx.problem.op does not exist. Define it in ProblemFunction(''setup'').');
    end

    op = problem.op;
end

%% ============================================================
% frozen Newton-Jacobian residual space (nonlinear PDEs)
%% ============================================================

function A = int_residual_response_local(ctx, term, family, cfg)
%INT_RESIDUAL_RESPONSE_LOCAL
% Interior residual-response matrix for a candidate block.
% For linear PDEs this is the fixed operator BasisOperator(...).
% For nonlinear PDEs with cfg.use_frozen_jacobian=true it is the Newton
% linearization at the current solution (frozen Jacobian), matching the
% legacy KDV_Newton_module 'freeze_jacobian' / 'jref_matrix_for_term'
% scoring.

    if cfg.use_frozen_jacobian
        A = frozen_jacobian_int_local(ctx, term, family);
    else
        A = BasisOperator(family, term, ctx.x_point, cfg.op);
    end
end

function A = frozen_jacobian_int_local(ctx, term, family)
%FROZEN_JACOBIAN_INT_LOCAL
% Interior part of the frozen Newton-Jacobian response of a candidate term:
%   J(u) phi = Jc.ut*phi_t + Jc.ux*phi_x + Jc.uxxx*phi_xxx
%            + (Jc.u*phi + Jc.uxx*phi_xx when present),
% where the Jacobian coefficients Jc are evaluated at the current solution u.

    X = ctx.x_point;
    sol = struct();
    sol.u    = current_solution_local(ctx, X, 0, 0);
    sol.ux   = current_solution_local(ctx, X, 1, 1);
    sol.ut   = current_solution_local(ctx, X, 1, 2);
    sol.uxx  = current_solution_local(ctx, X, 2, 1);
    sol.uxxx = current_solution_local(ctx, X, 3, 1);

    Jc = ProblemFunction('jacobian', X, ctx.problem, sol);

    Phi    = BasisFunction(family, term, X, 0, 0);
    Phix   = BasisFunction(family, term, X, 1, 1);
    Phit   = BasisFunction(family, term, X, 1, 2);
    Phixx  = BasisFunction(family, term, X, 2, 1);
    Phixxx = BasisFunction(family, term, X, 3, 1);

    A = Jc.ut .* Phit + Jc.ux .* Phix + Jc.uxxx .* Phixxx;
    if isfield(Jc, 'u')
        A = A + Jc.u .* Phi;
    end
    if isfield(Jc, 'uxx')
        A = A + Jc.uxx .* Phixx;
    end
end

function [Q, R, yW] = frozen_jacobian_state_local(ctx, term0, cfg)
%FROZEN_JACOBIAN_STATE_LOCAL
% Rebuild the scoring basis Q and the current weighted residual R under the
% frozen Newton Jacobian at the current solution.
%   R = [F(u); eta0*(u-g)]   (true nonlinear interior residual + boundary)
%   Q = orth([J(u)*Phi; eta0*Phi_b])
% Only scoring objects are returned; the RBE solution stays in raw terms +
% raw coefficients (set in init_orb_state_local).

    X = ctx.x_point;
    sol = struct();
    sol.u    = current_solution_local(ctx, X, 0, 0);
    sol.ux   = current_solution_local(ctx, X, 1, 1);
    sol.ut   = current_solution_local(ctx, X, 1, 2);
    sol.uxx  = current_solution_local(ctx, X, 2, 1);
    sol.uxxx = current_solution_local(ctx, X, 3, 1);
    F_int = ProblemFunction('residual', X, ctx.problem, sol);

    % boundary residual computed by bc_residual_local
    r_bc = bc_residual_local(ctx, cfg);
    R = gather([F_int; r_bc]);
    yW = gather([ctx.right; bc_target_local(ctx, cfg)]);

    B_int = frozen_jacobian_int_local(ctx, term0, 'tanh');
    B_bd  = bc_matrix_local(ctx, term0, 'tanh', cfg);
    B_full = gather([B_int; B_bd]);
    [U_r, ~, ~] = YY_utils('svd_truncate', B_full, ctx.orth_cfg);
    Q = U_r;
end

function R = true_nonlinear_residual_local(ctx, cfg)
%TRUE_NONLINEAR_RESIDUAL_LOCAL
% Weighted nonlinear residual [F(u); eta0*(u-g)] of the current RBE solution.

    X = ctx.x_point;
    sol = struct();
    sol.u    = current_solution_local(ctx, X, 0, 0);
    sol.ux   = current_solution_local(ctx, X, 1, 1);
    sol.ut   = current_solution_local(ctx, X, 1, 2);
    sol.uxx  = current_solution_local(ctx, X, 2, 1);
    sol.uxxx = current_solution_local(ctx, X, 3, 1);
    F_int = ProblemFunction('residual', X, ctx.problem, sol);
    r_bc = bc_residual_local(ctx, cfg);
    R = gather([F_int; r_bc]);
end

function v = current_solution_local(ctx, X, ord, dir)
%CURRENT_SOLUTION_LOCAL
% Current solution (or its derivative) at points X, in whichever state the
% context currently represents it (ORB terms, orth basis, or raw tanh).

    if isfield(ctx, 'orb_initialized') && ctx.orb_initialized == 1
        v = RBE_Eval('solution', ctx.orb_terms, X, ord, dir, ctx.gamma_orb_current);
    elseif isfield(ctx, 'use_orth_basis') && ctx.use_orth_basis == 1 && ...
           isfield(ctx, 'T_current') && isfield(ctx, 'gamma_current')
        term_tmp = struct('w', ctx.w, 'b', ctx.b);
        v = BasisFunction('tanh', term_tmp, X, ord, dir) * ctx.T_current * ctx.gamma_current;
    else
        term_tmp = struct('w', ctx.w, 'b', ctx.b);
        v = BasisFunction('tanh', term_tmp, X, ord, dir) * ctx.weight;
    end
    v = gather(v);
end

%% ============================================================
% boundary/initial response helpers (value / ux / uxx / initial)
%% ============================================================

function B = bc_matrix_local(ctx, term, family, cfg)
%BC_MATRIX_LOCAL
% Weighted boundary/initial response matrix of a candidate term.
% When cfg.use_frozen_jacobian is true, use the full PDE constraints
% (value / ux / uxx / initial) from ProblemFunction('constraints'), matching
% the legacy KDV_Newton_module boundary handling. Otherwise keep the legacy
% value-only boundary (ctx.x_0) at weight ctx.eta0.

    if cfg.use_frozen_jacobian
        cons = get_problem_constraints_local(ctx);
        if isempty(cons)
            B = ctx.eta0 * BasisFunction(family, term, ctx.x_0, 0, 0);
            B = [B; neumann_flux_matrix_local(family,term,ctx)];
            return;
        end
        X = cons.X; types = cons.types;
        idxV   = strcmpi(types, 'value');
        idxUx  = strcmpi(types, 'ux');
        idxUxx = strcmpi(types, 'uxx');
        eta = ctx.eta0;
        B = [eta * BasisFunction(family, term, X(idxV, :), 0, 0);
             eta * BasisFunction(family, term, X(idxUx, :), 1, 1);
             eta * BasisFunction(family, term, X(idxUxx, :), 2, 1);
             neumann_flux_matrix_local(family,term,ctx)];
    else
        B = ctx.eta0 * BasisFunction(family, term, ctx.x_0, 0, 0);
        B = [B; neumann_flux_matrix_local(family,term,ctx)];
    end
end

function y = bc_target_local(ctx, cfg)
%BC_TARGET_LOCAL weighted boundary/initial target vector.

    if cfg.use_frozen_jacobian
        cons = get_problem_constraints_local(ctx);
        if isempty(cons)
            y = ctx.eta0 * ctx.y0_exact;
            y = [y; zeros(neumann_row_count_local(ctx),1)];
            return;
        end
        types = cons.types; target = cons.target(:);
        idxV   = strcmpi(types, 'value');
        idxUx  = strcmpi(types, 'ux');
        idxUxx = strcmpi(types, 'uxx');
        eta = ctx.eta0;
        y = [eta * target(idxV); eta * target(idxUx); eta * target(idxUxx); ...
             zeros(neumann_row_count_local(ctx),1)];
    else
        y = ctx.eta0 * ctx.y0_exact;
        y = [y; zeros(neumann_row_count_local(ctx),1)];
    end
end

function r = bc_residual_local(ctx, cfg)
%BC_RESIDUAL_LOCAL weighted boundary/initial residual of the current solution.

    if cfg.use_frozen_jacobian
        cons = get_problem_constraints_local(ctx);
        if isempty(cons)
            u_bd = current_solution_local(ctx, ctx.x_0, 0, 0);
            r = ctx.eta0 * (u_bd - ctx.y0_exact);
            r = [r; neumann_flux_residual_local(ctx)];
            return;
        end
        X = cons.X; types = cons.types; target = cons.target(:);
        idxV   = strcmpi(types, 'value');
        idxUx  = strcmpi(types, 'ux');
        idxUxx = strcmpi(types, 'uxx');
        eta = ctx.eta0;
        uV   = current_solution_local(ctx, X(idxV, :), 0, 0);
        uUx  = current_solution_local(ctx, X(idxUx, :), 1, 1);
        uUxx = current_solution_local(ctx, X(idxUxx, :), 2, 1);
        r = [eta * (uV - target(idxV));
             eta * (uUx - target(idxUx));
             eta * (uUxx - target(idxUxx));
             neumann_flux_residual_local(ctx)];
    else
        u_bd = current_solution_local(ctx, ctx.x_0, 0, 0);
        r = ctx.eta0 * (u_bd - ctx.y0_exact);
        r = [r; neumann_flux_residual_local(ctx)];
    end
end

function B = neumann_flux_matrix_local(family,term,ctx)
%NEUMANN_FLUX_MATRIX_LOCAL Weighted natural-boundary flux response rows.
    info = [];
    if isfield(ctx,'neumann_info'), info = ctx.neumann_info; end
    if isempty(info) || ~isstruct(info) || ~isfield(info,'X') || isempty(info.X)
        if isfield(term,'w'), ncol = size(term.w,2); elseif isfield(term,'A'), ncol = size(term.A,2); else, ncol = 0; end
        B = zeros(0,ncol);
        return;
    end
    eta = getfield_default_local(ctx,'eta_neumann',1.0);
    X = info.X;
    phix = BasisFunction(family,term,X,1,1);
    phiy = BasisFunction(family,term,X,1,2);
    B = eta * info.coefficient(:) .* ...
        (info.normal(:,1).*phix + info.normal(:,2).*phiy);
end

function n = neumann_row_count_local(ctx)
    n = 0;
    if isfield(ctx,'neumann_info') && isstruct(ctx.neumann_info) && ...
            isfield(ctx.neumann_info,'X')
        n = size(ctx.neumann_info.X,1);
    end
end

function r = neumann_flux_residual_local(ctx)
%NEUMANN_FLUX_RESIDUAL_LOCAL Natural-boundary flux residual.
    if neumann_row_count_local(ctx) == 0
        r = zeros(0,1);
        return;
    end
    info = ctx.neumann_info;
    eta = getfield_default_local(ctx,'eta_neumann',1.0);
    ux = current_solution_local(ctx,info.X,1,1);
    uy = current_solution_local(ctx,info.X,1,2);
    r = eta * info.coefficient(:) .* ...
        (info.normal(:,1).*ux + info.normal(:,2).*uy);
end

function c = current_raw_coef_local(ctx)
%CURRENT_RAW_COEF_LOCAL current solution coefficients in the raw tanh basis.

    if isfield(ctx, 'use_orth_basis') && ctx.use_orth_basis == 1 && ...
       isfield(ctx, 'T_current') && isfield(ctx, 'gamma_current')
        c = ctx.T_current * ctx.gamma_current;
    else
        c = ctx.weight;
    end
    c = gather(c(:));
end

function n = orb_basis_column_count_local(ctx)
%ORB_BASIS_COLUMN_COUNT_LOCAL total rank of the block-stored response basis.
    if isfield(ctx,'Q_orb_block_sizes') && ~isempty(ctx.Q_orb_block_sizes)
        n = sum(ctx.Q_orb_block_sizes(:,2));
    elseif isfield(ctx,'Q_orb_blocks') && ~isempty(ctx.Q_orb_blocks)
        n = 0;
        for kk = 1:numel(ctx.Q_orb_blocks)
            n = n + size(ctx.Q_orb_blocks{kk},2);
        end
    else
        n = size(ctx.Q_orb_current,2);
    end
end

function [H,C_perp] = project_from_orb_basis_local(ctx,C)
%PROJECT_FROM_ORB_BASIS_LOCAL apply Q'*C and C-Q*(Q'*C) by Q blocks.
% The row-chunked subtraction avoids a full-size Q_k*H_k temporary.
    if isfield(ctx,'Q_orb_blocks') && ~isempty(ctx.Q_orb_blocks)
        ncols = orb_basis_column_count_local(ctx);
        H = zeros(ncols,size(C,2),'like',C);
        C_perp = C;
        col0 = 0;
        row_chunk = 4096;
        for kk = 1:numel(ctx.Q_orb_blocks)
            if isempty(ctx.Q_orb_blocks{kk}) && ...
                    isfield(ctx,'Q_orb_block_files') && ...
                    numel(ctx.Q_orb_block_files) >= kk && ...
                    ~isempty(ctx.Q_orb_block_files{kk})
                loaded = load(ctx.Q_orb_block_files{kk},'Q_block');
                Qk = loaded.Q_block;
                clear loaded
            else
                Qk = ctx.Q_orb_blocks{kk};
            end
            nk = size(Qk,2);
            cols = col0 + (1:nk);
            H(cols,:) = Qk' * C;
            Hk = H(cols,:);
            for row0 = 1:row_chunk:size(C,1)
                rows = row0:min(row0+row_chunk-1,size(C,1));
                C_perp(rows,:) = C_perp(rows,:) - Qk(rows,:)*Hk;
            end
            col0 = col0 + nk;
            clear Qk Hk
        end
    else
        H = ctx.Q_orb_current' * C;
        C_perp = C - ctx.Q_orb_current * H;
    end
end

function ctx = append_orb_basis_block_local(ctx,Z)
%APPEND_ORB_BASIS_BLOCK_LOCAL append a response block in memory or on disk.
    if isfield(ctx,'orb_basis_cache_dir') && ~isempty(ctx.orb_basis_cache_dir)
        block_id = numel(ctx.Q_orb_block_files) + 1;
        block_file = fullfile(ctx.orb_basis_cache_dir, ...
            sprintf('q_block_%04d.mat',block_id));
        Q_block = Z; %#ok<NASGU>
        save(block_file,'Q_block','-v7.3');
        ctx.Q_orb_blocks{end+1} = [];
        ctx.Q_orb_block_files{end+1} = block_file;
    else
        ctx.Q_orb_blocks{end+1} = Z;
        if ~isfield(ctx,'Q_orb_block_files')
            ctx.Q_orb_block_files = {};
        end
        ctx.Q_orb_block_files{end+1} = '';
    end
    if ~isfield(ctx,'Q_orb_block_sizes') || isempty(ctx.Q_orb_block_sizes)
        ctx.Q_orb_block_sizes = zeros(0,2);
    end
    ctx.Q_orb_block_sizes(end+1,:) = [size(Z,1),size(Z,2)];
end

function cleanup_orb_basis_cache_local(cache_dir)
%CLEANUP_ORB_BASIS_CACHE_LOCAL remove only this run's private temp folder.
    if ~isempty(cache_dir) && isfolder(cache_dir)
        rmdir(cache_dir,'s');
    end
end
