function makePaperFigures(cfg, pd)
%MAKEPAPERFIGURES export paper figures (compressed PDF + 300dpi PNG).
%   File names follow cfg.paper.fig_file (paper-style names, no figNN_ prefix);
%   fallback is fig<NN>_<problem>_<content>.
%
%   style follows legacy plotting script:
%     - solution/error figures: contourf 80 levels + shading interp + jet + caxis + axis equal tight,
%       720x580 canvas, box on / TickDir in / FontSize 13;
%     - anchor figure: background=blue, structure region 1=red, structure region 2=yellow, other structure regions=green/purple, with legend.
%   pd fields: xp1, xp2, exact, pred_initial, pred_sfr, pred_final,
%            centers(2xm), diagS (with domain_labels/xx1/xx2 for region labels)
    % ---- Fig 06: reference/exact solution / numerical solution / SFR anchors ----
    % Some benchmarks use a high-accuracy numerical reference rather than a
    % closed-form exact solution.  The label is supplied by the experiment
    % configuration so that only those figures are renamed.
    solutionLabel = 'Exact solution';
    if isfield(cfg, 'paper') && isfield(cfg.paper, 'reference_label') && ...
            ~isempty(cfg.paper.reference_label)
        solutionLabel = cfg.paper.reference_label;
    end
    exportHeatmap(cfg, figureTag(cfg, '06', 'exact_sol'), ...
        pd.xp1, pd.xp2, gridField(pd, pd.exact), ...
        solutionLabel, getfield_default_pd(pd, 'mask', []));
    exportHeatmap(cfg, figureTag(cfg, '06', 'num_sol'), ...
        pd.xp1, pd.xp2, gridField(pd, pd.pred_final), ...
        'Final numerical solution', getfield_default_pd(pd, 'mask', []));
    if ~isempty(pd.centers)
        exportAnchors(cfg, figureTag(cfg, '06', 'sfr_anchor'), ...
            pd.centers, pd.diagS);
    end

    % ---- Fig 07: initial / after-SFR / after-RBE errors ----
    exportHeatmap(cfg, figureTag(cfg, '07', 'ini_error'), ...
        pd.xp1, pd.xp2, gridField(pd, abs(pd.exact - pd.pred_initial)), ...
        'Initial absolute error', getfield_default_pd(pd, 'mask', []));
    exportHeatmap(cfg, figureTag(cfg, '07', 'sfr_error'), ...
        pd.xp1, pd.xp2, gridField(pd, abs(pd.exact - pd.pred_sfr)), ...
        'Absolute error after SFR', getfield_default_pd(pd, 'mask', []));
    exportHeatmap(cfg, figureTag(cfg, '07', 'rbe_error'), ...
        pd.xp1, pd.xp2, gridField(pd, abs(pd.exact - pd.pred_final)), ...
        'Absolute error after RBE', getfield_default_pd(pd, 'mask', []));
end

%% ============================================================
function U = gridField(pd, vec)
    vec = vec(:);
    if numel(vec) == numel(pd.xp1) * numel(pd.xp2)
        U = reshape(vec, numel(pd.xp2), numel(pd.xp1));   % already a full grid
    elseif isfield(pd, 'mask') && ~isempty(pd.mask)
        U = nan(numel(pd.xp2), numel(pd.xp1));
        U(pd.mask) = vec;                                  % scatter in-domain values
    else
        U = reshape(vec, numel(pd.xp2), numel(pd.xp1));
    end
end

%% ============================================================
function v = getfield_default_pd(pd, name, default_val)
    if isfield(pd, name) && ~isempty(pd.(name))
        v = pd.(name);
    else
        v = default_val;
    end
end

%% ============================================================
function tag = figureTag(cfg, nn, content)
    if isfield(cfg.paper, 'fig_file') && isfield(cfg.paper.fig_file, content)
        tag = cfg.paper.fig_file.(content);
    else
        tag = ['fig', nn, '_', cfg.problem, '_', content];
    end
end

%% ============================================================
function exportHeatmap(cfg, tag, xp1, xp2, U, titleStr, mask)
    if nargin < 7, mask = []; end
    figure('Color', 'w', 'Position', [120, 120, 720, 580], 'Visible', 'on');
    ax = axes('Position', [0.13, 0.11, 0.71, 0.815]);
    if isempty(mask)
        contourf(ax, xp1, xp2, U, 80, 'LineColor', 'none');
        shading(ax, 'interp');
    else
        h = imagesc(ax, xp1, xp2, U);
        set(h, 'AlphaData', double(mask));   % cut out points outside the domain
        axis(ax, 'square');
        set(ax, 'YDir', 'normal');
    end
    axis(ax, 'square');
    set(ax, 'YDir', 'normal');

    xlabel(ax, '$x$', 'Interpreter', 'latex', 'FontSize', 15);
    ylabel(ax, '$y$', 'Interpreter', 'latex', 'FontSize', 15);
    title(ax, titleStr, 'Interpreter', 'latex', 'FontSize', 16);

    colormap(ax, jet);
    cb = colorbar(ax);
    set(cb, 'Position', [0.87, 0.11, 0.04, 0.815]);

    if contains(titleStr, 'error') || contains(titleStr, 'Error')
        emax = max(U(:));
        if isfinite(emax) && emax > 0
            caxis(ax, [0, emax]);
        end
    else
        umin = min(U(:));
        umax = max(U(:));
        if isfinite(umin) && isfinite(umax) && umin < umax
            caxis(ax, [umin, umax]);
        end
    end

    set(ax, 'Box', 'on', 'LineWidth', 1.2, 'FontSize', 13, 'TickDir', 'in');

    [pdfP, pngP] = figurePaths(cfg, tag);
    exportFigure(gcf, pdfP, pngP);
end

%% ============================================================
function exportAnchors(cfg, tag, centers, diagS)
    figure('Color', 'w', 'Position', [200, 200, 720, 580], 'Visible', 'on');
    ax = axes('Position', [0.13, 0.11, 0.71, 0.815]);
    hold(ax, 'on');

    labels = labelCenters(centers, diagS);
    maxLabel = max(labels);

    % background=blue, structure region 1=red, structure region 2=yellow, remaining ones usegreen/purple/cyan
    cmap = [
        0.00 0.20 1.00;   % 1 background
        1.00 0.00 0.00;   % 2 structure 1
        1.00 0.85 0.00;   % 3 structure 2
        0.00 0.65 0.20;   % 4
        0.60 0.20 0.80;   % 5
        0.00 0.80 0.80;   % 6
        0.95 0.40 0.00;   % 7
    ];

    legendEntries = {};
    for lab = 1:maxLabel
        idx = (labels == lab);
        if ~any(idx)
            continue;
        end
        if lab == 1
            sz = 12;
        else
            sz = 18;
        end
        col = cmap(min(lab, size(cmap, 1)), :);
        scatter(ax, centers(1, idx), centers(2, idx), sz, col, 'filled');
        if lab == 1
            legendEntries{end+1} = 'background'; %#ok<AGROW>
        else
            legendEntries{end+1} = sprintf('structure region %d', lab - 1); %#ok<AGROW>
        end
    end
    if ~isempty(legendEntries)
        legend(ax, legendEntries, 'Location', 'northwest');
    end

    axis(ax, 'square');
    grid(ax, 'on');
    if isfield(diagS,'bounds') && numel(diagS.bounds) == 4
        xlim(ax, diagS.bounds(1:2));
        ylim(ax, diagS.bounds(3:4));
    else
        xlim(ax, [min(diagS.xx1(:)), max(diagS.xx1(:))]);
        ylim(ax, [min(diagS.xx2(:)), max(diagS.xx2(:))]);
    end

    % Optional ownership overlay.  The SISC configurations use [1,1], so
    % no interior partition lines are drawn and the figure is single-domain.
    if isfield(diagS,'partition_grid') && numel(diagS.partition_grid) == 2 && ...
            isfield(diagS,'bounds') && numel(diagS.bounds) == 4
        pg = double(diagS.partition_grid(:).');
        bx = linspace(diagS.bounds(1),diagS.bounds(2),pg(1)+1);
        by = linspace(diagS.bounds(3),diagS.bounds(4),pg(2)+1);
        xl = xlim(ax); yl = ylim(ax);
        for kk = 2:numel(bx)-1
            plot(ax,[bx(kk),bx(kk)],yl,'k--','LineWidth',0.7, ...
                'HandleVisibility','off');
        end
        for kk = 2:numel(by)-1
            plot(ax,xl,[by(kk),by(kk)],'k--','LineWidth',0.7, ...
                'HandleVisibility','off');
        end
    end

    xlabel(ax, '$x$', 'Interpreter', 'latex', 'FontSize', 15);
    ylabel(ax, '$y$', 'Interpreter', 'latex', 'FontSize', 15);
    title(ax, 'SFR feature anchors', 'Interpreter', 'latex', 'FontSize', 16);

    set(ax, 'Box', 'on', 'LineWidth', 1.2, 'FontSize', 13, 'TickDir', 'in');
    hold(ax, 'off');

    [pdfP, pngP] = figurePaths(cfg, tag);
    exportFigure(gcf, pdfP, pngP);
end

%% ============================================================
function labels = labelCenters(centers, diagS)
    m = size(centers, 2);
    labels = ones(1, m);
    if isfield(diagS, 'domain_labels') && isfield(diagS, 'xx1') && isfield(diagS, 'xx2') ...
            && ~isempty(diagS.domain_labels) && ~isempty(diagS.xx1)
        Ny = numel(diagS.xx2);
        Nx = numel(diagS.xx1);
        labelMat = reshape(diagS.domain_labels, Ny, Nx);
        if isfield(diagS,'bounds') && numel(diagS.bounds) == 4 && ...
                isfield(diagS,'h') && numel(diagS.h) == 2
            b = diagS.bounds;
            ix = floor((centers(1,:)-b(1))/diagS.h(1)) + 1;
            iy = floor((centers(2,:)-b(3))/diagS.h(2)) + 1;
            inside = centers(1,:) >= b(1) & centers(1,:) <= b(2) & ...
                     centers(2,:) >= b(3) & centers(2,:) <= b(4);
            ix = min(max(ix,1),Nx);
            iy = min(max(iy,1),Ny);
            labels = ones(1,m);
            lin = sub2ind([Ny,Nx],iy(inside),ix(inside));
            labels(inside) = labelMat(lin);
            labels(labels < 1) = 1;
        else
            % Compatibility with result files produced by the old nodal grid.
            labels = interp2(diagS.xx1, diagS.xx2, labelMat, ...
                centers(1, :), centers(2, :), 'nearest', 1);
            labels(~isfinite(labels)) = 1;
        end
        labels = round(labels(:)).';
    end
end
