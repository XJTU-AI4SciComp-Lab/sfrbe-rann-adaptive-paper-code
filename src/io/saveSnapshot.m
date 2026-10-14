function saveSnapshot(dirName, tag, xp1, xp2, pred_vec, exact_vec, info, metrics, mask)
%SAVESNAPSHOT Save one stage plotting data (minimal) plus error metrics.
%   File contents: xp1, xp2, U_pred, U_exact, U_error, metrics, info, mask
%   mask: optional logical mask for non-rectangular domains (empty for rectangles).
    if nargin < 8 || isempty(metrics)
        metrics = struct('rel_L2', NaN, 'abs_L2', NaN, 'rel_H1', NaN, 'abs_H1', NaN);
    end
    if nargin < 9, mask = []; end
    pred_vec = gather(pred_vec(:));
    exact_vec = gather(exact_vec(:));
    if isempty(mask)
        U_pred  = reshape(pred_vec, numel(xp2), numel(xp1));
        U_exact = reshape(exact_vec, numel(xp2), numel(xp1));
    else
        U_pred  = nan(numel(xp2), numel(xp1));
        U_exact = nan(numel(xp2), numel(xp1));
        U_pred(mask)  = pred_vec;
        U_exact(mask) = exact_vec;
    end
    U_error = abs(U_exact - U_pred);

    matName = fullfile(dirName, [tag, '.mat']);
    save(matName, 'tag', 'xp1', 'xp2', 'U_pred', 'U_exact', 'U_error', ...
        'metrics', 'info', 'mask');
    fprintf('  [save] %s  Rel.L2=%.4e Abs.L2=%.4e Rel.H1=%.4e Abs.H1=%.4e\n', ...
        matName, metrics.rel_L2, metrics.abs_L2, metrics.rel_H1, metrics.abs_H1);
end