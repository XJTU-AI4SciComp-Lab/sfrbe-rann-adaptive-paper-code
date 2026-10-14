function m = computeErrors(pred_u, pred_ux, pred_uy, ex, xp1, xp2)
%COMPUTEERRORS Relative/absolute L2 and H1 errors on the eval grid.
%   pred_u/pred_ux/pred_uy : predicted value / x-derivative / y-derivative (column vectors)
%   ex                     : struct with u, ux, uy (exact values on the same points)
%   xp1, xp2               : eval grid coordinates
%   Returns m with fields: rel_L2, abs_L2, rel_H1, abs_H1.
    pred_u  = gather(pred_u(:));
    pred_ux = gather(pred_ux(:));
    pred_uy = gather(pred_uy(:));
    e_u  = pred_u  - ex.u(:);
    e_ux = pred_ux - ex.ux(:);
    e_uy = pred_uy - ex.uy(:);

    dx = (xp1(end) - xp1(1)) / (numel(xp1) - 1);
    dy = (xp2(end) - xp2(1)) / (numel(xp2) - 1);
    w = dx * dy;

    errL2 = sqrt(w * sum(e_u.^2));
    errH1 = sqrt(errL2^2 + w * sum(e_ux.^2 + e_uy.^2));
    exL2  = sqrt(w * sum(ex.u(:).^2));
    exH1  = sqrt(exL2^2 + w * sum(ex.ux(:).^2 + ex.uy(:).^2));

    m = struct();
    m.rel_L2 = errL2 / (exL2 + 1e-12);
    m.abs_L2 = errL2;
    m.rel_H1 = errH1 / (exH1 + 1e-12);
    m.abs_H1 = errH1;
end