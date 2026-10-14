function varargout = YY_utils(action, varargin)
%YY_UTILS a few generic utility functions in one file.

    switch lower(action)
        case 'solve_af'
            varargout{1} = Solve_AF_local(varargin{:});

        case 'get_inner_point'
            varargout{1} = get_inner_point_local(varargin{:});

        case 'label_components_8conn'
            [varargout{1},varargout{2}] = label_components_8conn_local(varargin{:});

        case 'sample_from_prob'
            varargout{1} = sample_from_prob_local(varargin{:});

        case 'svd_truncate'
            [varargout{1},varargout{2},varargout{3},varargout{4}] = svd_truncate_local(varargin{:});

        case 'plot_abs_error'
            plot_abs_error_local(varargin{:});
            varargout = {};

        otherwise
            error('Unknown YY_utils action = %s', action);
    end
end

function Ur = Solve_AF_local(A,F,GPU,QR)
    if nargin < 3
        GPU = 0;
        QR = 1;
    end

    if GPU == 1 && gpuDeviceCount > 0
        Ur = gpuArray(A) \ gpuArray(F);
    else
        if QR == 1
            opts.RECT = true;
            Ur = linsolve(A,F,opts);
        else
            Ur = A \ F;
        end
    end
end

function out = get_inner_point_local(a,b,c,d,n)
    eps0 = 1e-6;
    x = linspace(a+eps0,b-eps0,n(1));
    y = linspace(c+eps0,d-eps0,n(2));
    [f1,f2] = meshgrid(x,y);
    out = [f1(:),f2(:)];
end

function [label_mat,num_comp] = label_components_8conn_local(mask)
    mask = logical(mask);
    [nr,nc] = size(mask);

    label_mat = zeros(nr,nc);
    num_comp = 0;

    dirs = [-1 -1; -1 0; -1 1; ...
             0 -1;        0 1; ...
             1 -1;  1 0;  1 1];

    for i = 1:nr
        for j = 1:nc
            if ~mask(i,j) || label_mat(i,j) ~= 0
                continue;
            end

            num_comp = num_comp + 1;

            queue = zeros(nr*nc,2);
            head = 1;
            tail = 1;

            queue(tail,:) = [i,j];
            label_mat(i,j) = num_comp;

            while head <= tail
                cur = queue(head,:);
                head = head + 1;

                ci = cur(1);
                cj = cur(2);

                for d = 1:8
                    ni = ci + dirs(d,1);
                    nj = cj + dirs(d,2);

                    if ni < 1 || ni > nr || nj < 1 || nj > nc
                        continue;
                    end

                    if mask(ni,nj) && label_mat(ni,nj) == 0
                        tail = tail + 1;
                        queue(tail,:) = [ni,nj];
                        label_mat(ni,nj) = num_comp;
                    end
                end
            end
        end
    end
end

function idx = sample_from_prob_local(p,m)
    % Faster inverse-CDF sampling. It preserves the same sampling law as the
    % original loop-with-find implementation, but removes an O(m*N) loop.
    p = gather(p(:));
    p(~isfinite(p)) = 0;
    sp = sum(p);
    if sp <= 0
        p = ones(numel(p),1) / max(numel(p),1);
    else
        p = p / sp;
    end

    cdf = cumsum(p);
    cdf(end) = 1;

    u = rand(m,1);
    % histc is much faster than per-point find and works with older MATLAB versions.
    [~,idx] = histc(u,[0; cdf]);
    idx(idx < 1) = 1;
    idx(idx > numel(cdf)) = numel(cdf);
    idx = idx(:);
end

function [U_r,V_r,s_r,r_keep] = svd_truncate_local(B,orth_cfg)
    [U,S,V] = svd(B,'econ');
    s = diag(S);

    if isempty(s) || max(s) == 0
        U_r = [];
        V_r = [];
        s_r = [];
        r_keep = 0;
        return;
    end

    switch lower(orth_cfg.mode)
        case 'fixed_rank'
            r_keep = min([orth_cfg.fixed_rank, length(s), size(V,2)]);

        case 'energy'
            energy = cumsum(s.^2) / (sum(s.^2) + eps);
            r_keep = find(energy >= orth_cfg.energy_ratio, 1, 'first');

        case 'rel_tol'
            r_keep = sum(s > orth_cfg.rel_tol * max(s));

        otherwise
            error('Unknown orth_cfg.mode.');
    end

    r_keep = max(1,r_keep);

    U_r = U(:,1:r_keep);
    V_r = V(:,1:r_keep);
    s_r = s(1:r_keep);
end

function plot_abs_error_local(xp1,xp2,exact,pred,fig_title)
    pred = gather(pred);
    exact = gather(exact);

    err_matrix = reshape(abs(exact - pred), length(xp2), length(xp1));

    figure;
    contourf(xp1,xp2,err_matrix,40,'LineColor','none');
    shading interp;
    colormap jet;
    colorbar;
    title(fig_title);
    xlabel('x');
    ylabel('y');
end
