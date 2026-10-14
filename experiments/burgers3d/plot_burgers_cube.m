function plot_burgers_cube(DATA_DIR, SAVE_FIG)
%PLOT_BURGERS_CUBE Cube-surface figures for the 3D Burgers (x,y,t) result.
%   Reads all_slices_50x50x50.mat (produced by run_burgers3d) and draws
%   Exact / Prediction / Error space-time cube surfaces at view(45,25),
%   using all six boundary faces of the (x,y,t) cube.
%
%   Usage:
%     plot_burgers_cube()                       % DATA_DIR = pwd
%     plot_burgers_cube('/path/to/dense_folder') % folder with all_slices_50x50x50.mat
%
%   Outputs (PNG + FIG + EPS): Burgers_cube_{exact,prediction,error}_view45.*

    if nargin < 1 || isempty(DATA_DIR), DATA_DIR = pwd; end
    if nargin < 2 || isempty(SAVE_FIG), SAVE_FIG = true; end

    dataFile = fullfile(DATA_DIR, 'all_slices_50x50x50.mat');

    out_exact_png = fullfile(DATA_DIR, 'Burgers_cube_exact_view45.png');
    out_pred_png  = fullfile(DATA_DIR, 'Burgers_cube_prediction_view45.png');
    out_err_png   = fullfile(DATA_DIR, 'Burgers_cube_error_view45.png');
    out_exact_fig = fullfile(DATA_DIR, 'Burgers_cube_exact_view45.fig');
    out_pred_fig  = fullfile(DATA_DIR, 'Burgers_cube_prediction_view45.fig');
    out_err_fig   = fullfile(DATA_DIR, 'Burgers_cube_error_view45.fig');
    out_exact_eps = fullfile(DATA_DIR, 'Burgers_cube_exact_view45.eps');
    out_pred_eps  = fullfile(DATA_DIR, 'Burgers_cube_prediction_view45.eps');
    out_err_eps   = fullfile(DATA_DIR, 'Burgers_cube_error_view45.eps');

    FIG_POS_CUBE = [100, 100, 560, 520];
    FONT_SIZE_AXIS  = 13;
    FONT_SIZE_LABEL = 15;
    FONT_SIZE_TITLE = 12;
    AX_LINE_WIDTH   = 1.2;
    TITLE_SIZE_CUBE = 12;
    az0 = 45; el0 = 25;
    cmap_sol = jet(256);
    cmap_err = jet(256);

    if exist(dataFile,'file') ~= 2
        error('Cannot find file:\n%s', dataFile);
    end
    S = load(dataFile);
    if ~isfield(S,'x_vec') || ~isfield(S,'y_vec') || ~isfield(S,'t_values')
        error('The file must contain x_vec, y_vec, and t_values.');
    end
    x_vec = S.x_vec(:).';
    y_vec = S.y_vec(:).';
    t_values = S.t_values(:).';
    Nx = numel(x_vec); Ny = numel(y_vec); Nt = numel(t_values);

    if isfield(S,'U_exact_yx'), U_exact = S.U_exact_yx; elseif isfield(S,'U_exact'), U_exact = S.U_exact; else, error('no U_exact'); end
    if isfield(S,'U_pred_yx'),  U_pred  = S.U_pred_yx;  elseif isfield(S,'U_pred'),  U_pred  = S.U_pred;  else, error('no U_pred'); end

    U_exact = convert_to_yxt(U_exact, Nx, Ny, Nt, 'U_exact');
    U_pred  = convert_to_yxt(U_pred,  Nx, Ny, Nt, 'U_pred');
    U_error = abs(U_pred - U_exact);

    rel_L2 = norm(U_pred(:)-U_exact(:))/(norm(U_exact(:))+1e-12);
    max_err = max(U_error(:));
    fprintf('Loaded dense Burgers2D data: Nx=%d, Ny=%d, Nt=%d\n', Nx, Ny, Nt);
    fprintf('  Relative L2 error = %.6e, Max abs error = %.6e\n', rel_L2, max_err);

    u_min = min([U_exact(:); U_pred(:)]); u_max = max([U_exact(:); U_pred(:)]);
    e_min = 0; e_max = max(U_error(:));

    % ---- Exact ----
    fig1 = figure('Color','w','Name','Exact','Units','pixels','Position',FIG_POS_CUBE,'PaperPositionMode','auto','InvertHardcopy','off');
    ax1 = axes('Parent',fig1,'Position',[0.05 0.06 0.76 0.86]);
    plot_cube_faces_all(ax1, x_vec, y_vec, t_values, U_exact, [u_min,u_max], cmap_sol, az0, el0, FONT_SIZE_AXIS, FONT_SIZE_LABEL, AX_LINE_WIDTH);
    title(ax1,'Exact','Interpreter','latex','FontSize',TITLE_SIZE_CUBE,'FontWeight','normal');
    cb1 = colorbar(ax1); format_colorbar(cb1,FONT_SIZE_AXIS,AX_LINE_WIDTH); cb1.Position = [0.84 0.16 0.035 0.68];
    if SAVE_FIG, save_one_figure(fig1, out_exact_fig, out_exact_png, out_exact_eps); end

    % ---- Prediction ----
    fig2 = figure('Color','w','Name','Prediction','Units','pixels','Position',FIG_POS_CUBE+[60 30 0 0],'PaperPositionMode','auto','InvertHardcopy','off');
    ax2 = axes('Parent',fig2,'Position',[0.05 0.06 0.76 0.86]);
    plot_cube_faces_all(ax2, x_vec, y_vec, t_values, U_pred, [u_min,u_max], cmap_sol, az0, el0, FONT_SIZE_AXIS, FONT_SIZE_LABEL, AX_LINE_WIDTH);
    title(ax2,'Prediction','Interpreter','latex','FontSize',TITLE_SIZE_CUBE,'FontWeight','normal');
    cb2 = colorbar(ax2); format_colorbar(cb2,FONT_SIZE_AXIS,AX_LINE_WIDTH); cb2.Position = [0.84 0.16 0.035 0.68];
    if SAVE_FIG, save_one_figure(fig2, out_pred_fig, out_pred_png, out_pred_eps); end

    % ---- Error ----
    fig3 = figure('Color','w','Name','Error','Units','pixels','Position',FIG_POS_CUBE+[120 60 0 0],'PaperPositionMode','auto','InvertHardcopy','off');
    ax3 = axes('Parent',fig3,'Position',[0.05 0.06 0.76 0.86]);
    plot_cube_faces_all(ax3, x_vec, y_vec, t_values, U_error, [e_min,e_max], cmap_err, az0, el0, FONT_SIZE_AXIS, FONT_SIZE_LABEL, AX_LINE_WIDTH);
    title(ax3,'Error','Interpreter','latex','FontSize',TITLE_SIZE_CUBE,'FontWeight','normal');
    cb3 = colorbar(ax3); format_colorbar(cb3,FONT_SIZE_AXIS,AX_LINE_WIDTH); cb3.Position = [0.84 0.16 0.035 0.68];
    cb3.Label.String = '$|u_h-u_{\rm ref}|$'; cb3.Label.Interpreter = 'latex'; cb3.Label.FontSize = FONT_SIZE_LABEL;
    if SAVE_FIG, save_one_figure(fig3, out_err_fig, out_err_png, out_err_eps); end

    fprintf('\nFigures saved to:\n  %s\n  %s\n  %s\n', out_exact_png, out_pred_png, out_err_png);
end

%% ============================================================
%  Local functions
%% ============================================================
function U = convert_to_yxt(Uraw, Nx, Ny, Nt, name)
    sz = size(Uraw);
    if numel(sz) ~= 3, error('%s must be a 3D array.', name); end
    if isequal(sz,[Ny,Nx,Nt]),       U = Uraw;
    elseif isequal(sz,[Nx,Ny,Nt]),   U = permute(Uraw,[2,1,3]);
    elseif isequal(sz,[Nt,Ny,Nx]),   U = permute(Uraw,[2,3,1]);
    elseif isequal(sz,[Nt,Nx,Ny]),   U = permute(Uraw,[3,2,1]);
    else, error('%s has unexpected size [%d %d %d]. Expected Ny x Nx x Nt.', name, sz(1), sz(2), sz(3));
    end
end

function plot_cube_faces_all(ax, x_vec, y_vec, t_vec, U, climv, cmapv, az0, el0, FONT_SIZE_AXIS, FONT_SIZE_LABEL, AX_LINE_WIDTH)
    axes(ax); hold(ax,'on');
    % t = t_min
    [X,Y] = meshgrid(x_vec,y_vec); T = t_vec(1)*ones(size(X)); C = U(:,:,1);
    surf(ax,X,Y,T,C,'EdgeColor','none','FaceColor','interp');
    % t = t_max
    T = t_vec(end)*ones(size(X)); C = U(:,:,end);
    surf(ax,X,Y,T,C,'EdgeColor','none','FaceColor','interp');
    % x = x_min
    [Y,T] = ndgrid(y_vec,t_vec); X = x_vec(1)*ones(size(Y)); C = squeeze(U(:,1,:));
    surf(ax,X,Y,T,C,'EdgeColor','none','FaceColor','interp');
    % x = x_max
    X = x_vec(end)*ones(size(Y)); C = squeeze(U(:,end,:));
    surf(ax,X,Y,T,C,'EdgeColor','none','FaceColor','interp');
    % y = y_min
    [X,T] = ndgrid(x_vec,t_vec); Y = y_vec(1)*ones(size(X)); C = squeeze(U(1,:,:));
    surf(ax,X,Y,T,C,'EdgeColor','none','FaceColor','interp');
    % y = y_max
    Y = y_vec(end)*ones(size(X)); C = squeeze(U(end,:,:));
    surf(ax,X,Y,T,C,'EdgeColor','none','FaceColor','interp');
    % cube edges
    draw_cube_edges(ax,x_vec,y_vec,t_vec,AX_LINE_WIDTH);
    view(ax,az0,el0);
    axis(ax,[x_vec(1) x_vec(end) y_vec(1) y_vec(end) t_vec(1) t_vec(end)]);
    daspect(ax,[1 1 1]); pbaspect(ax,[1 1 1]); axis(ax,'vis3d');
    box(ax,'on'); grid(ax,'off');
    colormap(ax,cmapv); caxis(ax,climv);
    xlabel(ax,'$x$','Interpreter','latex','FontSize',FONT_SIZE_LABEL);
    ylabel(ax,'$y$','Interpreter','latex','FontSize',FONT_SIZE_LABEL);
    zlabel(ax,'$t$','Interpreter','latex','FontSize',FONT_SIZE_LABEL);
    set(ax,'FontSize',FONT_SIZE_AXIS,'LineWidth',AX_LINE_WIDTH,'TickDir','in','TickLabelInterpreter','latex','Color','w');
    set(ax,'XTick',[0 0.5 1]); set(ax,'YTick',[0 0.5 1]); set(ax,'ZTick',[0 0.5 1]);
    set(ax,'LooseInset',max(get(ax,'TightInset'),0.01));
    shading(ax,'interp'); hold(ax,'off');
end

function draw_cube_edges(ax, x_vec, y_vec, t_vec, AX_LINE_WIDTH)
    x0=x_vec(1); x1=x_vec(end); y0=y_vec(1); y1=y_vec(end); z0=t_vec(1); z1=t_vec(end);
    P = [x0 y0 z0; x1 y0 z0; x1 y1 z0; x0 y1 z0; x0 y0 z1; x1 y0 z1; x1 y1 z1; x0 y1 z1];
    E = [1 2;2 3;3 4;4 1;5 6;6 7;7 8;8 5;1 5;2 6;3 7;4 8];
    for k = 1:size(E,1)
        p1=P(E(k,1),:); p2=P(E(k,2),:);
        plot3(ax,[p1(1) p2(1)],[p1(2) p2(2)],[p1(3) p2(3)],'k-','LineWidth',0.75*AX_LINE_WIDTH);
    end
end

function format_colorbar(cb, FONT_SIZE_AXIS, AX_LINE_WIDTH)
    cb.FontSize = FONT_SIZE_AXIS;
    cb.LineWidth = AX_LINE_WIDTH;
    cb.TickLabelInterpreter = 'latex';
end

function save_one_figure(fig, figFile, pngFile, epsFile)
    set(fig,'PaperPositionMode','auto'); set(fig,'InvertHardcopy','off');
    savefig(fig,figFile);
    exportgraphics(fig,pngFile,'Resolution',300);
    set(fig,'Renderer','opengl');
    print(fig,epsFile,'-depsc','-opengl','-r600');
    fprintf('Saved:\n  %s\n  %s\n  %s\n',figFile,pngFile,epsFile);
end
