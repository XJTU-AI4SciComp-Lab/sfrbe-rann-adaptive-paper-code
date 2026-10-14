function plot_burgers_bias(runDir, SAVE_FIG)
%PLOT_BURGERS_BIAS 3D SAB structure/background bias centers for Burgers2D.
%   Reads sab_bias.mat (produced by run_burgers3d) and draws the SAB adaptive
%   bias centers in the (x,y,t) cube at view(45,25):
%     gray  = background centers
%     red   = structure-region centers
%   Outputs burgers_bias.png / .fig / .eps in the same run directory.
%
%   Usage:
%     plot_burgers_bias()                          % look in pwd
%     plot_burgers_bias('results/burgers3d')       % run dir with sab_bias.mat

    if nargin < 1 || isempty(runDir), runDir = pwd; end
    if nargin < 2 || isempty(SAVE_FIG), SAVE_FIG = true; end

    fb = fullfile(runDir,'sab_bias.mat');
    if exist(fb,'file') ~= 2
        error('Cannot find sab_bias.mat in %s.\nRe-run run_burgers3d with the current code (it saves sab_bias.mat).', runDir);
    end
    S = load(fb);
    sb = S.sab_bias;
    if isfield(sb,'centersS') && ~isempty(sb.centersS)
        C_S = sb.centersS;
    else
        C_S = sb.centers(:,1:sb.mS_total);
    end
    if isfield(sb,'centersB') && ~isempty(sb.centersB)
        C_B = sb.centersB;
    else
        C_B = sb.centers(:,sb.mS_total+1:end);
    end

    FIG_POS = [100, 100, 560, 520];
    FONT_SIZE_AXIS = 13;
    FONT_SIZE_LABEL = 15;
    FONT_SIZE_TITLE = 16;
    AX_LINE_WIDTH = 1.2;
    MARKER_BACKGROUND = 8;
    MARKER_STRUCTURE = 12;

    fig = figure('Color','w','Name','SAB bias centers','Units','pixels', ...
        'Position',FIG_POS,'PaperPositionMode','auto','InvertHardcopy','off');
    hold on;
    scatter3(C_B(1,:), C_B(2,:), C_B(3,:), MARKER_BACKGROUND, [0.65 0.65 0.65], 'filled');
    scatter3(C_S(1,:), C_S(2,:), C_S(3,:), MARKER_STRUCTURE, 'r', 'filled');
    draw_bias_cube_edges([0 1 0 1 0 1], AX_LINE_WIDTH);
    view(45,25);
    axis([0 1 0 1 0 1]); daspect([1 1 1]); pbaspect([1 1 1]); axis vis3d;
    box on; grid off;
    xlabel('$x$','Interpreter','latex','FontSize',FONT_SIZE_LABEL);
    ylabel('$y$','Interpreter','latex','FontSize',FONT_SIZE_LABEL);
    zlabel('$t$','Interpreter','latex','FontSize',FONT_SIZE_LABEL);
    title('SFR adaptive bias centers','FontSize',FONT_SIZE_TITLE,'FontWeight','normal');
    legend('background bias centers','structure bias centers','Location','best');
    set(gca,'FontSize',FONT_SIZE_AXIS,'LineWidth',AX_LINE_WIDTH,'TickDir','in', ...
        'TickLabelInterpreter','latex','Color','w');
    set(gca,'XTick',[0 0.5 1]); set(gca,'YTick',[0 0.5 1]); set(gca,'ZTick',[0 0.5 1]);
    set(gca,'LooseInset',max(get(gca,'TightInset'),0.01));
    hold off;

    if SAVE_FIG
        out_fig = fullfile(runDir,'burgers_bias.fig');
        out_png = fullfile(runDir,'burgers_bias.png');
        out_eps = fullfile(runDir,'burgers_bias.eps');
        set(fig,'PaperPositionMode','auto'); set(fig,'InvertHardcopy','off');
        savefig(fig, out_fig);
        exportgraphics(fig, out_png, 'Resolution', 300);
        set(fig,'Renderer','opengl');
        print(fig, out_eps, '-depsc', '-opengl', '-r600');
        fprintf('Saved:\n  %s\n  %s\n  %s\n', out_fig, out_png, out_eps);
    end
end

function draw_bias_cube_edges(dom, LW)
    x0=dom(1); x1=dom(2); y0=dom(3); y1=dom(4); z0=dom(5); z1=dom(6);
    P = [x0 y0 z0; x1 y0 z0; x1 y1 z0; x0 y1 z0; x0 y0 z1; x1 y0 z1; x1 y1 z1; x0 y1 z1];
    E = [1 2;2 3;3 4;4 1;5 6;6 7;7 8;8 5;1 5;2 6;3 7;4 8];
    for k = 1:size(E,1)
        p1=P(E(k,1),:); p2=P(E(k,2),:);
        plot3([p1(1) p2(1)],[p1(2) p2(2)],[p1(3) p2(3)],'k-','LineWidth',0.75*LW);
    end
end
