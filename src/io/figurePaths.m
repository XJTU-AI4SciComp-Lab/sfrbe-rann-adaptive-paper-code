function [pdfPath, pngPath] = figurePaths(cfg, tag)
%FIGUREPATHS generate unified paper figure paths: figures/<problem>/<tag>.pdf and .png
%   tag is of the form 'fig06_poisson2peak_exact_sol'.
    problem = cfg.problem;
    figDir = fullfile(cfg.output.figures_root, problem);
    if exist(figDir, 'dir') ~= 7
        mkdir(figDir);
    end
    pdfPath = fullfile(figDir, [tag, '.pdf']);
    pngPath = fullfile(figDir, [tag, '.png']);
end