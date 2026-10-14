function startup()
%STARTUP one-click add of SFRBE project paths (src only; problems are added by run_experiment as needed).
    root = fileparts(mfilename('fullpath'));
    addpath(genpath(fullfile(root, 'src')));
    fprintf('[SFRBE] src paths added: %s\n', fullfile(root, 'src'));
end