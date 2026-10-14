%RUN_BURGERS3D_SISC Unambiguous launcher for this SFRBE-SISC checkout.
%
% Use:
%   run('D:\Matlab项目\SFRBE-SISC\run_burgers3d_sisc.m')
%
% The repository also contains SFRBE-run and SFRBE-SISC - 副本, both of
% which define run_burgers3d.  MATLAB gives the current directory special
% precedence, so merely adding SFRBE-SISC to path is not enough when the
% current directory is one of those sibling experiments.

sisc_project_root = fileparts(mfilename('fullpath'));
sisc_experiment_dir = fullfile(sisc_project_root,'experiments','burgers3d');
sisc_source_dir = fullfile(sisc_project_root,'src');

if ~isfolder(sisc_experiment_dir) || ~isfolder(sisc_source_dir)
    error('SFRBE-SISC launcher cannot find its experiment or source directory.');
end

sisc_previous_dir = pwd;
sisc_restore_dir = onCleanup(@() cd(sisc_previous_dir)); %#ok<NASGU>
cd(sisc_experiment_dir);
addpath(sisc_experiment_dir,'-begin');
addpath(genpath(sisc_source_dir),'-begin');
rehash path;
clear run_burgers3d burgers3d_config Burgers2D_Newton_module3D

fprintf('SFRBE-SISC launcher selected:\n');
fprintf('  driver: %s\n',which('run_burgers3d'));
fprintf('  config: %s\n',which('burgers3d_config'));
fprintf('  solver: %s\n',which('Burgers2D_Newton_module3D'));

run_burgers3d();
