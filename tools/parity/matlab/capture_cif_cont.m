function capture_cif_cont()
%CAPTURE_CIF_CONT  Gold-fixture capture for PointProcessSimulationCont.slx.
%
% Produces the deterministic conditional-intensity (lambda) gold traces used
% to validate the native-Python continuous-time CIF simulator
% (nstat.extras.simulate_cif_continuous).
%
% The continuous Simulink model computes, on a variable-step ode45 solver:
%     eta(t)      = mu + S*stim(t) + E*ens(t) + H*pp_delayed(t)
%     lambdaDelta = exp(eta)               (Poisson,  simTypeSelect = 1)
%                 = exp(eta)/(1+exp(eta))  (Binomial, simTypeSelect = 0)
%     spike(t)    = 1  iff  U(0,1) < lambdaDelta
%     "lambdaDelta" output port = lambdaDelta / Ts  = lambda (rate, Hz)
%
% With H = 0 (no self-history feedback) the lambda trace is deterministic in
% the inputs and therefore reproducible bit-close in Python; that is the
% strong validation fixture captured here.  History-feedback behaviour is
% validated Python-side with injected uniforms (MATLAB's DSP Random Source
% RNG is not reproducible in NumPy).
%
% Reproduces: tests/parity/fixtures/matlab_gold/cif_cont_lambda.mat
%
% MATLAB nSTAT repo (which holds the .slx) is resolved from NSTAT_MATLAB_PATH,
% defaulting to /Users/iahncajigas/projects/nstat.

matlabRepo = getenv('NSTAT_MATLAB_PATH');
if isempty(matlabRepo); matlabRepo = '/Users/iahncajigas/projects/nstat'; end
here = fileparts(mfilename('fullpath'));
outMat = fullfile(here, '..', '..', '..', 'tests', 'parity', 'fixtures', ...
                  'matlab_gold', 'cif_cont_lambda.mat');

cd(matlabRepo);
warning('off','all');
rng(42);
mdl = 'PointProcessSimulationCont';

% ---- deterministic (H = 0) configuration ----
Ts   = 0.001;
tMax = 1.0;
t    = (0:Ts:tMax)';
mu   = -1;
Snum = 1; Sden = [0.05 1];           % continuous 1st-order lowpass, tau = 50 ms
S    = tf(Snum, Sden);
H    = tf(0,1);  E = tf(0,1);        % no history / ensemble -> deterministic lambda
seed = 42; TsInt = Ts; dsp_sampFrame = 1;
u    = 2*sin(2*pi*2*t);              % stimulus
e    = zeros(size(t));               % ensemble

stim_ext.time = t; stim_ext.signals.values = u; stim_ext.signals.dimensions = 1;
ens_ext.time  = t; ens_ext.signals.values  = e; ens_ext.signals.dimensions  = 1;

base = {'Ts','mu','S','H','E','seed','TsInt','dsp_sampFrame','t','u','e','stim_ext','ens_ext'};
for i = 1:numel(base); assignin('base', base{i}, eval(base{i})); end

load_system(mdl);
cs = getActiveConfigSet(mdl);
set_param(cs, 'OutputOption','SpecifiedOutputTimes', 'OutputTimes','t', ...
             'SaveFormat','Array', 'SaveOutput','on', 'SaveTime','on');

% analytic linear predictor (independent check of the S filter in Python)
eta = mu + lsim(S, u, t);

lambda_poisson = run_case(mdl, tMax, 1);   % simTypeSelect = 1
lambda_binom   = run_case(mdl, tMax, 0);   % simTypeSelect = 0

% lambda_* are the "lambdaDelta" output port = link(eta)/Ts
save(outMat, 't','u','e','mu','Ts','Snum','Sden','eta', ...
     'lambda_poisson','lambda_binom','-v7');
fprintf('Saved gold fixture: %s\n', outMat);
fprintf('  poisson lambda range [%.4g .. %.4g]\n', min(lambda_poisson), max(lambda_poisson));
fprintf('  binomial lambda range [%.4g .. %.4g]\n', min(lambda_binom), max(lambda_binom));
end

function lam = run_case(mdl, tMax, sel)
    assignin('base','simTypeSelect', sel);
    so = sim(mdl, 'StopTime', num2str(tMax), ...
             'LoadExternalInput','on', 'ExternalInput','stim_ext, ens_ext', ...
             'ReturnWorkspaceOutputs','on');
    yout = so.yout;               % [N x 2] = [pp, lambdaDelta-port]
    lam  = yout(:, 2);            % lambda (rate) trace
end
