%% RL–MPC Demonstration under Model Mismatch
clc; clear; close all;

% ---------------------------------------------------------
% 1. System Parameters
% ---------------------------------------------------------
P_rail_nominal = 1000;   % Bar (Model belief inside MPC)
P_rail_faulty  = 800;    % Bar (Actual Plant Pressure)
alpha = sqrt(P_rail_faulty / P_rail_nominal);  % Mismatch Factor ≈ 0.894

% ---------------------------------------------------------
% 2. Simulation Setup
% ---------------------------------------------------------
N = 2000;               % Number of cycles (simulation steps)
IMEP_target = generate_reference_signal(N);  % Ref trajectory
IMEP_actual = zeros(N,1);
RL_action   = 0;        % Initial offset correction

% ---------------------------------------------------------
% 3. Main Simulation Loop
% ---------------------------------------------------------
for k = 2:N
    % ---- (1) RL correction (outer loop) ----
    error = IMEP_target(k-1) - IMEP_actual(k-1);      % tracking error
    RL_action = RL_action + 0.1 * error;              % integral-like update
    
    % ---- (2) MPC (inner loop) ----
    modified_ref = IMEP_target(k) + RL_action;        % virtual boosted ref
    u_DOI_command = inverse_plant_model_simple(modified_ref, 1.0); % MPC assumes α=1
    
    % ---- (3) Real Plant (α < 1) ----
    IMEP_actual(k) = plant_dynamics_simple(u_DOI_command, alpha);
end

% ---------------------------------------------------------
% 4. Plot Results
% ---------------------------------------------------------
figure('Color','w');
plot(IMEP_target,'r','LineWidth',2); hold on;
plot(IMEP_actual,'b','LineWidth',1.5);
title('Performance of RL–Augmented MPC under Mismatch');
xlabel('Cycle'); ylabel('IMEP / bar');
legend('Target Reference','RL+MPC Output');
grid on;

% ---------------------------------------------------------
% --- Helper Functions (inline below for standalone run) ---
% ---------------------------------------------------------
function ref = generate_reference_signal(N)
    % Tri‑step reference with ramps – similar to your report pattern
    ref = 4*ones(1,N);
    ref(1:400)   = linspace(4,6,400);
    ref(401:800) = linspace(6,8,400);
    ref(801:1200)= linspace(8,5,400);
    ref(1201:end)= linspace(5,7.5,N-1200);
    ref = ref(:);
end

function u = inverse_plant_model_simple(imep_desired, alpha_assumed)
    % Simple linearized inverse: DOI ≈ IMEP / (10*alpha)
    u = imep_desired ./ (10*alpha_assumed);
end

function imep = plant_dynamics_simple(u_DOI, alpha_real)
    % Linear plant + small first‑order smoothing and noise
    persistent last_imep;
    if isempty(last_imep), last_imep = 4; end
    imep = alpha_real * 10 * u_DOI + 0.2*(last_imep - 4) + 0.05*randn;
    last_imep = imep;
end
