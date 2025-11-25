% ----------------------------------------------------------------------- %
% ----------------------- EnVar Optimization Test ----------------------- %
% ------------------- Nonlinear Double-Well Observer -------------------- %
% ----------------------------------------------------------------------- %
clc;
clear; close all;

%% 1. Problem setup
Nen    = 3;        % Number of ensemble members
N_Dofs = 2;        % Number of degrees of freedom
lambda = 0;      % Regularisation strength (0 = off)

% Reference / target in observation space:
% Nonlinear observer: z(a) = (a1^2 - 1)^2 + a2^2
z_target = 0.0;

% Initial guess (choose off-centre to see which well you fall into)
a_e = [2.0; 1.5; 0.0];    % mean control (3 x 1)

% Storage
history_error = [];
history_a     = a_e;      % mean trajectory (3 x n_it)

fprintf('Nonlinear observer: z(a) = (a1^2 - 1)^2 + a2^2\n');
fprintf('True cost: J_true(a) = 0.5*(z(a) - z_target)^2 + 0.5*lambda*||L a||^2\n');
fprintf('Start: a = [%.2f %.2f %.2f]^T\n\n', a_e(1), a_e(2), a_e(3));

%% 2. Prepare visualisation: J_true(a1,a2) with a3 = 0
figure('Position', [50, 100, 800, 600]);

a1_range = -2.0:0.02:2.0;
a2_range = -2.0:0.02:2.0;
[A1, A2] = meshgrid(a1_range, a2_range);

Zobs = (A1.^2 - 1).^2 + A2.^2;          % z(a1,a2) for a3 = 0
Jtrue = 0.5 * (Zobs - z_target).^2;     % lambda-term ignored in the plot

contourf(A1, A2, Jtrue, 40, 'LineColor', 'none'); hold on;
colorbar;
title('EnVar optimisation on J_{true}(a_1,a_2) = 0.5\{[(a_1^2-1)^2 + a_2^2] - z_t\}^2');
xlabel('a_1'); ylabel('a_2');
axis equal tight; grid on;

% Initial mean and placeholders for ensemble + trajectory
h_mean = plot(a_e(1), a_e(2), 'ro', 'MarkerFaceColor', 'r', ...
              'MarkerSize', 8, 'DisplayName', 'Mean');
h_ens  = plot(NaN, NaN, 'b.', 'MarkerSize', 10, ...
              'DisplayName', 'Ensemble');
h_traj = plot(a_e(1), a_e(2), 'k-', 'LineWidth', 1.5, ...
              'DisplayName', 'Trajectory');
legend([h_mean, h_ens, h_traj], 'Location', 'best');

%% 3. EnVar optimisation loop
maxIter = 50;
alpha   = 0.2;      % relaxation factor, <1.0 to slow down steps

for iter = 1:maxIter

    % --- A. Generate ensemble around current mean -----------------------
    a_r_ensemble = Ensemble(Nen, a_e);      % size: N_Dofs x Nen
    E_prime      = a_r_ensemble - a_e;      % deviation matrix E'

    % --- B. Nonlinear observer for this iteration -----------------------
    % Observer: z(a) = (a1^2 - 1)^2 + a2^2, scalar
    % Observation mismatch: o(a) = z(a) - z_target

    % Mean observation
    a1_e = a_e(1);
    a2_e = a_e(2);
    z_e  = (a1_e^2 - 1)^2 + a2_e^2;
    o_e  = z_e - z_target;            % scalar: 1 x 1

    % Ensemble observations
    o_r_mat = zeros(1, Nen);          % 1 x Nen
    for k = 1:Nen
        a1_k = a_r_ensemble(1, k);
        a2_k = a_r_ensemble(2, k);
        z_k  = (a1_k^2 - 1)^2 + a2_k^2;
        o_r_mat(1, k) = z_k - z_target;
    end

    % Observation perturbation matrix H (1 x Nen)
    H = o_r_mat - o_e;

    % --- C. Local EnVar cost in weight space, solve for w ----------------
    J_fun = @(w) Cost(o_e, H, w, a_e, E_prime, lambda);

    w0   = zeros(Nen, 1);
    LB   = -Inf(Nen, 1);
    UB   =  Inf(Nen, 1);
    opts = optimset('Display', 'off', 'Algorithm', 'sqp');

    [w_opt, J_loc] = fmincon(J_fun, w0, [], [], [], [], LB, UB, [], opts);

    % --- D. Update mean in control space --------------------------------
    step_vec  = E_prime * w_opt;         % N_Dofs x 1
    a_e       = a_e + alpha * step_vec;  % relaxed update

    % Store trajectory
    history_a(:, end+1) = a_e;

    % Compute true cost for monitoring (using same J_true definition)
    a1     = a_e(1); 
    a2     = a_e(2);
    z_val  = (a1^2 - 1)^2 + a2^2;
    J_true = 0.5 * (z_val - z_target)^2;
    % simple L a regularisation if desired
    if lambda > 0
        L = diag(1:3);
        J_true = J_true + 0.5 * lambda * norm(L * a_e)^2;
    end

    current_error = abs(z_val - z_target);  % error in observable
    history_error = [history_error; current_error];

    % --- E. Update plots -------------------------------------------------
    set(h_ens,  'XData', a_r_ensemble(1,:), 'YData', a_r_ensemble(2,:));
    set(h_mean, 'XData', a_e(1),            'YData', a_e(2));
    set(h_traj, 'XData', history_a(1,:),    'YData', history_a(2,:));
    drawnow;

    % --- F. Logging ------------------------------------------------------
    fprintf(['Iter %2d | J_loc(w) = %.3e | J_true(a) = %.3e | ' ...
             '|z - z_t| = %.3e | a = [%.3f %.3f %.3f]^T\n'], ...
             iter, J_loc, J_true, current_error, a_e(1), a_e(2), a_e(3));

    if current_error < 1e-6
        fprintf('Converged: |z(a) - z_target| < 1e-6.\n');
        break;
    end
end

%% 4. Error convergence
figure;
plot(history_error, '-o', 'LineWidth', 1.5);
xlabel('Iteration');
ylabel('|z(a) - z_{target}|');
title('EnVar convergence of nonlinear observable');
grid on;

%% 5. Final trajectory on J_true(a1,a2)
figure('Position', [900, 100, 800, 600]);
contourf(A1, A2, Jtrue, 40, 'LineColor', 'none'); hold on; grid on;
axis equal tight;
colorbar;
title('Mean trajectory on J_{true}(a_1,a_2)');
xlabel('a_1'); ylabel('a_2');

plot(history_a(1,:), history_a(2,:), 'k-o', 'LineWidth', 1.5, ...
     'MarkerFaceColor', 'k', 'DisplayName', 'Mean trajectory');
plot(history_a(1,1),  history_a(2,1),  'gs', 'MarkerSize', 10, ...
     'MarkerFaceColor', 'g', 'DisplayName', 'Start');
plot(history_a(1,end),history_a(2,end),'ro', 'MarkerSize', 10, ...
     'MarkerFaceColor', 'r', 'DisplayName', 'End');
legend('show', 'Location', 'best');