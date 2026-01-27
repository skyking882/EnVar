clc; clear; close all;

% 1. Setup Phase
fly = InsectModel(); 
opt = EnVarOptimizer();

% Configure Optimizer
opt.Nen     = 10;
opt.lambda  = 0.5;
opt.r_lift  = 2000;
opt.maxIter = 500;

% 2. Initial Guess
u_phys_start = [150; 15*pi/180; 55*pi/180; 45*pi/180; 0; 80*pi/180; 0.5; 3; 0; 90*pi/180];
u_norm_start = fly.phys2norm(u_phys_start);

% --- 3. Visualization Setup (Pre-loop) ---
hFig = figure('Name', 'Real-time EnVar Opt', 'Color', 'w', 'Position', [100, 100, 1200, 600]);

subplot(2,2,1); h_J = plot(0,0,'k-o'); title('Total Cost (J)'); grid on; hold on;
subplot(2,2,2); h_L = plot(0,0,'b-o'); yline(1.0, 'r--'); title('Lift (Target=1.0)'); grid on; hold on;
subplot(2,2,3); h_P = plot(0,0,'r-o'); title('Aerodynamic Power (P*)'); grid on; hold on;
subplot(2,2,4); h_D = semilogy(0,0,'m-'); title('TR Radius (\Delta)'); grid on; hold on;

opt.OutputFcn = @(hist, k) update_plots(hist, k, h_J, h_L, h_P, h_D);

% 4. Run Optimization
fprintf('Starting Optimization for Fruit Fly...\n');
t_start = tic;

[u_best_norm, history] = opt.run(fly, u_norm_start);

t_end = toc(t_start);
fprintf('Optimization completed in %.2f seconds.\n', t_end);

% 5. Post-Processing
fprintf('\n=== Optimal Parameters ===\n');
fly.print_parameters(u_best_norm);

%% --- Helper Function for Visualization ---
function update_plots(hist, k, h_J, h_L, h_P, h_D)

    
    x_data = 1:k;
    
    set(h_J, 'XData', x_data, 'YData', hist.J);
    set(h_L, 'XData', x_data, 'YData', hist.L);
    set(h_P, 'XData', x_data, 'YData', hist.P);
    set(h_D, 'XData', x_data, 'YData', hist.Delta);
    
    drawnow limitrate; 
end