classdef EnVarOptimizer < handle
    % ENVAROPTIMIZER Ensemble-based Variational Optimization Algorithm
    %   Implements a Trust-Region EnVar method.
    %   Separates the algorithmic logic from the physical model.
    
    properties
        % --- Configuration ---
        Nen         = 10;       % Ensemble size
        lambda      = 0;        % Regularization parameter
        maxIter     = 1000;     % Maximum iterations
        r_lift      = 2000;     % Penalty weight for lift constraint
        
        % --- Trust Region Parameters ---
        Delta       = 0.01;     % Initial TR radius
        Delta_min   = 1e-4;
        Delta_max   = 5e-2;
        eta1        = 0.25;     % Reject threshold
        eta2        = 0.75;     % Expansion threshold
        gamma_dec   = 0.5;      % Shrink factor
        gamma_inc   = 1.5;      % Expansion factor
        
        % --- Solver Options ---
        fmin_opts               % Options for fmincon
        % --- Visulization ---
        OutputFcn   = [];
    end
    
    methods
        function obj = EnVarOptimizer()
            % CONSTRUCTOR: Initialize default settings
            obj.fmin_opts = optimoptions('fmincon', ...
                'Display', 'none', ...
                'Algorithm', 'sqp', ...
                'SpecifyObjectiveGradient', true); % We provide gradients in Cost_paper
        end
        
        function [u_opt, history] = run(obj, model, u_init)
            % RUN Execute the optimization loop.
            %
            % Inputs:
            %   model  : Instance of InsectModel
            %   u_init : Initial normalized guess [-1, 1]
            %
            % Outputs:
            %   u_opt   : Optimized normalized parameter
            %   history : Struct containing iteration logs
            
            fprintf('--- Starting EnVar Optimization ---\n');
            
            % 1. Initialization
            u_current = u_init;
            iter      = 0;
            converged = false;
            
            % Initialize History Log
            history.J = []; 
            history.L = []; 
            history.P = []; 
            history.Delta = [];
            history.rho = [];
            
            % Initial Evaluation
            [J_center, L_center, P_center] = model.evaluate(u_current, obj.r_lift);
            
            % 2. Main Loop
            while iter < obj.maxIter && ~converged
                iter = iter + 1;
                
                % --- A. Generate Ensemble (Perturbations) ---
                % Generate matrix E' where columns are perturbations
                [u_ensemble, E_prime] = obj.generate_ensemble(u_current, model);
                
                % --- B. Physics Evaluation (Parallelizable) ---
                % If using high-order CFD, change 'for' to 'parfor' here.
                J_samples = zeros(obj.Nen, 1);
                L_samples = zeros(obj.Nen, 1);
                
                for k = 1:obj.Nen
                    u_k = u_ensemble(:, k);
                    [J_k, L_k, ~] = model.evaluate(u_k, obj.r_lift);
                    J_samples(k) = J_k;
                    L_samples(k) = L_k;
                end
                
                % --- C. Build Linear Surrogate Models ---
                % Gradient approximations based on ensemble spread
                H_cost = (J_samples - J_center)'; % Sensitivity of Cost
                H_lift = (L_samples - L_center)'; % Sensitivity of Lift
                
                % --- D. Solve Trust-Region Subproblem ---
                % Find weights 'w' that minimize modeled cost subject to constraints
                [w_opt, pred_red] = obj.solve_subproblem(...
                    u_current, E_prime, J_center, L_center, ...
                    H_cost, H_lift, model, L_samples);
                
                % --- E. Compute Trial Step ---
                step_norm = E_prime * w_opt;
                step_size = norm(step_norm, Inf);
                
                % Helper: Predict Lift for logging
                L_pred = L_center + H_lift * w_opt;
                
                % Evaluate Trial Point
                u_trial = model.project_feasible(u_current + step_norm);
                [J_trial, L_trial, P_trial] = model.evaluate(u_trial, obj.r_lift);
                
                % --- F. Trust Region Update ---
                act_red = J_center - J_trial; % Actual reduction
                
                % Calculate ratio rho
                if pred_red <= 1e-12 || ~isfinite(pred_red)
                    rho = -Inf;
                else
                    rho = act_red / pred_red;
                end
                
                % Accept/Reject Logic
                accepted = (rho > obj.eta1) && (act_red > 0);
                
                if accepted
                    u_current = u_trial;
                    J_center  = J_trial;
                    L_center  = L_trial;
                    P_center  = P_trial;
                    
                    % Update Radius: Expand if model is good
                    if rho > obj.eta2 && (step_size > 0.8 * obj.Delta)
                        obj.Delta = min(obj.Delta_max, obj.gamma_inc * obj.Delta);
                    end
                else
                    % Reject: Shrink Radius
                    obj.Delta = max(obj.Delta_min, obj.gamma_dec * obj.Delta);
                end
                
               % --- G. Logging ---
                history.J(end+1)     = J_center;
                history.L(end+1)     = L_center;
                history.P(end+1)     = P_center;
                history.Delta(end+1) = obj.Delta;
                history.rho(end+1)   = rho;
                
                % --- H. Real-time Callback  ---
                if ~isempty(obj.OutputFcn)
                    obj.OutputFcn(history, iter); 
                end
                % Console Output
                fprintf('It %3d | J: %.4f | L: %.3f | P*: %.2f | TR: %.1e | rho: %.2f | %s\n', ...
                    iter, J_center, L_center, P_center, obj.Delta, rho, ...
                    string(matlab.lang.OnOffSwitchState(accepted)));
                
                % Convergence Check
                if step_size < 1e-6 && abs(1 - L_center) < 1e-2
                    fprintf('>>> Converged.\n');
                    converged = true;
                end
            end
            
            u_opt = u_current;
        end
    end
    
    methods (Access = private)
        
        function [u_matrix, E_prime] = generate_ensemble(obj, u_center, ~)
            % GENERATE_ENSEMBLE Create perturbations in normalized space.
            % Using a simple orthogonal sampling or random sampling strategy.
            
            N_params = length(u_center);
            
            % Generate random perturbations scaled by Delta
            % (In rigorous EnVar, this might come from a covariance matrix)
            pert = (2*rand(N_params, obj.Nen) - 1); 
            
            % Normalize columns to have length <= Delta
            for k = 1:obj.Nen
                nrm = norm(pert(:,k));
                if nrm > 0
                    pert(:,k) = pert(:,k) / nrm * obj.Delta;
                end
            end
            
            E_prime = pert; 
            u_matrix = u_center + E_prime;
        end
        
        function [w_opt, pred_red] = solve_subproblem(obj, u_e, E_prime, ...
                                    J_cent, L_cent, H_J, H_L, model, L_samples)
            % SOLVE_SUBPROBLEM Setup and solve fmincon for weights w.
            
            % 1. Define Objective Function (Anonymous)
            % J(w) = 0.5 || o_e + H w ||^2 + Regularization
            % Note: Target o_e is 0 (minimize cost), so residual r = J_cent + H*w
            
            o_e = J_cent; % We want J to go to 0, but J is scalar here.
                          % Actually, we want to minimize J_pred = J_cent + H*w
            
            % Cost_paper expects: (o_e, H, w, u_e, E_prime, lambda, W2)
            % Here we adapt strictly to the paper's specific cost structure
            N_params = length(u_e);
            W2 = ones(N_params, 1); % Identity weighting for regularization
            
            fun = @(w) obj.cost_function_wrapper(o_e, H_J, w, u_e, E_prime, obj.lambda, W2);
            
            % 2. Build Constraints
            % This extracts the specific geometric constraints (Table 1)
            % via the helper method.
            [A, b, lb, ub] = obj.build_constraints(u_e, E_prime, model, ...
                                                   L_cent, L_samples, obj.Delta);
            
            w0 = zeros(obj.Nen, 1); % Warm start at zero
            
            % 3. Run Optimization
            try
                [w_opt, ~, ~, ~] = fmincon(fun, w0, A, b, [], [], lb, ub, [], obj.fmin_opts);
            catch
                % Fallback if fmincon fails
                w_opt = zeros(obj.Nen, 1);
            end
            
            % 4. Calculate Predicted Reduction
            % Linear prediction of J reduction
            J_pred = J_cent + H_J * w_opt;
            pred_red = J_cent - J_pred;
        end
        
        function [A, b, lb, ub] = build_constraints(obj, u_e, E_prime, model, ...
                                                    ~, ~, Delta_curr)
            % BUILD_CONSTRAINTS Construct Linear Constraints for 'w'.
            % This maps physical constraints into w-space: A*w <= b
            
            % --- 1. Trust Region Box (log-magnitude bounds logic) ---
            % aaa = max_i |E_prime(i,j)|
            aaa = max(abs(E_prime), [], 1)';
            eps0 = 1e-12;
            bbb = 10.^floor(log10(aaa + eps0));
            ub_log10 = (obj.Nen ./ bbb) / (obj.Nen^2);
            
            % Hard clamp to avoid huge weights
            w_limit = 5.0; 
            ub = min(ub_log10, w_limit);
            lb = -ub;
            
            % --- 2. Physical Constraints (Linearized) ---
            % We need to check: u_phys = u_mid + u_half * (u_e + E*w)
            % Constraint: C * u_phys <= K
            
            % Get scaling factors from model
            mid  = model.u_mid;
            half = model.u_half;
            
            % Pre-calculate coefficient matrix in w-space
            % For a constraint a*u_i + b*u_j <= k:
            % a*(mid_i + half_i*(u_e_i + E_i*w)) + ... <= k
            % (a*half_i*E_i + ...) * w <= k - (a*mid_i + ...) - ...
            
            % A. Box Bounds [-1, 1] on u_norm (Linear in w)
            % -1 <= u_e + E*w <= 1
            idx_no_phase = 1:8; % Skip phases (9,10) as they wrap
            E_box = E_prime(idx_no_phase, :);
            u_box = u_e(idx_no_phase);
            
            A_box = [ E_box; -E_box ];
            b_box = [ ones(8,1) - u_box;  % 1 - u
                      u_box - (-ones(8,1)) ]; % u - (-1)
            b_box = max(b_box, 0); % Safety
            
            % B. The Coupled Wedge Constraints (Table 1)
            % Specific to insect morphology (Indices 3,4,5,6)
            % theta_m(3), eta_m(4), theta_0(5), eta_0(6)
            E = E_prime;
            
            % Current physical center
            u_phys_e = mid + half .* u_e;
            
            % Matrix rows for:
            % 1. theta_m + theta_0 <= pi/2
            % 2. theta_m - theta_0 <= pi/2
            % 3. eta_m + eta_0 <= pi
            % 4. eta_m - eta_0 <= pi
            
            A_cpl = [
                half(3)*E(3,:) + half(5)*E(5,:);
                half(3)*E(3,:) - half(5)*E(5,:);
                half(4)*E(4,:) + half(6)*E(6,:);
                half(4)*E(4,:) - half(6)*E(6,:);
            ];
            
            b_cpl = [
                (pi/2) - (u_phys_e(3) + u_phys_e(5));
                (pi/2) - (u_phys_e(3) - u_phys_e(5));
                (pi)   - (u_phys_e(4) + u_phys_e(6));
                (pi)   - (u_phys_e(4) - u_phys_e(6));
            ];
            b_cpl = max(b_cpl, 0);
            
            % C. Trust Region Step Constraint (Infinity Norm)
            % |E*w| <= Delta
            A_step = [E_prime; -E_prime];
            b_step = Delta_curr * ones(2*size(E_prime,1), 1);
            
            % Combine
            A = [A_box; A_cpl; A_step];
            b = [b_box; b_cpl; b_step];
        end
        
        function [J, dJ] = cost_function_wrapper(~, o_e, H, w, u_e, E_prime, lambda, W2)
            % COST_FUNCTION_WRAPPER Matches the definition in original paper
            % J = 0.5 * || o_e + H*w ||^2 + Regularization
            
            u = u_e + E_prime * w; 
            r = o_e + H * w;       % Residual
            
            % Objective
            J = 0.5 * (r' * r) + 0.5 * lambda * sum(W2 .* (u.^2));
            
            if nargout > 1
                % Gradient w.r.t w
                % d/dw (0.5*(o+Hw)^T(o+Hw)) = H^T(o+Hw) = H^T * r
                % d/dw (0.5*lambda*u^T W2 u) = lambda * E^T * (W2 .* u)
                dJ = H' * r + lambda * (E_prime' * (W2 .* u));
            end
        end
    end
end