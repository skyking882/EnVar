classdef InsectModel < handle
    % INSECTMODEL Fruit Fly Aerodynamic & Physical Model
    %   This class handles the morphology, kinematic parameter scaling, 
    %   and the interface to the aerodynamic solver.
    %
    %   Units: SI (kg, m, s, rad) unless otherwise specified.
    
    properties (SetAccess = private)
        % --- Morphology & Aerodynamic Constants ---
        opts        % Struct containing mass, inertia, aero coeffs
        
        % --- Optimization Bounds (Physical) ---
        LB_phys     % Lower bounds [10x1]
        UB_phys     % Upper bounds [10x1]
        
        % --- Scaling Factors (Affine Mapping) ---
        u_mid       % Midpoint offset
        u_half      % Half-range scale
        
        % --- Metadata ---
        param_names % Cell array of parameter names
        is_angle    % Boolean array (true if parameter is an angle)
    end
    
    methods
        function obj = InsectModel()
            % CONSTRUCTOR: Initialize morphology and bounds
            obj.init_morphology();
            obj.init_bounds();
            obj.init_scaling();
        end
        
        %% --- Core Physics Interface ---
        
        function [J, L_mean, P_star] = evaluate(obj, u_norm, r_lift)
            % EVALUATE Run the forward aerodynamic model.
            %
            % Input:
            %   u_norm : Normalized parameter vector [-1, 1]
            %   r_lift : Penalty weight for lift constraint
            %
            % Output:
            %   J      : Augmented cost function value
            %   L_mean : Mean Lift (normalized by weight)
            %   P_star : Aerodynamic Power
            
            % 1. Map normalized parameters to physical space
            u_phys = obj.norm2phys(u_norm);
            
            % 2. Unpack parameters for readability
            p = num2cell(u_phys);
            [f, phi_m, theta_m, eta_m, theta_0, eta_0, ...
             K, C_eta, Phi_th, Phi_et] = p{:};
             
            % 3. Call external solver (Quasi-steady solver)
            % Note: 'N' is fixed to 2 as per original script
            N_harmonics = 2; 
            
            % Call the external function (assumed to be in path)
            data = calculate_wing_aero(...
                f, phi_m, theta_m, eta_m, theta_0, eta_0, ...
                K, C_eta, N_harmonics, Phi_th, Phi_et, obj.opts);
            
            % 4. Extract Metrics
            L_mean = data.aero.L_mean;
            P_star = data.power.Pstar;
            
            % 5. Compute Cost J = P* + r * (1 - L)^2
            lift_gap = (1.0 - L_mean);
            J = P_star + r_lift * (lift_gap^2);
        end
        
        %% --- Feasibility & Projection ---
        
        function u_norm_clean = project_feasible(obj, u_norm)
            % PROJECT_FEASIBLE Project a normalized vector into the valid
            % physical domain, respecting Box Bounds and Coupling Constraints.
            
            % 1. Map to physical
            u = obj.norm2phys(u_norm);
            
            % 2. Box Clamp (Simple bounds)
            u = min(max(u, obj.LB_phys), obj.UB_phys);
            
            % 3. Coupled Constraints (Berman & Wang, 2007, Table 1)
            % Ensure offsets do not exceed kinematic limits relative to amplitudes.
            theta_m = u(3);
            eta_m   = u(4);
            
            % Constraint: |theta_0| + theta_m <= pi/2
            u(5) = min(max(u(5), theta_m - pi/2), pi/2 - theta_m);
            
            % Constraint: |eta_0| + eta_m <= pi
            u(6) = min(max(u(6), eta_m - pi), pi - eta_m);
            
            % 4. Phase Wrapping [-pi, pi]
            % Indices 9 (Phi_th) and 10 (Phi_et)
            u(9)  = mod(u(9)  + pi, 2*pi) - pi;
            u(10) = mod(u(10) + pi, 2*pi) - pi;
            
            % 5. Map back to normalized
            u_norm_clean = obj.phys2norm(u);
            
            % 6. Numerical safety clamp (keep slightly inside [-1, 1])
            eps_safe = 1e-12;
            u_norm_clean = min(max(u_norm_clean, -1 + eps_safe), 1 - eps_safe);
        end
        
        %% --- Helper: Space Conversion ---
        
        function u_phys = norm2phys(obj, u_norm)
            % Convert Normalized [-1, 1] -> Physical
            u_phys = obj.u_mid + obj.u_half .* u_norm;
        end
        
        function u_norm = phys2norm(obj, u_phys)
            % Convert Physical -> Normalized [-1, 1]
            u_norm = (u_phys - obj.u_mid) ./ obj.u_half;
        end
        
        function print_parameters(obj, u_norm)
            % Utility to print physical values from a normalized vector
            u_p = obj.norm2phys(u_norm);
            fprintf('--- Current Parameters ---\n');
            for i = 1:length(u_p)
                val = u_p(i);
                unit = '';
                if obj.is_angle(i)
                    val = val * (180/pi);
                    unit = 'deg';
                elseif i == 1
                    unit = 'Hz'; 
                end
                fprintf('%-8s : %.4f %s\n', obj.param_names{i}, val, unit);
            end
        end

    end
    
    methods (Access = private)
        function init_morphology(obj)
            % Initialize Fruit Fly parameters (Table 2)
            o.m_insect = 0.72e-6;          % kg
            o.M_wing   = 8.6e-10;          % kg
            o.R        = 2.02e-3;          % m
            o.cbar     = 0.67e-3;          % m
            o.Iwing    = 0.80e-15;         % kg m^2
            
            % Aerodynamic Coefficients
            o.CT    = 1.833;
            o.CD0   = 0.21;
            o.CDpi2 = 3.35;
            o.CR    = pi;
            
            % Viscous torque parameters
            o.mu1   = 0.2;
            o.mu2   = 0.2;
            
            % Environment & Simulation
            o.rho_f = 1.29;
            o.g     = 9.81;
            o.Nr    = 200;  % Radial integration points
            o.Nt    = 1000; % Time integration points
            
            % Body Inertias (Approximated based on wing)
            o.Iphi   = o.Iwing;
            o.Itheta = o.Iwing;
            o.Ieta   = o.Iwing * (o.cbar / o.R)^2;
            
            obj.opts = o;
        end
        
        function init_bounds(obj)
            % Define Physical Bounds (Lower and Upper)
            % Order: [f, phi_m, th_m, et_m, th_0, et_0, K, C_et, Ph_th, Ph_et]
            
            obj.LB_phys = [10;   0;    0;    0;    -pi/2; -pi;  0.01; 0.1; -pi; -pi]; 
            obj.UB_phys = [500;  pi/2; pi/2; pi;    pi/2;  pi;  0.99; 20;   pi;  pi];
            
            obj.param_names = {'f', 'phi_m', 'theta_m', 'eta_m', ...
                               'theta_0', 'eta_0', 'K', 'C_eta', ...
                               'Phi_theta', 'Phi_eta'};
                               
            obj.is_angle = [0, 1, 1, 1, 1, 1, 0, 0, 1, 1];
        end
        
        function init_scaling(obj)
            % Pre-calculate affine scaling matrices
            obj.u_mid  = 0.5 * (obj.UB_phys + obj.LB_phys);
            obj.u_half = 0.5 * (obj.UB_phys - obj.LB_phys);
        end
    end
end