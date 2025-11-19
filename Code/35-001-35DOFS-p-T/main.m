% ----------------------------------------------------------------------- %
% --------------------------- Optimization of --------------------------- %
% ---------------------------- Cylinder Wake ---------------------------- %
% --------------------- with Stochastic Forcing term -------------------- %
% ----------------------------------------------------------------------- %
% ------------------------ Javier Lorente Macías ------------------------ %
% ------------------------    Yacine Bengana     ------------------------ %
% ----------------------------------------------------------------------- %
clc;
clear all;
close all;

name = ['Results/Drag_m01.dat';'Results/Drag_m02.dat';'Results/Drag_m03.dat';'Results/Drag_m04.dat';'Results/Drag_m05.dat';...
'Results/Drag_m06.dat';'Results/Drag_m07.dat';'Results/Drag_m08.dat';'Results/Drag_m09.dat';'Results/Drag_m0T.dat';...
'Results/Drag_m11.dat';'Results/Drag_m12.dat';'Results/Drag_m13.dat';'Results/Drag_m14.dat';'Results/Drag_m15.dat';...
'Results/Drag_m16.dat';'Results/Drag_m17.dat';'Results/Drag_m18.dat';'Results/Drag_m19.dat';'Results/Drag_mTT.dat';...
'Results/Drag_m21.dat';'Results/Drag_m22.dat';'Results/Drag_m23.dat';'Results/Drag_m24.dat';'Results/Drag_m25.dat';...
'Results/Drag_m26.dat';'Results/Drag_m27.dat';'Results/Drag_m28.dat';'Results/Drag_m29.dat';'Results/Drag_m0Q.dat';...
'Results/Drag_m31.dat';'Results/Drag_m32.dat';'Results/Drag_m33.dat';'Results/Drag_m34.dat';'Results/Drag_m35.dat';...
];

%% Initial computation
Nen = 35;			               % Number of ensemble members
lambda = 20;			           % Weight of the penalty term
a_e = importdata('a_Initial.txt'); % Initial control vector
e0 = norm(a_e)^2;	               % squared norm of the control vector
a_e_mat(:,1) = a_e;
N_Dofs = length(a_e);		       % Number of DOFs (components of the control vector)

% Compute cost function corresponding to the initial geometry
system(['FreeFem++ ' 'First.edp']);	% Freefem first geometry
Cd = 2*importdata('Drag.dat');	    % Drag coefficient
o_e = obs_vec(Cd(10000:end,1));	    % Observation vector
Cost_Function(1,1) = o_e		% Cost function (mean drag coefficient, equal to the observation vector in this case)
Cd = [];

% Compute cost function including penalty term
for i = 1:N_Dofs
    v(i,1) = (i)*a_e(i,1);	% Penalty term to ensure smooth geometries (first derivative)
end
val(1,1) = (1/2)*(norm(o_e))^2 + (1/2)*lambda*(norm(v))^2	% J = 1/2*norm(Drag_coefficient)^2 + penalty term

%% Iterations
for i = 1:30

    % Generation of the ensemble members
    a_r_ensemble = Ensemble(Nen, a_e);
    writematrix(a_r_ensemble', "a_r_ensemble.txt",'Delimiter',' ');

    % Deviation matrix, E'
    E_prime = a_r_ensemble - a_e;
    
    % Run Nen simulations (one for each ensemble member)
    system('sh RunSimulations.sh');
    pause(1700) % Pause to get all drag files saved before continuing
    
    % Compute drag coefficient of each ensemble member
    for k = 1:Nen
        Cd = 2*importdata(name(k,:));
        o_r_mat(k) = obs_vec(Cd(10000:end,1));
        Cd = [];
    end
    
    % Compute observation matrix
    H = (o_r_mat - o_e);
    
    % Constrain for small step vector size: |E*a|<eps
    aaa = max(abs(E_prime));
    bbb = 10.^floor(log10(aaa));
    bbb = ((1./bbb))'*Nen;
    LB = -ones(Nen,1).*bbb/Nen^2;	% Lower bound
    UB = ones(Nen,1).*bbb/Nen^2;	% Upper bound

    % Solve optimization problem
    J = @(w) Cost(o_e, H, w, a_e, E_prime, lambda);	% Cost function
    Con = @(w) Constraints(a_e, E_prime, e0, w);	% Constraints
    Hess = @(w,lambda) Hessianfcn(w,lambda,E_prime,H);	% Hessian matrix
    options = optimset('GradConstr','off','GradObj','off','MaxFunEvals',100000);
    [w_opt, J_opt] = fmincon(J, rand(Nen,1), [], [], [], [], LB, UB, Con, options); % Optimal weight vector and cost function
    max(abs(E_prime*w_opt))	% Print maximum step size (just to check it is small)
    
    % Update control vector
    a_e = a_e + E_prime*w_opt;
    a_e_mat(:,i+1) = a_e;
    delete 'a_e.txt';
    writematrix(a_e, "a_e.txt",'Delimiter',' ');
    
    % Run simulation for the optimal control vector
    system(['FreeFem++ ' 'Second.edp']);
    Cd = 2*importdata('DragSecond.dat');
    o_e = obs_vec(Cd(10000:end,1));
    Cost_Function(1,i+1) = o_e
    writematrix([i,o_e], "Cost_Iter.txt",'Delimiter',' ');
    Cd = [];
    
    % Compute cost function including penalty term of the optimal solution
    for j = 1:N_Dofs
        v(j,1) = (j)*a_e(j,1);
    end
    val(i+1,1) = (1/2)*(norm(o_e))^2 + (1/2)*lambda*(norm(v))^2
    
    % Save variables
    writematrix(a_e_mat, "Evolution_a_e.txt",'Delimiter',' ');	 % Evolution of the optimal control vector
    writematrix(Cost_Function', "Cost_Function.txt",'Delimiter',' '); % Evolution of the drag coefficient
    writematrix(val, "J_Obj.txt",'Delimiter', ' ');			% Evolution of the cost function (including penalty term)
end

