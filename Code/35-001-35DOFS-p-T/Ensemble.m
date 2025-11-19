function u_r_ensemble = Ensemble(Nen,u_e)

    N_Dofs = length(u_e);

    % Generate symmetric matrix such that the sum of the elements of each row is zero
    A = ones(N_Dofs)*(-1/(N_Dofs-1));
    for i = 1:N_Dofs
        A(i,i) = 1;
    end
    covar = A;

    % Augmented covar matrix
    C = [1, zeros(1,N_Dofs);
        zeros(N_Dofs,1), covar];
    
    % Singular value decomposition
    [AA, BB, CC] = svd(C);
    
    % form a new covar matrix that is positive definite
    SS = AA(1:end,1:end-1)*BB(1:end-1,1:end-1)*(CC(1:end,1:end-1))';
    covar = SS(1:end-1,1:end-1);
   
    % Generate an ensemble of Y*Nen members with zero mean and desired covariance 
    Y = 1000;           % Factor to generate Y*Nen members (User-defined value)
    R = chol(covar);    % Cholesky factorization
    L = R';
    mu = zeros(1,N_Dofs);
    variance = eye(N_Dofs);
    
    for k = 1:Y*Nen
       y_hat(:,k) = rand(N_Dofs,1)*2-1; 
       y_hat(:,k) = y_hat(:,k) - mean(y_hat(:,k));
       y_hat(:,k) = y_hat(:,k)/std(y_hat(:,k));
    end
    P_hat = L*y_hat; % Matrix storing vectors with zero mean and desired covariance
    
    % Extract Nen members using SVD
    [U_hat, S_hat, V_hat] = svd(P_hat);
    U = U_hat(1:N_Dofs,1:Nen);
    S = S_hat(1:Nen,1:Nen);
    V = V_hat(1:Nen,1:Nen);
    P = (U*(1/sqrt(Y))*S*V'); % Matrix storing Nen vectors that satisfy the constraints
    
    % Revert the change of variable
    e_e = u_e.*u_e;
    
    S = zeros(N_Dofs);
    for i = 1:N_Dofs
        S(i,i) = 0.005;    % Desired variance (User-defined value). It defines how far are the ensemble members from their mean
    end
    
    % Revert change of variable
    for i = 1:Nen
        e_r_ensemble = S*P(:,i) + e_e;
        mask = find(u_e(:)<0);
    
        if (e_r_ensemble(:)) > 0
            u_r_ensemble(:,i) = sqrt(e_r_ensemble);
            u_r_ensemble(mask,i) = - u_r_ensemble(mask,i);
        else
           e_r_ensemble = -S*P(:,i) + e_e;
           u_r_ensemble(:,i) = sqrt(abs(e_r_ensemble));
           u_r_ensemble(mask,i) = - u_r_ensemble(mask,i);
        end
    end 
    
end

