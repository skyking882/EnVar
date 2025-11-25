clear 
clc

N_Dofs=6;
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