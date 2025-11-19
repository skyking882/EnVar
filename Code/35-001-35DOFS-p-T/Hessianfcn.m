function Hout = Hessianfcn(a,lambda,H,E_prime)
    % Hessian of the objective function
    Nen = size(E_prime,2);
    N_Dofs = size(E_prime,1);
             for i = 1:Nen
                for j = 1:N_Dofs
                    ddf(i,j) = E_prime(j,i)*E_prime(j,i)*a(i,1);
                end
            end
            Hessian = H'*H + lambda*ddf;


    %Hessian = H'*H;
    % Hessian of the nonlinear equality constraint
    Hh = 2*E_prime'*E_prime;
    % Hessian of the nonlinear inequality constraint
    Hg = zeros(size(E_prime,1));
    
    % Hessian of the Lagrangian
    Hout = Hessian + lambda.eqnonlin*Hh + lambda.ineqnonlin*Hg;
    
end

