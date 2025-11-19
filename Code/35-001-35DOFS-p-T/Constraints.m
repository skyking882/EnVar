function [C, Ceq, gC, gCeq] = Constraints(u_e, E_prime, c, a)
    
    indices = 1:length(u_e);
    indices_even = indices(2:2:end);
    E_prime_even = E_prime([indices_even],:);
    u_e_even = u_e([indices_even],1);

    L = 3; % Maximum length
    a0 = 0.5; % zeroth Fourier coefficient

    % Constraints
    C(1:length(indices_even),1) = u_e_even + E_prime_even*a - (L/2 - a0);	% Length constraint
    C(((length(indices_even)+1):(length(indices_even)+length(indices))),1) = abs(E_prime*a) - 0.075*ones(length(u_e),1);	% Step size constraint
    Ceq = u_e'*u_e + 2*u_e'*E_prime*a + a'*E_prime'*E_prime*a - c; % Norm constraint
    
    % Gradient of constraints
    if nargout > 2
        gC = E_prime_even';
        gCeq = (2*(u_e'*E_prime)*eye(size(E_prime,2)))' + ...
               eye(size(E_prime,2))*E_prime'*(E_prime*a) + ...
               ((a'*E_prime')*E_prime*eye(size(E_prime,2)))';
    end
    
end

