function [J, dJ, Hessian] = Cost(o_e, H, a, u_e, E_prime, lambda)
    Nen = size(E_prime,2);
    N_Dofs = length(u_e);

    for i = 1:N_Dofs
        v(i,1) = (i)*(u_e(i,1) + E_prime(i,:)*a);
    end

    % Cost Function
    J = (1/2)*(norm(o_e + H*a))^2 + (1/2)*lambda*(norm(v))^2;
    
    % Gradient of cost function
    if nargout > 1 % Gradient required
        dJ = H'*(o_e + H*a);

	for i = 1:Nen
            s = 0;
            for j = 1:N_Dofs
                s = s + E_prime(j,i)*E_prime(j,:)*a;
		s = j*s;
	    end
        df(i,1) = s;
	end
    df = df/norm(vv);


	dJ = dJ + lambda*df;
        
        % Hessian of cost function
        if nargout > 2 % Hessian required
	    
            for i = 1:Nen
                for j = 1:N_Dofs
                    ddf(i,j) = E_prime(j,i)*E_prime(j,i)*a(i,1);
		end
	    end
            Hessian = H'*H + lambda*ddf;
        end
        
    end  
    
end

