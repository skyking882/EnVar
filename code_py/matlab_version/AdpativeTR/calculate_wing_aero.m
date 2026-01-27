function data = calculate_wing_aero( ...
    f, phi_m, theta_m, eta_m, theta_0, eta_0, K, C_eta, N, Phi_theta, Phi_eta, opts)
%CALCULATE_WING_AERO_BW2007
% Strict rewrite of Berman & Wang (2007), JFM 582:153–168, Eqs (2.1)–(2.25).
%
% Inputs: (11 optimizable parameters, Table 1)
%   f         : flapping frequency (Hz)
%   phi_m     : azimuthal amplitude (rad)
%   theta_m   : vertical amplitude (rad)
%   eta_m     : pitching amplitude (rad)
%   theta_0   : vertical offset (rad)
%   eta_0     : pitching offset (rad)
%   K         : shape parameter for phi(t), 0<K<1
%   C_eta     : sharpness parameter for eta(t), >=0
%   N         : multiplier of theta(t) period (1 or 2)
%   Phi_theta : phase offset for theta(t) (rad)
%   Phi_eta   : phase offset for eta(t) (rad)
%
% opts (optional struct):
%   opts.Nt      : number of time steps per period (default 1000 per paper)
%   opts.Nr      : number of radial blade elements (default 200)
%   opts.rho_f   : fluid density (kg/m^3), default 1.29
%   opts.g       : gravity (m/s^2), default 9.81
%
%   Hawkmoth morphology (Table 2) defaults:
%     opts.m_insect  : insect mass (kg), default 1648 mg
%     opts.M_wing    : wing mass (kg), default 47 mg
%     opts.R         : wing length (m), default 51.9 mm
%     opts.cbar      : mean chord (m), default 18.26 mm
%     opts.Iwing     : wing inertia (kg m^2), default 0.184 g cm^2
%
%   Aerodynamic coeffs (Table 2 + text):
%     opts.CT     : translational lift coeff, default 1.678
%     opts.CD0    : CD(0), default 0.07
%     opts.CDpi2  : CD(pi/2), default 3.06
%     opts.CR     : rotational lift coeff, default pi
%
%   Thickness/viscous torque params (paper gives form but not numbers here):
%     opts.b      : semi-minor axis thickness (m), default 0.01*cbar
%     opts.mu1    : viscous torque coeff, default 0.2 (placeholder)
%     opts.mu2    : viscous torque coeff, default 0.0 (placeholder)
%
%   Inertia per rotation axis (for Eq 2.22):
%     opts.Iphi, opts.Itheta, opts.Ieta : default all = opts.Iwing
%
% Output struct:
%   data.time
%   data.kinematics.(phi,theta,eta and derivatives)
%   data.aero.(F_lab, Fz, L_t, L_mean, Fh_t, tau_aero_*, p_*, P_*)

    if nargin < 12 || isempty(opts), opts = struct(); end

    % ---------- Defaults ----------
    if ~isfield(opts,'Nt'),     opts.Nt = 1000; end
    if ~isfield(opts,'Nr'),     opts.Nr = 200; end
    if ~isfield(opts,'rho_f'),  opts.rho_f = 1.29; end
    if ~isfield(opts,'g'),      opts.g = 9.81; end

    % Hawkmoth morphology defaults (Table 2)
    if ~isfield(opts,'m_insect'), opts.m_insect = 1648e-6; end % kg
    if ~isfield(opts,'M_wing'),   opts.M_wing   = 47e-6;   end % kg
    if ~isfield(opts,'R'),        opts.R        = 51.9e-3; end % m
    if ~isfield(opts,'cbar'),     opts.cbar     = 18.26e-3;end % m
    if ~isfield(opts,'Iwing')
        % 0.184 g cm^2 -> 0.184 * 1e-7 kg m^2
        opts.Iwing = 0.184e-7;
    end

    % Aero coeff defaults (Table 2 + text around (2.16)-(2.17))
    if ~isfield(opts,'CT'),    opts.CT = 1.678; end
    if ~isfield(opts,'CD0'),   opts.CD0 = 0.07; end
    if ~isfield(opts,'CDpi2'), opts.CDpi2 = 3.06; end
    if ~isfield(opts,'CR'),    opts.CR = pi; end

    % b, mu1, mu2 are not numerically specified in this paper; keep configurable
    if ~isfield(opts,'b'),   opts.b = 0.01 * opts.cbar; end
    if ~isfield(opts,'mu1'), opts.mu1 = 0.2; end
    if ~isfield(opts,'mu2'), opts.mu2 = 0.2; end

    % Inertia per axis for Eq (2.22)
    if ~isfield(opts,'Iphi'),   opts.Iphi   = opts.Iwing; end
    if ~isfield(opts,'Itheta'), opts.Itheta = opts.Iwing; end
    if ~isfield(opts,'Ieta'),   opts.Ieta   = opts.Iwing * (opts.cbar / opts.R)^2; end

    Nt   = opts.Nt;
    Nr   = opts.Nr;
    rho  = opts.rho_f;
    g    = opts.g;
    R    = opts.R;
    cbar = opts.cbar;
    Mwing = opts.M_wing;

    % ---------- Time grid ----------
    T = 1 / f;
    t = linspace(0, T, Nt);
    dt = t(2) - t(1);
    w  = 2*pi*f;

    % ---------- Radial grid & chord (Eq 2.9) ----------
    dr    = R / Nr;
    r_vec = (dr/2 : dr : R-dr/2)';                       % Nr x 1
    c_r   = (4*cbar/pi) * sqrt(1 - (r_vec./R).^2);       % Nr x 1

    % ---------- Kinematics (Eqs 2.10-2.12) ----------
    % phi(t): smoothed triangular
    s = w*t; % 1xNt

    if abs(K) < 1e-8
        % K -> 0 limit: phi = phi_m * sin(w t)
        phi      = phi_m * sin(s);
        phi_dot  = phi_m * w * cos(s);
        phi_ddot = -phi_m * w^2 * sin(s);
    else
        Aphi = phi_m / asin(K);
        u    = K * sin(s);
        D    = sqrt(1 - u.^2);

        phi      = Aphi * asin(u);
        phi_dot  = Aphi * (K*w*cos(s)) ./ D;

        % analytic second derivative
        B = Aphi * K * w;
        % phi_ddot = B * d/dt( cos(s)/D )
        % = B * [(-w sin(s))/D + cos(s) * (K^2 w sin(s) cos(s))/D^3]
        phi_ddot = B .* ( (-w*sin(s))./D + (cos(s) .* (K^2*w*sin(s).*cos(s)) ./ (D.^3)) );
    end

    % theta(t): sinusoid (Eq 2.11)
    st = 2*pi*N*f*t + Phi_theta;
    theta      = -theta_m * cos(st) + theta_0;
    theta_dot  = theta_m * (2*pi*N*f) * sin(st);
    theta_ddot = theta_m * (2*pi*N*f)^2 * cos(st);

    % eta(t): tanh waveform (Eq 2.12)
    if abs(C_eta) < 1e-8
        % C_eta -> 0 limit: eta = eta_m * sin(w t + Phi_eta) + eta_0
        se = w*t + Phi_eta;
        eta      = eta_m * sin(se) + eta_0;
        eta_dot  = eta_m * w * cos(se);
        eta_ddot = -eta_m * w^2 * sin(se);
    else
        Ae  = eta_m / tanh(C_eta);
        se  = w*t + Phi_eta;
        u   = C_eta * sin(se);
        th  = tanh(u);
        sech2 = 1 - th.^2;

        eta     = Ae * th + eta_0;

        du_dt   = C_eta * cos(se) * w;
        d2u_dt2 = -C_eta * sin(se) * w^2;

        eta_dot  = Ae * sech2 .* du_dt;

        % eta_ddot = Ae * sech^2(u) * [ d2u_dt2 - 2*tanh(u)*(du_dt)^2 ]
        eta_ddot = Ae * sech2 .* ( d2u_dt2 - 2*th.*(du_dt.^2) );
    end

    % ---------- Preallocate outputs ----------
    F_lab = zeros(3, Nt);           % total force (one wing) in lab frame
    tau_lab = zeros(3, Nt);         % torque from r x dF (for phi/theta) in lab frame
    tau_eta = zeros(1, Nt);         % pitching torque about radial axis (Eq 2.15 integrated)

    % ---------- Blade-element loop over time ----------
    for it = 1:Nt
        ph  = phi(it);    phd  = phi_dot(it);    phdd  = phi_ddot(it);
        thh = theta(it);  thd  = theta_dot(it);  thdd  = theta_ddot(it);
        et  = eta(it);    etd  = eta_dot(it);    etdd  = eta_ddot(it);

        % Co-moving velocities/accelerations (Eqs 2.5-2.8)
        % vx'' = r( phidot cosθ cosη + thetadot sinη )
        % vy'' = r( thetadot cosη - phidot cosθ sinη )
        cth = cos(thh); sth = sin(thh);
        cet = cos(et);  set = sin(et);

        vx = r_vec .* (phd*cth*cet + thd*set);
        vy = r_vec .* (thd*cet - phd*cth*set);

        ax = r_vec .* ( (phdd*cth + thd*(etd - phd*sth))*cet + (thdd - etd*phd*cth)*set );
        ay = r_vec .* ( (thd*(etd - phd*sth) - phdd*cth)*set + (thdd - etd*phd*cth)*cet );

        vmag = sqrt(vx.^2 + vy.^2);
        alpha = atan2(-vy, -vx);
        aoa(it,:) = alpha;
        % Circulation (Eq 2.16)
Gamma_T = -0.5*opts.CT .* c_r .* vmag .* sin(2*alpha);
Gamma_R =  0.5*opts.CR .* (c_r.^2) .* etd;
Gamma   =  Gamma_T + Gamma_R;


        % Added-mass terms (Eq 2.19)
        b = opts.b;
        m11 = (pi/4) * rho * b^2;              % scalar
        m22 = (pi/4) * rho .* (c_r.^2);        % Nr x 1
        Ia  = (pi/128) * rho .* ( (c_r.^2 + b^2).^2 ); % Nr x 1

        % Viscous force (Eq 2.17): dFnu = 1/2 rho c CD |v| [vx, vy] dr
        CD = opts.CD0 * (cos(alpha).^2) + opts.CDpi2 * (sin(alpha).^2);
        dFnu_x = 0.5*rho .* c_r .* CD .* vmag .* vx * dr;
        dFnu_y = 0.5*rho .* c_r .* CD .* vmag .* vy * dr;

        % Forces on slice (Eqs 2.13-2.14)
        mass_dist = (c_r/(cbar*R)) * Mwing; % Nr x 1

        dFx = ( (mass_dist + m22) .* (vy*etd) ...
               - rho .* Gamma .* vy ...
               - m11 .* ax ) * dr ...
               - dFnu_x;

        dFy = ( -(mass_dist + m11) .* (vx*etd) ...
               + rho .* Gamma .* vx ...
               - m22 .* ay ) * dr ...
               - dFnu_y;

        % Pitching torque about radial axis (Eq 2.15)
        dTau_nu = (1/16)*pi*rho .* (c_r.^4) .* (opts.mu1*f + opts.mu2*abs(etd)) .* etd * dr;
        dTau_eta = ( (m11 - m22) .* (vx.*vy) - Ia .* etdd ) * dr - dTau_nu;

        % Transform each slice force back to lab frame using (2.2)-(2.4)
        % [xhat''; yhat''] = R1 R2 [xhat;yhat;zhat]  =>  Flab = (R1 R2)' * [Fx''; Fy'']
        R2 = [ -sin(ph),          cos(ph),          0; ...
               -sin(thh)*cos(ph), -sin(thh)*sin(ph), cos(thh) ];
        R1 = [  cos(et),  sin(et); ...
               -sin(et),  cos(et) ];
        A = R1*R2; % 2x3

        % Accumulate total force & torque
        Fsum = zeros(3,1);
        tausum = zeros(3,1);

        % Unit vectors for projections (spherical basis)
        phi_hat   = [-sin(ph); cos(ph); 0];
        theta_hat = [-sin(thh)*cos(ph); -sin(thh)*sin(ph); cos(thh)];
        r_hat     = [cos(thh)*cos(ph);  cos(thh)*sin(ph);  sin(thh)];

        for ir = 1:Nr
            dF_lab = A' * [dFx(ir); dFy(ir)];  % 3x1

            % position vector of element (Eq 2.1)
            rr = r_vec(ir);
            r_lab = rr * r_hat;

            Fsum = Fsum + dF_lab;

            % torque from r x dF (Eq 2.20)
            tausum = tausum + cross(r_lab, dF_lab);
        end

        F_lab(:,it) = Fsum;
        tau_lab(:,it) = tausum;

        % tau_eta is separate integral of dTau_eta (about radial axis)
        tau_eta(it) = sum(dTau_eta);

        % Note: component of tausum along r_hat is (numerically) ~0 by construction,
        % so eta-torque comes from Eq (2.15) only.
    end

    % ---------- Lift definition (Eq 2.21) ----------
    Fz = F_lab(3,:);    
    L_t = 2 * Fz / (opts.m_insect * g);   
    L_mean = mean(L_t);

    % Horizontal force magnitude (for diagnostics like Fig.4 middle)
    Fh = sqrt(F_lab(1,:).^2 + F_lab(2,:).^2);

    % ---------- Aerodynamic torque components ----------
    % Correction: 
    % - Phi rotation is about the Lab Z-axis.
    % - Theta rotation is about the Line of Nodes ([-sin(phi), cos(phi), 0]).
    
    tau_phi = zeros(1,Nt);
    tau_theta = zeros(1,Nt);
    
    z_hat = [0; 0; 1]; % Axis of rotation for phi (Azimuth)
    
    for it=1:Nt
        ph = phi(it); 
        
        % Axis of rotation for theta (Line of Nodes, perpendicular to Z and r)
        n_hat = [-sin(ph); cos(ph); 0]; 
        
        % Project torque onto the correct rotation axes
        tau_phi(it)   = dot(tau_lab(:,it), z_hat);
        tau_theta(it) = dot(tau_lab(:,it), n_hat);
    end
    tau_eta_arr = tau_eta;

    % ---------- Power model (Eqs 2.22-2.25) ----------
    % Interpret Ωi as angular velocity in coordinate i: Ωφ=φ̇, Ωθ=θ̇, Ωη=η̇
    Ophi = phi_dot;
    Oth  = theta_dot;
    Oet  = eta_dot;

    Odphi = phi_ddot;
    Odth  = theta_ddot;
    Odet  = eta_ddot;

    Iphi = opts.Iphi;
    Ith  = opts.Itheta;
    Iet  = opts.Ieta;

    % Eq (2.22): p_i = Ω_i [ I_i Ωdot_i - Ω_j Ω_k (I_j - I_k) - τ_aero_i ]
    p_phi   = Ophi .* ( Iphi*Odphi - (Oth.*Oet).*(Ith - Iet) - tau_phi );
    p_theta = Oth  .* ( Ith *Odth  - (Oet.*Ophi).*(Iet - Iphi) - tau_theta );
    p_eta   = Oet  .* ( Iet *Odet  - (Ophi.*Oth).*(Iphi - Ith) - tau_eta_arr );

    % Positive power operator Xi (Eqs 2.23-2.24)
    Pphi_t = max(p_phi,   0);
    Pth_t  = max(p_theta, 0);
    Pet_t  = max(p_eta,   0);

% Time-averaged positive power per component over a period (One Wing)
    Pphi = (1/T) * trapz(t, Pphi_t);
    Pth  = (1/T) * trapz(t, Pth_t);
    Pet  = (1/T) * trapz(t, Pet_t);

    % Total Power = 2 * (Power of one wing)
    % Pstar = Total Power / Mass
    Pstar = 2 * (Pphi + Pth + Pet) / opts.m_insect; % Eq (2.25), W/kg, adjusted for 2 wings

    % ---------- Pack outputs ----------
    data.time = t;

    data.kinematics.phi = phi;
    data.kinematics.theta = theta;
    data.kinematics.eta = eta;
    data.kinematics.phi_dot = phi_dot;
    data.kinematics.theta_dot = theta_dot;
    data.kinematics.eta_dot = eta_dot;
    data.kinematics.phi_ddot = phi_ddot;
    data.kinematics.theta_ddot = theta_ddot;
    data.kinematics.eta_ddot = eta_ddot;

    data.aero.F_lab = F_lab;          % N, one wing
    data.aero.Fz = Fz;               % N, magnitude of z-component, one wing
    data.aero.Fh = Fh;               % N, horizontal magnitude, one wing
    data.aero.L_t = L_t;             % nondim lift, Eq (2.21)
    data.aero.L_mean = L_mean;

    data.aero.tau_phi = tau_phi;     % N*m, Eq (2.20) projection
    data.aero.tau_theta = tau_theta; % N*m, Eq (2.20) projection
    data.aero.tau_eta = tau_eta_arr; % N*m, Eq (2.15) integrated

    data.power.p_phi = p_phi;
    data.power.p_theta = p_theta;
    data.power.p_eta = p_eta;

    data.power.Pphi_t = Pphi_t;
    data.power.Ptheta_t = Pth_t;
    data.power.Peta_t = Pet_t;

    data.power.Pphi = Pphi;          % W
    data.power.Ptheta = Pth;         % W
    data.power.Peta = Pet;           % W
    data.power.Pstar = Pstar;        % W/kg (Eq 2.25)

    % echo used parameters for reproducibility
    data.meta.params.f = f;
    data.meta.params.phi_m = phi_m;
    data.meta.params.theta_m = theta_m;
    data.meta.params.eta_m = eta_m;
    data.meta.params.theta_0 = theta_0;
    data.meta.params.eta_0 = eta_0;
    data.meta.params.K = K;
    data.meta.params.C_eta = C_eta;
    data.meta.params.N = N;
    data.meta.params.Phi_theta = Phi_theta;
    data.meta.params.Phi_eta = Phi_eta;

    data.meta.opts = opts;
    data.aero.alpha = aoa;
    
end