function morphing_drone_mpc_switching
% SWITCHING-MPC morphing quadrotor simulator (planar X/H/Y/T formations)
% Replaces CODE B's inner PD with an ATTITUDE SWITCHING MPC (Np=40, Nc=12,
% Ts=0.01 s) and switches models by formation. A trajectory tracker maps
% position->(phi_d,theta_d,thrust), then the switching MPC tracks
% [phi,theta,psi] and outputs body torques [tau_x,tau_y,tau_z]. A
% geometry-varying mixer maps [T; tau] -> motor forces with bounds.
%
% HOW TO USE
%   1) Save as morphing_drone_mpc_switching.m
%   2) In MATLAB: clear; close all; clc; morphing_drone_mpc_switching

%% -------------------- PARAMETERS ----------------------------------------
P = struct();
% Mass & geometry (planar morphing about z)
P.m_body = 0.65; P.m_arm = 0.04; P.m_motor = 0.07;
P.m = P.m_body + 4*(P.m_arm + P.m_motor);
P.w = 0.15; P.l = 0.15; P.alpha = 0.09;         % hinge half-width/length, motor offset in XY
P.g = 9.81;

% Aerodynamic / allocation constants
P.b = 1.0;             % thrust coeff (N per motor command unit)
P.kappa = 0.02;        % reaction torque coeff (Nm per unit)

% Motors (bounds + rate limits for allocator)
P.f_min = 0.0; P.f_max = 8.0; P.df_max = 8.0;   % N

% Attitude torque physical saturations (for scaling MPC inputs)
P.tau_max = [0.6; 0.6; 0.3]; % [Nm] about x,y,z

% Attitude torque actuation dynamics (first-order time-constant)
P.tau_alpha = 0.05;    % s  (tau_dot = (u_d - tau)/tau_alpha)

% MPC timing & horizons (attitude switching MPC)
P.Ts = 0.01;           % s (attitude MPC + sim step)
P.Np = 40;             % prediction horizon
P.Nc = 12;             % control horizon

% MPC weights (attitude): Qx on [phi theta psi p q r tau_x tau_y tau_z], Ru on input u_d
P.Qx = diag([40, 40, 40, 80, 80, 80, 0.1, 0.1, 0.1]);
P.Ru = diag([80, 80, 120]);

% MPC constraints in NORMALIZED input domain (u_norm), then scaled -> tau
P.du_max = [0.03; 0.03; 0.03];  % per-sample |Δu_norm|
P.u_max  = [0.10; 0.10; 0.10];  % |u_norm|

% Trajectory tracker (outer loop) — simple PD on position (can swap with your tracker)
P.outer.Kp_pos = diag([1.8, 1.8, 6.0]);
P.outer.Kd_pos = diag([1.4, 1.2, 3.2]);
P.outer.max_tilt = deg2rad(18);
P.outer.Ki_z = 1.0;   % small integral on z

% Simulation horizon
P.Tend = 12.0;                          % seconds

% Formation switching schedule (timed, X->H->Y->T->X every 3 s)
P.switch.Tseg = 3.0;                    % seconds per formation
P.switch.order = {'X','H','Y','T'};     % cycle order

% Scenario selection: 'gap' or 'timed'
P.scenario = 'timed';

% Optional export
P.save_plots = false; P.outdir_name = 'Out_SwitchingMPC';

%% -------------------- PREP: MODELS & MPC FOR EACH FORMATION -------------
% Inertia tensors per formation (illustrative values)
I_table = struct();
I_table.X = diag([0.004233, 0.004380, 0.007834]);
I_table.H = diag([0.005885, 0.001812, 0.006918]);
I_table.Y = diag([0.005042, 0.003096, 0.007369]);
I_table.T = diag([0.003654, 0.003917, 0.006792]);

% Discrete attitude models (9-state) & lifted MPC matrices for each formation
models = struct();
forms = {'X','H','Y','T'};
for k = 1:numel(forms)
    f = forms{k};
    models.(f) = build_att_model_and_mpc(P, I_table.(f));
end

%% -------------------- RUN SCENARIO --------------------------------------
% Trajectories
trajA = @(t) [0.8*t; 0; 1.5];   % straight line (x increases)
trajB = @(t) [0; 0; 1.5];       % hold position

switch P.scenario
    case 'gap'
        ENV = struct('gap_center_x', 4.0, 'gap_span_x', 1.3, 'gap_width_y', 0.20, ...
                     'safe_margin', 0.02, 'lookahead', 1.0);
        log = run_sim_switching(P, models, trajA, 'gap', ENV);
        plot_tracking_reports(P, log, ENV);
    otherwise
        ENV2 = struct('timed', true, 't_fold_on', 3.0, 't_fold_off', 6.5);
        log = run_sim_switching(P, models, trajB, 'timed', ENV2);
        plot_hold_reports(P, log);
end

plot_stability_thrust_dashboard(P, log);
print_metrics(P, log, 'Switching-MPC');

if P.save_plots
    scriptDir = fileparts(mfilename('fullpath'));
    outdir    = fullfile(scriptDir, P.outdir_name);
    try, save_all_plots(outdir); catch ME, warning('Save failed: %s', ME.message); end
    fprintf('All figures saved to: %s\n', outdir);
end

save('switching_mpc_results.mat','P','log');
fprintf('Saved results to switching_mpc_results.mat\n');
end

%% ==================== SIMULATOR WITH SWITCHING MPC =======================
function log = run_sim_switching(P, models, traj, mode, env)
% State: [p; v; eul; omg; tau] = 15x1
x = zeros(15,1); x(1:3) = [0; 0; 1.5];

% MPC memory for inputs (normalized domain)
u_prev = zeros(3,1);

N = floor(P.Tend/P.Ts) + 1; log = alloc_log(N);
int_z = 0; f_prev = (P.m*P.g/4)*ones(4,1);

for k = 1:N
    t  = (k-1)*P.Ts;
    p  = x(1:3); v = x(4:6); eul = x(7:9); omg = x(10:12); %#ok<NASGU>

    % Formation (switch on schedule or by gap lookahead)
    form = pick_formation(P, t, mode, env, p);
    M = models.(form);

    % Outer trajectory tracker -> desired thrust & euler
    [T_des, eul_des, int_z] = outer_tracker(P, traj(t), [0;0;0], p, v, eul, int_z);

    % Attitude Switching MPC (track eul_des; yaw_ref=0)
    yref = [eul_des(1); eul_des(2); 0];
    z_meas = [eul; omg; x(13:15)]; % 9x1 attitude-related state: [eul; omg; tau]

    [du, u_cmd_norm] = mpc_solve_once(M, z_meas, yref, u_prev); % Δu_norm and u_norm
    u_prev = u_cmd_norm;                                         % receding horizon

    % Map normalized input -> physical torques via tau_max scaling
    tau_cmd = (u_cmd_norm ./ P.u_max) .* P.tau_max; % |u_norm|<=0.1 -> |tau|<=tau_max

    % Mixer allocation using planar geometry of current formation
    geom = compute_geometry_planar(P, form);
    A = allocation_matrix_planar(P, geom);
    [f_cmd, infoQP] = motor_qp(P, A, T_des, tau_cmd, f_prev); %#ok<NASGU>

    % Dynamics step (6DoF + 1st-order torque dynamics)
    [~, x_next] = dynamics_step_planar(P, x, f_cmd, geom, form);
    x = x_next; f_prev = f_cmd;

    % Log
    log = push_log_switching(log,k,t,x,yref,T_des,u_cmd_norm,tau_cmd,f_cmd,form,geom,P,infoQP);
end
end

%% ==================== OUTER TRAJECTORY TRACKER ===========================
function [T_des, eul_des, int_z] = outer_tracker(P, pd, vd, p, v, eul, int_z)
    ep = pd - p; ev = vd - v; int_z = int_z + ep(3)*P.Ts;
    a_cmd = P.outer.Kp_pos*ep + P.outer.Kd_pos*ev + [0;0;P.g] + [0;0;P.outer.Ki_z*int_z];
    phi_des   = -(1/P.g)*a_cmd(2);   % roll
    theta_des =  (1/P.g)*a_cmd(1);   % pitch
    phi_des   = max(min(phi_des,  P.outer.max_tilt), -P.outer.max_tilt);
    theta_des = max(min(theta_des, P.outer.max_tilt), -P.outer.max_tilt);
    T_des = P.m * a_cmd(3);
    eul_des = [phi_des; theta_des; 0];
end

%% ==================== FORMATION / GEOMETRY ===============================
function form = pick_formation(P, t, mode, env, p)
    if strcmp(mode,'gap')
        % Simple: choose narrower formation as you approach the gap
        x_min = env.gap_center_x - env.gap_span_x/2; 
        x_max = env.gap_center_x + env.gap_span_x/2;
        if p(1) < x_min - env.lookahead
            form = 'X';
        elseif p(1) < x_min
            form = 'Y';
        elseif p(1) <= x_max
            form = 'T';
        else
            form = 'H';
        end
    else
        % Timed cycling X->H->Y->T
        seg = floor(t / P.switch.Tseg);
        idx = mod(seg, numel(P.switch.order)) + 1;
        form = P.switch.order{idx};
    end
end

function geom = compute_geometry_planar(P, form)
% Planar morphing about z: define servo angles per formation (illustrative)
    switch form
        case 'X', ang = deg2rad([ 45, 135, -135, -45]);
        case 'H', ang = deg2rad([  0, 180,   0, 180]);
        case 'Y', ang = deg2rad([ 60, 180, -60, 180]);
        case 'T', ang = deg2rad([ 90, 180,  90, 180]);
        otherwise, ang = deg2rad([45,135,-135,-45]);
    end
    rHxy = [-P.w, P.l;  P.w, P.l;  P.w,-P.l;  -P.w,-P.l];
    rc2  = zeros(4,2);
    for i = 1:4
        dir = [cos(ang(i)), sin(ang(i))];
        rc2(i,:) = rHxy(i,:) + P.alpha*dir; % motor location in XY (z=0)
    end
    width_y = max(rc2(:,2)) - min(rc2(:,2));
    width_x = max(rc2(:,1)) - min(rc2(:,1));
    geom = struct('rc2',rc2,'width_x',width_x,'width_y',width_y);
end

function A = allocation_matrix_planar(P, geom)
    rc = geom.rc2; % [x,y]
    A = [ P.b,     P.b,     P.b,     P.b;       % T
           rc(1,2), rc(2,2), rc(3,2), rc(4,2); % tau_x
          -rc(1,1),-rc(2,1),-rc(3,1),-rc(4,1); % tau_y
          -P.kappa, P.kappa,-P.kappa, P.kappa];% tau_z
end

%% ==================== ATTITUDE MODEL & MPC PREP ==========================
function M = build_att_model_and_mpc(P, I)
% Continuous linearized model around hover, small angles
% States z = [phi theta psi p q r tau_x tau_y tau_z] (9)
% Inputs u = desired torques u_d = [u_x u_y u_z] (normalized domain later)
    nx = 9; nu = 3; ny = 3;
    % Continuous A_c, B_c
    Ac = zeros(nx,nx);
    % euler_dot ≈ [p; q; r]
    Ac(1,4)=1; Ac(2,5)=1; Ac(3,6)=1;
    % omega_dot = I^{-1} * tau
    Iinv = inv(I);
    Ac(4,7)=Iinv(1,1); Ac(5,8)=Iinv(2,2); Ac(6,9)=Iinv(3,3);
    % tau_dot = (u_d - tau)/tau_alpha
    Ac(7,7) = -1/P.tau_alpha; Ac(8,8) = -1/P.tau_alpha; Ac(9,9) = -1/P.tau_alpha;
    Bc = zeros(nx,nu);
    Bc(7,1) = 1/P.tau_alpha; Bc(8,2) = 1/P.tau_alpha; Bc(9,3) = 1/P.tau_alpha;
    Cc = [eye(3), zeros(3,6)]; Dc = zeros(3,nu);

    % Discretize (forward Euler for Ts=0.01)
    Ad = eye(nx) + P.Ts*Ac; Bd = P.Ts*Bc; Cd = Cc; Dd = Dc;

    % Lifted prediction matrices for outputs Y = Hy*ΔU + Py*x0
    [Sx,Su] = lift_state_mats(Ad,Bd,P.Np,P.Nc);     % Sx: (Np*nx x nx), Su: (Np*nx x Nc*nu)
    Hy = kron(eye(P.Np), Cd) * Su;
    Py = kron(eye(P.Np), Cd) * Sx;

    % Cost: (Y-R)^T Qbar (Y-R) + U^T Rbar U with U = TΔΔU + (1⊗I)u_prev
    Qbar = kron(eye(P.Np), P.Qx(1:3,1:3)); % penalize output angle error (phi,theta,psi)
    Rbar = kron(eye(P.Nc), P.Ru);

    % Map ΔU -> U stack: U = TΔ * ΔU + (1⊗I) * u_prev
    TDelta = kron(tril(ones(P.Nc)), eye(nu));  % (Nc*nu x Nc*nu)

    % Build Hessian for ΔU
    H1 = Hy; E = 2*(H1' * Qbar * H1 + TDelta' * Rbar * TDelta);

    % Constraints for Δu and u (normalized domain)
    [CC, dd, D_uprev] = build_constraints(P, nu);

    M = struct('Ad',Ad,'Bd',Bd,'Cd',Cd,'Dd',Dd,'Sx',Sx,'Su',Su, ...
               'Hy',Hy,'Py',Py,'Qbar',Qbar,'Rbar',Rbar,'TDelta',TDelta, ...
               'E',E,'CC',CC,'dd',dd,'D_uprev',D_uprev, 'u_max', P.u_max, ...
               'nx',nx,'nu',nu,'ny',ny);
end

function [Sx,Su] = lift_state_mats(Ad,Bd,Np,Nc)
    [nx,nu] = size(Bd);
    Sx = zeros(Np*nx, nx); Su = zeros(Np*nx, Nc*nu);
    A_pow = eye(nx);
    for i=1:Np
        A_pow = Ad*A_pow;                 % A^i
        Sx((i-1)*nx+1:i*nx, :) = A_pow;
        for j=1:min(i,Nc)
            Aij = Ad^(i-j);
            Su((i-1)*nx+1:i*nx, (j-1)*nu+1:j*nu) = Aij*Bd;
        end
    end
end

function [CC, dd, D_uprev] = build_constraints(P, nu)
% Build inequality CC*ΔU <= d + D_uprev * u_prev
% Rate constraints: |Δu_k| <= du_max
% Absolute constraints: |u_k| <= u_max, with U = TΔ ΔU + (1⊗I) u_prev
    Nc = P.Nc;
    % Rate constraints
    I_kron = eye(Nc*nu);
    CC_rate = [ I_kron; -I_kron];
    dd_rate = [ repmat(P.du_max, Nc, 1); repmat(P.du_max, Nc, 1)];
    D_rate  = zeros(size(CC_rate,1), nu);

    % Absolute u constraints
    TDelta = kron(tril(ones(Nc)), eye(nu)); % (Nc*nu x Nc*nu)
    OneKronI = kron(ones(Nc,1), eye(nu));   % (Nc*nu x nu)
    CC_abs = [ TDelta; -TDelta ];
    dd_abs = [ repmat(P.u_max, Nc, 1); repmat(P.u_max, Nc, 1) ];
    D_abs  = [ -OneKronI; +OneKronI ];

    % Stack
    CC = [CC_rate; CC_abs];
    dd = [dd_rate; dd_abs];
    D_uprev = [D_rate; D_abs];
end

function [du, u_cmd] = mpc_solve_once(M, z_meas, yref, u_prev)
% Single MPC step: decision ΔU (Nc*nu), apply first du, update u_prev externally
    % Prediction origin state
    x0 = z_meas;  % treat measured attitude state as prediction start

    % Reference stack (3*Np x 1)
    Np = size(M.Hy,1) / 3;            % number of output steps
    r_stack = repmat(yref, Np, 1);

    % Predicted output from state only
    Yx = M.Py * x0;                 % (3*Np x 1)
    y_err = Yx - r_stack;           % tracking error

    % Gradient terms
    F1 = 2 * (M.Hy' * M.Qbar * y_err);

    % Absolute input part: U = TΔ*ΔU + (1⊗I) u_prev
    Nc = size(M.TDelta,1) / 3;      % since nu=3
    OneKronI = kron(ones(Nc,1), eye(3));
    F2 = 2 * (M.TDelta' * M.Rbar * OneKronI * u_prev);

    F = F1 + F2;

    % Constraints: CC*ΔU <= d + D_uprev*u_prev
    d = M.dd + M.D_uprev * u_prev;

    % Solve QP: E*ΔU + F subject to CC*ΔU <= d
    DeltaU = QPhild(M.E, F, M.CC, d);

    % Extract first move
    du = DeltaU(1:3);
    u_cmd = u_prev + du;
    % Safety clamp (redundant with constraints)
    for i=1:3
        u_cmd(i) = max(min(u_cmd(i), M.u_max(i)), -M.u_max(i));
    end
end

%% ==================== DYNAMICS (PLANAR GEOMETRY) ========================
function [xdot, x_next] = dynamics_step_planar(P, x, f, geom, form)
% x = [p(3); v(3); eul(3); omg(3); tau(3)]
    p   = x(1:3); 
    v   = x(4:6); 
    eul = x(7:9); 
    omg = x(10:12); 
    tau = x(13:15);

    % Rotation & force (total thrust along body +z)
    R = eul2rotm_ZYX(eul);
    T = sum(f);
    Fw = R*[0;0;T];

    % Translational dynamics
    pdot = v;
    vdot = (1/P.m)*Fw + [0;0;-P.g];

    % Rotational dynamics (formation-dependent inertia)
    I = inertia_by_form(form);

    % Mixer torques from current geometry
    A = allocation_matrix_planar(P, geom);
    tau_mixed = A(2:4,:)*f;                  % torques from motor forces

    % Angular-rate dynamics
    omegadot = I \ (tau_mixed - cross(omg, I*omg));

    % 1st-order actuator (torque) dynamics
    taudot = (tau_mixed - tau)/P.tau_alpha;

    % Euler-angle kinematics
    euldot = euler_rates_ZYX(eul) * omg;

    xdot   = [pdot; vdot; euldot; omegadot; taudot];
    x_next = x + P.Ts*xdot;
end

function I = inertia_by_form(form)
    switch form
        case 'X', I = diag([0.004233, 0.004380, 0.007834]);
        case 'H', I = diag([0.005885, 0.001812, 0.006918]);
        case 'Y', I = diag([0.005042, 0.003096, 0.007369]);
        case 'T', I = diag([0.003654, 0.003917, 0.006792]);
        otherwise, I = diag([0.004233, 0.004380, 0.007834]);
    end
end

%% ==================== QP SOLVER & ALLOCATION ============================
function [f_cmd, info] = motor_qp(P, A, T_des, tau_des, f_prev)
    w = [T_des; tau_des(:)];
    H = (A.'*A) + 1e-6*eye(4); q = -(A.'*w) - 1e-6*f_prev;
    lb = max(P.f_min*ones(4,1), f_prev - P.df_max);
    ub = min(P.f_max*ones(4,1), f_prev + P.df_max);
    info = struct('exitflag',1,'solver','pgd');
    % Projected gradient descent
    f = f_prev; L = eigmax_pd(H) + 1e-9; alpha = 1.0/L;
    for it = 1:200
        g = H*f + q; f = f - alpha*g; f = min(max(f, lb), ub);
    end
    f_cmd = f;
end

function L = eigmax_pd(H)
    try, L = eigs(H,1,'LM'); catch, ev = eig(H); L = max(ev); end
end

function U = QPhild(E,F,CC,d)
% Hildreth-like dual ascent for small QPs: min 0.5 x'E x + F'x s.t. CC x <= d
    x = -E\F; % unconstrained solution
    if all(CC*x <= d + 1e-9), U = x; return; end
    T = CC*(E\CC'); K = CC*(E\F) + d;
    lam = zeros(size(CC,1),1);
    for it=1:200
        lam_old = lam;
        for i=1:size(CC,1)
            w = T(i,:)*lam - T(i,i)*lam(i) + K(i);
            lam(i) = max(0, -w / T(i,i));
        end
        if (lam - lam_old)'*(lam - lam_old) < 1e-8, break; end
    end
    U = -E\(F + CC'*lam);
end

%% ==================== UTILS: ROTM, EULER RATES, LOG, PLOTS ===============
function M = eul2rotm_ZYX(eul)
    phi=eul(1); th=eul(2); psi=eul(3);
    c1=cos(psi); s1=sin(psi); c2=cos(th); s2=sin(th); c3=cos(phi); s3=sin(phi);
    M=[ c1*c2, c1*s2*s3 - s1*c3, c1*s2*c3 + s1*s3; ...
        s1*c2, s1*s2*s3 + c1*c3, s1*s2*c3 - c1*s3; ...
        -s2,   c2*s3,            c2*c3];
end

function T = euler_rates_ZYX(eul)
    phi=eul(1); th=eul(2); c=cos(phi); s=sin(phi); t=tan(th); sec=1/cos(th);
    T=[1, s*t, c*t; 0, c, -s; 0, s*sec, c*sec]; T=inv(T);
end

function log = alloc_log(N)
    log.t=zeros(N,1); log.p=zeros(N,3); log.v=zeros(N,3);
    log.eul=zeros(N,3); log.omg=zeros(N,3); log.tau=zeros(N,3);
    log.u_norm=zeros(N,3); log.T_des=zeros(N,1); log.form= strings(N,1);
    log.f=zeros(N,4); log.width_y=zeros(N,1); log.width_x=zeros(N,1);
    log.eul_ref=zeros(N,3);
end

function log = push_log_switching(log,k,t,x,yref,T_des,u_cmd,tau_cmd,f,form,geom,P,infoQP) %#ok<INUSD>
    log.t(k)=t; log.p(k,:)=x(1:3).'; log.v(k,:)=x(4:6).'; log.eul(k,:)=x(7:9).';
    log.omg(k,:)=x(10:12).'; log.tau(k,:)=x(13:15).'; log.f(k,:)=f.';
    log.u_norm(k,:)=u_cmd.'; log.T_des(k)=T_des; log.form(k)=string(form);
    log.width_y(k)=geom.width_y; log.width_x(k)=geom.width_x; log.eul_ref(k,:)=yref.';
end

function plot_tracking_reports(P, log, env) %#ok<INUSD>
    t=log.t; 
    figure('Name','(A) Tracking — Trajectory performance (switching MPC)');
    subplot(3,1,1); plot(t, log.p(:,1),'LineWidth',1.5); grid on; ylabel('x [m]');
    yyaxis right; plot(t, log.width_y,'-','LineWidth',1.0); ylabel('width_y [m]');
    subplot(3,1,2); plot(t, log.p(:,2),'LineWidth',1.5); grid on; ylabel('y [m]');
    subplot(3,1,3); plot(t, log.p(:,3),'LineWidth',1.5); grid on; ylabel('z [m]'); xlabel('t [s]');

    figure('Name','(A) Switching MPC — Attitude tracking');
    plot(t, log.eul(:,1:2),'LineWidth',1.5); hold on; plot(t, log.eul_ref(:,1:2),'--','LineWidth',1.2); grid on;
    xlabel('t [s]'); ylabel('\phi, \theta [rad]'); legend('\phi','\theta','\phi_d','\theta_d');

    figure('Name','(A) Motor commands');
    plot(t, log.f,'LineWidth',1.2); grid on; xlabel('t [s]'); ylabel('f_i [N]'); legend('f1','f2','f3','f4');

    figure('Name','(A) Formation timeline');
    stairs(t, categorical(log.form)); grid on; ylabel('formation'); xlabel('t [s]');
end

function plot_hold_reports(P, log)
    t=log.t;
    figure('Name','(B) Hold — Position response (switching MPC)');
    subplot(3,1,1); plot(t, log.p(:,1),'LineWidth',1.5); grid on; ylabel('x [m]'); title('Position hold while switching formations');
    subplot(3,1,2); plot(t, log.p(:,2),'LineWidth',1.5); grid on; ylabel('y [m]');
    subplot(3,1,3); plot(t, log.p(:,3),'LineWidth',1.5); grid on; ylabel('z [m]'); xlabel('t [s]');

    figure('Name','(B) Hold — Attitude & refs');
    plot(t, log.eul(:,1:2),'LineWidth',1.2); hold on; plot(t, log.eul_ref(:,1:2),'--','LineWidth',1.2); grid on;
    xlabel('t [s]'); ylabel('[rad]'); legend('\phi','\theta','\phi_d','\theta_d');

    figure('Name','(B) Hold — Motor commands');
    plot(t, log.f,'LineWidth',1.2); grid on; xlabel('t [s]'); ylabel('f_i [N]'); legend('f1','f2','f3','f4');
end

function plot_stability_thrust_dashboard(P, log)
    t=log.t; T_total = sum(log.f,2)*P.b; T2W = T_total/(P.m*P.g);
    df = [zeros(1,4); diff(log.f)];
    figure('Name','Stability & Thrust Dashboard');
    subplot(2,2,1); plot(t, T_total,'LineWidth',1.5); hold on; yline(P.m*P.g,'--k','weight'); grid on; ylabel('Total thrust [N]'); title('Total Thrust vs Weight');
    subplot(2,2,2); plot(t, T2W,'LineWidth',1.5); grid on; ylabel('T/W'); title('Thrust-to-Weight');
    subplot(2,2,3); plot(t, df,'LineWidth',1.0); grid on; ylabel('Δf [N/step]'); xlabel('t [s]'); title('Motor rate');
    subplot(2,2,4); plot(t, log.u_norm,'LineWidth',1.0); grid on; ylabel('u_{norm}'); xlabel('t [s]'); title('MPC input (normalized)');
end

function print_metrics(P, log, tag)
    rmse = sqrt(mean((log.eul(:,1:2) - log.eul_ref(:,1:2)).^2, 1));
    fprintf('\n[%s] RMSE [phi theta] = [%.4f %.4f] rad\n', tag, rmse(1), rmse(2));
    fprintf('[%s] avg T/W = %.2f\n', tag, mean(sum(log.f,2)*P.b/(P.m*P.g)));
end

function save_all_plots(outdir)
    if ~exist(outdir,'dir'), mkdir(outdir); end
    figs = sort(findall(0, 'Type', 'figure'));
    if isempty(figs), warning('No figures to save.'); return; end
    for i = 1:numel(figs)
        f = figs(i);
        nm = get(f,'Name'); if isempty(nm), nm = sprintf('Figure_%d', f.Number); end
        nm = regexprep(char(nm),'[^a-zA-Z0-9_-]','_');
        set(f,'PaperPositionMode','auto');
        try
            exportgraphics(f, fullfile(outdir, [nm '.png']), 'Resolution', 200);
            exportgraphics(f, fullfile(outdir, [nm '.jpg']), 'Resolution', 200);
        catch
            print(f, fullfile(outdir, nm), '-dpng', '-r200');
            print(f, fullfile(outdir, nm), '-djpeg', '-r200');
        end
    end
    pdfPath = fullfile(outdir, 'All_Figures.pdf');
    try
        if exist(pdfPath,'file'), delete(pdfPath); end
        for i = 1:numel(figs)
            exportgraphics(figs(i), pdfPath, 'ContentType','vector', 'Append', true);
        end
    catch
        for i = 1:numel(figs)
            f = figs(i);
            nm = get(f,'Name'); if isempty(nm), nm = sprintf('Figure_%d', f.Number); end
            nm = regexprep(char(nm),'[^a-zA-Z0-9_-]','_');
            print(f, fullfile(outdir, [nm '.pdf']), '-dpdf', '-painters');
        end
    end
end
