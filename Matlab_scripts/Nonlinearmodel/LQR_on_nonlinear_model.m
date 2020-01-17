clc, clearvars

r_wheel = 0.311; % Radius of the wheel [m]
h = 0.2085 + r_wheel;           % height of center of mass [m]
b = 1.095;            % length between wheel centers [m]
c = 0.06;             % length between front wheel contact point and the 
                           % extention of the fork axis [m]
lambda = deg2rad(70); % angle of the fork axis [deg]
a = 0.4964;          % distance from rear wheel to frame's center of mass [m]
IMUheight = 0.45;   % IMU height [m]
m = 44.2; % Mass of the Bike [kg]
g = 9.81;                  % gravity [m/s^2]
Ts = 0.04; % Sampling Time [s]
J = m*h^2; % Inertia [kg m^2]
D_inertia = m*a*h; % Inertia [kg m^2]

v = 2.1985; % m/s

Ac =[0, 0, 1
    0, 0, 0
    g/h, (sin(lambda)*(h*v^2 - a*c*g))/(b*h^2), 0];
Bc = [0
     1
 (a*v*sin(lambda))/(b*h)];
Cc = eye(3);


plant = ss(Ac, Bc, Cc, []);
plant = c2d(plant, Ts); % Discretize model
A = plant.A; B = plant.B; C = plant.C;

Q = 1*diag([10, 10, 10]);
R = 1;
[K,~,~] = lqr(plant,Q,R); % Performs DARE and returns gain K
% Pre-filter for reference for discrete system. Designed only to pre-filter
% a reference for the first state element phi, and not the second state element:
M = [1, 0, 0];
Kr = pinv(M*inv(eye(3) + B*K - A)*B); 


%% Simulate:

phi_0 = deg2rad(5); % Initial roll angle
x_0 = [phi_0; 0; 0]; % Initial state
ref_phi = deg2rad(0); % Reference for roll angle

num_steps = 100;
t = 0:Ts:num_steps*Ts;

x = zeros(3, num_steps + 1);
x(:, 1) = x_0;
xlin = zeros(3, num_steps + 1);
xlin(:, 1) = x_0;
u = zeros(1, num_steps);
ulin = zeros(1, num_steps);
for i = 1:num_steps
    u(i) = -K*x(:, i);
    [t_temp, new_x] = ode45(@(t, x) x_dot_fun(x, u(i), v), [0, Ts], x(:, i));
    x(:, i+1) = new_x(end, :)';
    ulin(i) = -K*xlin(:, i);
    xlin(:, i + 1) = A*xlin(:, i) + B*ulin(i);
end

figure('Position', [680   517   964   461])
subplot(2, 3, 1)
plot(t, rad2deg(x(1, :)));
title('$\varphi$', 'interpreter', 'latex')
subplot(2, 3, 2)
plot(t, rad2deg(x(2, :)))
title('$\delta$', 'interpreter', 'latex')
subplot(2, 3, 3)
plot(t(2:end), rad2deg(u))
title('$\dot{\delta}$', 'interpreter', 'latex')

subplot(2, 3, 4)
plot(t, rad2deg(xlin(1, :)));
title('$\varphi$ Lin', 'interpreter', 'latex')
subplot(2, 3, 5)
plot(t, rad2deg(xlin(2, :)))
title('$\delta$ Lin', 'interpreter', 'latex')
subplot(2, 3, 6)
plot(t(2:end), rad2deg(ulin))
title('$\dot{\delta}$ Lin', 'interpreter', 'latex')


