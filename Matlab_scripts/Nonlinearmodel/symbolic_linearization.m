clc, clearvars

% r_wheel = 0.311; % Radius of the wheel [m]
% h = 0.2085 + r_wheel;           % height of center of mass [m]
% b = 1.095;            % length between wheel centers [m]
% c = 0.06;             % length between front wheel contact point and the 
%                            % extention of the fork axis [m]
% lambda = deg2rad(70); % angle of the fork axis [deg]
% a = 0.4964;          % distance from rear wheel to frame's center of mass [m]
% IMUheight = 0.45;   % IMU height [m]
% m = 44.2; % Mass of the Bike [kg]
% g = 9.81;                  % gravity [m/s^2]
% Ts = 0.04; % Sampling Time [s]
% J = m*h^2; % Inertia [kg m^2]
% D_inertia = m*a*h; % Inertia [kg m^2]

syms phi delta phidot deltadot v real
syms h g a c b lambda real
x=[phi delta phidot]';
xdot1=phidot;
xdot2=deltadot;
xdot3=(g/h)*tan(phi)+((a*v)/(b*h))*sin(lambda)*tan(deltadot)+((h*v^2-g*a*c)/(b*h^2))*sin(lambda)*tan(delta);
f=[xdot1 xdot2 xdot3]';

Asym = simplify(jacobian(f, x));
Bsym = simplify(jacobian(f, deltadot));

matlabFunction(Asym, Bsym, 'file', 'state_matrices_fun', 'vars', {phi, delta, deltadot, v, h, g, a, c, b, lambda})

[A, B] = state_matrices_fun(0, 0, 0, v, h, g, a, c, b, lambda);
A
B