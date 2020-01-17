% Use this script to get the A and B matrices from the continous transer
% function:

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

v = 5; % m/s

% Choosing the state as x = [phi, b*h*dphi - a*v*delta], results in the following state
% space model, equivalent to G = (a*v*s + v^2)/(b*h*s^2 - b*g), apart from
% that not just phi is measured, but also the b*h*dphi-a*v*delta state element:
Ac = [0 1/(b*h); b*g 0];
Bc = [a*v/(b*h); v^2];
% Measure both states. If Cc would have been [1 0] (i.e. if we only
% measured phi instead of both states), then the plant would be equivalent
%to phi(s) = (a*v*s + v^2)/(b*h*s^2 - b*g) * delta(s):
Cc = eye(2);
plant = ss(Ac, Bc, Cc, []);
plant = c2d(plant, Ts); % Discretize model
A = plant.A; B = plant.B; C = plant.C;

M = [1 0]; % phi = M*x
Q_phi = 1e1; 
Q = M'*Q_phi*M; % i.e. only care about deviations from the phi state
R = 1;
[K,P,~] = lqr(plant,Q,R); % Performs DARE and returns gain K
% Pre-filter for reference for discrete system. Designed only to pre-filter
% a reference for the first state element phi, and not the second state element:
Kr = pinv(M*inv(eye(2) + B*K - A)*B); 

cov_w = 0*1e-5; % process noise covariance
% Despite not using a kalman filter for this particular setup, one can experiment
% with measurement noise if one wants to, as long as the noises for each state
% is considered decorelated (otherwise you have to modify the simulink file):
cov_v = 0 * 1e-5*eye(2);

phi_0 = deg2rad(0.5); % Initial roll angle
x_0 = [phi_0; 0]; % Initial state
ref_phi = deg2rad(0); % Reference for roll angle

