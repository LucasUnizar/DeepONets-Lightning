% Main Script for Burgers' Equation Dataset Generation
% This script generates training data for a Burgers' PDE learning task.
% It creates pairs of initial conditions (inputs) and corresponding PDE solutions (outputs).
%
% Dataset structure:
% - u_train: Input initial conditions (pure sine waves with varying amplitude)
% - y_train: Output sensor locations (x,t coordinates)
% - s_train: Output sensor values (PDE solutions)
%
% Boundary Conditions: u(-1,t) = 0, u(1,t) = 0
% Initial Condition: u(x,0) = A * sin(pi * x), where A is a random amplitude,
%                    OR Gaussian peaks modulated by sin(pi*x)
% No forcing term
%
% Solver: Fully Explicit Finite Difference Method (FTCS-like)
%
% Author: Lucas Tesan
% Date: June 10, 2025
clear; close all; clc;
%% ================ Parameters Setup ================
% Physical parameters
Nx = 256;           % Spatial resolution (number of grid points in x)
Nt_solver = 10000;  % Temporal resolution for the SOLVER (can be very large for stability)
m = Nx;             % Number of input sensors (same as spatial resolution)
nu = 0.025;         % Kinematic viscosity (diffusion coefficient) for Burgers' equation
% Dataset parameters
N_train = 5000;      % Number of training samples
P_train = 100;      % Number of output sensor points per sample
Nt_sample = 100;     % Desired number of time points for SAMPLING per solution.
                    % Total (x,t) output samples will be P_train. If P_train > Nx * Nt_sample,
                    % it will randomly select more (x,t) pairs from the available Nt_sample points.

% Domain definition
xmin = -1; xmax = 1; % Spatial domain [-1, 1]
tmin = 0; tmax = 1;   % Temporal domain [0,1]
x = linspace(xmin, xmax, Nx)'; % Spatial grid
t_solver = linspace(tmin, tmax, Nt_solver)'; % Temporal grid for the solver

% Grid for sampled time points (used for dataset and visualization)
% Ensure these sampled times are actual points from t_solver or representative
t_sampled_indices = round(linspace(1, Nt_solver, Nt_sample)); % Get Nt_sample indices evenly spaced
t_sampled = t_solver(t_sampled_indices); % The actual time points used for sampling

%% ================ Data Initialization ================
% Initialize arrays for training data:
% - u_train: Input initial conditions (N_train*P_train x m)
% - y_train: Output sensor locations (N_train*P_train x 2) [x,t coordinates]
% - s_train: Output sensor values (N_train*P_train x 1)
u_train = zeros(N_train*P_train, m);
y_train = zeros(N_train*P_train, 2);
s_train = zeros(N_train*P_train, 1);
% Create figure for visualization
fig = figure('Position', [100, 100, 1200, 800], ...
    'Name', 'Burgers'' Equation Data Generation Progress (Explicit FDM)');
%% ================ Data Generation Loop ================
for n = 1:N_train
    % 1. Generate random initial condition u(x,0) as a Gaussian peak
    %    modulated by a sine wave for boundary conditions.
    u0 = generate_gaussian_IC(x);
    
    % 2. Solve the Burgers' equation with this initial condition using the solver's Nt
    UU_full = solve_Burgers_Explicit(x, t_solver, u0, nu);
    
    % Check for NaN/Inf in solution, which indicates divergence
    if any(isnan(UU_full(:))) || any(isinf(UU_full(:)))
        fprintf('Sample %d/%d: Solver diverged! Skipping this sample.\n', n, N_train);
        n = n - 1; % Re-do this sample if it diverged
        continue;
    end
    
    % Subsample the full solution to get data only at the desired Nt_sample time points
    UU_sampled = UU_full(:, t_sampled_indices); % Nx x Nt_sample
    
    % 3. Create input-output pairs for this sample:
    %    Input: The initial condition u(x,0) evaluated at all spatial points
    u_input_ic = u0; % m x 1 vector
    
    % 4. Randomly sample P output sensor locations from the *sampled* solution
    x_idx = randi(Nx, P_train, 1); % Random x indices from the Nx spatial points
    t_idx_sampled = randi(Nt_sample, P_train, 1); % Random t indices from the Nt_sample *sampled* time points
    
    % Calculate array indices for this sample
    start_idx = (n-1)*P_train + 1;
    end_idx = n*P_train;
    
    % 5. Store training data:
    %    - Repeat input initial condition P_train times
    u_train(start_idx:end_idx, :) = repmat(u_input_ic', P_train, 1);
    
    %    - Store output sensor locations (x,t coordinates) using the sampled times
    y_train(start_idx:end_idx, 1) = x(x_idx);
    y_train(start_idx:end_idx, 2) = t_sampled(t_idx_sampled); % Use the *sampled* time values
    
    %    - Store solution values at sampled points from UU_sampled
    s_train(start_idx:end_idx) = arrayfun(@(xi,ti) UU_sampled(xi,ti), x_idx, t_idx_sampled);
    
    % 6. Visualization (plot every 100 samples)
    if mod(n, 100) == 1 || n == N_train
        plot_sample_data(fig, x, t_sampled, u_input_ic, UU_sampled, ...
            y_train(start_idx:end_idx,:), s_train(start_idx:end_idx), n, N_train);
        drawnow;
    end
end
%% ================ Save Dataset ================
% Create filename with dataset information
dataset_name = sprintf('burgers_dataset_IC_gaussian_N%d_P%d_nu%.4f_%dx%d_sampled%d.mat', ...
    N_train, P_train, nu, Nx, Nt_solver, Nt_sample);
% Save all variables to MAT file
save(dataset_name, 'u_train', 'y_train', 's_train', 'x', 't_solver', 't_sampled', 'nu', 'Nt_sample');
fprintf('Dataset saved as: %s\n', dataset_name);
%% Helper Functions (plot_sample_data, solve_Burgers_Explicit, generate_gaussian_IC)

function u0 = generate_gaussian_IC(x)
    % Generates an initial condition as a Gaussian peak modulated by sin(pi*x)
    % to ensure Dirichlet boundary conditions u(-1)=0, u(1)=0.
    %
    % Inputs:
    %   x - spatial grid
    % Output:
    %   u0 - initial condition vector
    
    % Random parameters for the Gaussian
    overall_amplitude_scale = 0.5 + 1.0 * rand(); % Overall amplitude scale (0.5 to 1.5)
    mu = -0.7 + 1.4 * rand(); % Peak center between -0.7 and 0.7 (avoiding very close to boundaries)
    sigma = 0.1 + 0.2 * rand(); % Width of the peak (standard deviation, 0.1 to 0.3)
    
    % Generate the Gaussian profile
    gaussian_profile = exp(-((x - mu).^2) / (2 * sigma^2));
    
    % Modulate by sin(pi*x) to ensure boundary conditions
    u0 = overall_amplitude_scale * sin(pi * x) .* gaussian_profile; 

    % Optional: Add a random sign to amplitude
    if rand() > 0.5
        u0 = -u0;
    end

    % Ensure it's not all zeros (e.g. if amplitude_scale was zero)
    if max(abs(u0)) < 1e-6
        u0 = overall_amplitude_scale * sin(pi * x); % Fallback to a simple sine wave if it became negligible
    end
end

function plot_sample_data(fig, x, t, u0, UU, y_samples, s_samples, sample_num, total_samples)
    % Visualizes a generated sample from the dataset
    % Inputs:
    %   fig - figure handle
    %   x - spatial grid
    %   t - temporal grid (this is now the *sampled* temporal grid)
    %   u0 - initial condition
    %   UU - PDE solution matrix (this is now the *sampled* solution matrix)
    %   y_samples - output sensor locations
    %   s_samples - output sensor values
    %   sample_num - current sample number
    %   total_samples - total number of samples
    
    % Clear previous plots
    clf(fig);
    
    % Create meshgrid for surface plot
    [X, T] = meshgrid(x, t);
    
    % Plot 1: Initial Condition u(x,0)
    subplot(2, 2, 1);
    plot(x, u0, 'b', 'LineWidth', 1.5);
    title(sprintf('Sample %d/%d: Initial Condition u(x,0)', sample_num, total_samples));
    xlabel('x');
    ylabel('u(x,0)');
    grid on;
    
    % Plot 2: Solution surface u(x,t) - using the sampled UU and t
    subplot(2, 2, 2);
    surf(X, T, UU', 'EdgeColor', 'none');
    title('Solution Surface u(x,t) (Sampled Times)');
    xlabel('x');
    ylabel('t');
    zlabel('u(x,t)');
    view(30, 45);
    colorbar;
    
    % Plot 3: Selected sample points on solution
    subplot(2, 2, 3);
    scatter3(y_samples(:,1), y_samples(:,2), s_samples, 40, s_samples, 'filled');
    title('Sampled Points (x,t,u)');
    xlabel('x');
    ylabel('t');
    zlabel('u');
    view(30, 45);
    colorbar;
    
    % Plot 4: Time snapshots - using the sampled UU and t
    subplot(2, 2, 4);
    hold on;
    time_indices = round(linspace(1, length(t), 5)); % Show 5 time snapshots from the sampled times
    colors = lines(5);
    for i = 1:length(time_indices)
        ti = time_indices(i);
        plot(x, UU(:,ti), 'Color', colors(i,:), 'LineWidth', 1.5, ...
            'DisplayName', sprintf('t = %.2f', t(ti)));
    end
    hold off;
    title('Solution Snapshots at Different Times (Sampled)');
    xlabel('x');
    ylabel('u(x,t)');
    legend('Location', 'best');
    grid on;
    
    % Add overall title
    sgtitle('Burgers'' Equation Dataset Generation Progress');
end

function UU = solve_Burgers_Explicit(x, t, u0, nu)
    % Solves the 1D Burgers' equation explicitly using FTCS-like scheme:
    %   u_t + u u_x = nu u_xx
    % with initial condition u(x,0) = u0(x) and Dirichlet boundary conditions
    % u(-1,t) = 0, u(1,t) = 0.
    %
    % WARNING: This explicit scheme is highly conditionally stable and
    % likely to diverge without extremely small dt.
    %
    % Inputs:
    %   x - spatial grid (should be linspace from -1 to 1)
    %   t - temporal grid (this is the *solver's* temporal grid)
    %   u0 - initial condition (vector of size Nx)
    %   nu - kinematic viscosity
    % Output:
    %   UU - solution matrix (Nx x Nt)
    
    Nx = length(x);
    Nt = length(t);
    h = x(2) - x(1);     % Spatial step size
    dt = t(2) - t(1);    % Temporal step size
    
    % Initialize solution matrix
    UU = zeros(Nx, Nt);
    UU(:, 1) = u0; % Set the initial condition
    
    % Loop through time steps
    for n = 1:Nt-1
        % Get solution at current time step
        u_n = UU(:, n);
        
        % Create a temporary array for the next time step (u^(n+1))
        u_nplus1 = zeros(Nx, 1);
        
        % Enforce boundary conditions directly for the next time step
        u_nplus1(1) = 0; % u(-1, t) = 0
        u_nplus1(Nx) = 0; % u(1, t) = 0
        
        % Apply explicit finite difference updates for interior points (from 2 to Nx-1)
        for j = 2:Nx-1
            % Advection term: u * u_x approx u_n * (u_{j+1}^n - u_{j-1}^n) / (2h)
            advection_term = u_n(j) * (u_n(j+1) - u_n(j-1)) / (2*h);
            
            % Diffusion term: nu * u_xx approx nu * (u_{j+1}^n - 2u_j^n + u_{j-1}^n) / h^2
            diffusion_term = nu * (u_n(j+1) - 2*u_n(j) + u_n(j-1)) / (h^2);
            
            % Update for next time step: u_j^(n+1) = u_j^n + dt * (-advection_term + diffusion_term)
            u_nplus1(j) = u_n(j) + dt * (-advection_term + diffusion_term);
        end
        
        % Store the updated solution
        UU(:, n+1) = u_nplus1;
    end
end