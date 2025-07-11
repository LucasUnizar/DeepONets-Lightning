%% Main Script for Reaction-Diffusion Dataset Generation
% This script generates training data for a reaction-diffusion PDE learning task.
% It creates pairs of forcing functions (inputs) and corresponding PDE solutions (outputs).
%
% Dataset structure:
% - u_train: Input forcing functions (Gaussian processes)
% - y_train: Output sensor locations (x,t coordinates)
% - s_train: Output sensor values (PDE solutions)
%
% Author: Lucas Tesan

clear; close all; clc;

%% ================ Parameters Setup ================
% Physical parameters
length_scale = 0.2; % Length scale for Gaussian Process kernel
Nx = 100;           % Spatial resolution (number of grid points in x)
Nt = 100;           % Temporal resolution (number of time steps)
m = Nx;             % Number of input sensors (same as spatial resolution)

% Dataset parameters
N_train = 500;     % Number of training samples
P_train = 50;      % Number of output sensor points per sample

% Domain definition
xmin = 0; xmax = 1; % Spatial domain [0,1]
tmin = 0; tmax = 1; % Temporal domain [0,1]
x = linspace(xmin, xmax, Nx)'; % Spatial grid
t = linspace(tmin, tmax, Nt)'; % Temporal grid

%% ================ Data Initialization ================
% Initialize arrays for training data:
% - u_train: Input forcing functions (N_train*P_train x m)
% - y_train: Output sensor locations (N_train*P_train x 2) [x,t coordinates]
% - s_train: Output sensor values (N_train*P_train x 1)
u_train = zeros(N_train*P_train, m);
y_train = zeros(N_train*P_train, 2);
s_train = zeros(N_train*P_train, 1);

% Create figure for visualization
fig = figure('Position', [100, 100, 1200, 800], ...
    'Name', 'Reaction-Diffusion Data Generation Progress');

%% ================ Data Generation Loop ================
for n = 1:N_train
    % 1. Generate random forcing function using Gaussian Process with RBF kernel
    f = generate_gp_sample(x, length_scale);
    
    % 2. Solve the reaction-diffusion equation with this forcing function
    UU = solve_ADR(x, t, f);
    
    % 3. Create input-output pairs for this sample:
    %    Input: The forcing function f(x) evaluated at all spatial points
    u = f; % m x 1 vector
    
    % 4. Randomly sample P output sensor locations
    x_idx = randi(Nx, P_train, 1); % Random x indices
    t_idx = randi(Nt, P_train, 1); % Random t indices
    
    % Calculate array indices for this sample
    start_idx = (n-1)*P_train + 1;
    end_idx = n*P_train;
    
    % 5. Store training data:
    %    - Repeat input function P_train times
    u_train(start_idx:end_idx, :) = repmat(u', P_train, 1);
    
    %    - Store output sensor locations (x,t coordinates)
    y_train(start_idx:end_idx, 1) = x(x_idx);
    y_train(start_idx:end_idx, 2) = t(t_idx);
    
    %    - Store solution values at sampled points
    s_train(start_idx:end_idx) = arrayfun(@(xi,ti) UU(xi,ti), x_idx, t_idx);
    
    % 6. Visualization (plot every 100 samples)
    if mod(n, 100) == 1 || n == N_train
        plot_sample_data(fig, x, t, f, UU, ...
            y_train(start_idx:end_idx,:), s_train(start_idx:end_idx), n, N_train);
        drawnow;
    end
end

%% ================ Save Dataset ================
% Create filename with dataset information
dataset_name = sprintf('reaction_diffusion_dataset_N%d_P%d_L%.2f_%dx%d.mat', ...
    N_train, P_train, length_scale, Nx, Nt);

% Save all variables to MAT file
save(dataset_name, 'u_train', 'y_train', 's_train', 'x', 't', 'length_scale');
fprintf('Dataset saved as: %s\n', dataset_name);

%% Helper Functions

function plot_sample_data(fig, x, t, f, UU, y_samples, s_samples, sample_num, total_samples)
    % Visualizes a generated sample from the dataset
    % Inputs:
    %   fig - figure handle
    %   x - spatial grid
    %   t - temporal grid
    %   f - forcing function
    %   UU - PDE solution matrix
    %   y_samples - output sensor locations
    %   s_samples - output sensor values
    %   sample_num - current sample number
    %   total_samples - total number of samples
    
    % Clear previous plots
    clf(fig);
    
    % Create meshgrid for surface plot
    [X, T] = meshgrid(x, t);
    
    % Plot 1: Forcing function f(x)
    subplot(2, 2, 1);
    plot(x, f, 'b', 'LineWidth', 1.5);
    title(sprintf('Sample %d/%d: Forcing Function f(x)', sample_num, total_samples));
    xlabel('x');
    ylabel('f(x)');
    grid on;
    
    % Plot 2: Solution surface u(x,t)
    subplot(2, 2, 2);
    surf(X, T, UU', 'EdgeColor', 'none');
    title('Solution Surface u(x,t)');
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
    
    % Plot 4: Time snapshots
    subplot(2, 2, 4);
    hold on;
    time_indices = round(linspace(1, length(t), 5)); % Show 5 time snapshots
    colors = lines(5);
    for i = 1:length(time_indices)
        ti = time_indices(i);
        plot(x, UU(:,ti), 'Color', colors(i,:), 'LineWidth', 1.5, ...
            'DisplayName', sprintf('t = %.2f', t(ti)));
    end
    hold off;
    title('Solution Snapshots at Different Times');
    xlabel('x');
    ylabel('u(x,t)');
    legend('Location', 'best');
    grid on;
    
    % Add overall title
    sgtitle('Reaction-Diffusion Dataset Generation Progress');
end

function f = generate_gp_sample(x, length_scale)
    % Generates a sample from a Gaussian Process with RBF kernel
    % Inputs:
    %   x - input points (spatial grid)
    %   length_scale - kernel length scale
    % Output:
    %   f - generated sample
    
    N = length(x);
    X = x(:); % Ensure column vector
    
    % RBF kernel matrix
    K = exp(-0.5 * (pdist2(X/length_scale, X/length_scale).^2));
    
    % Add small jitter for numerical stability
    jitter = 1e-10;
    K = K + jitter * eye(N);
    
    % Cholesky decomposition for sampling
    L = chol(K, 'lower');
    
    % Generate sample from N(0,I) and transform
    f = L * randn(N, 1);
end

function UU = solve_ADR(x, t, f)
    % Solves the 1D reaction-diffusion equation:
    %   u_t = (k(x) u_x)_x - v(x) u_x + g(u) + f(x)
    % with zero initial and boundary conditions
    %
    % Uses finite difference method with implicit time stepping
    %
    % Inputs:
    %   x - spatial grid
    %   t - temporal grid
    %   f - forcing function
    % Output:
    %   UU - solution matrix (Nx x Nt)
    
    Nx = length(x);
    Nt = length(t);
    h = x(2) - x(1);     % Spatial step size
    dt = t(2) - t(1);    % Temporal step size
    h2 = h^2;
    
    % PDE coefficients (can be modified for different problems)
    k = 0.01 * ones(Nx, 1);  % Diffusion coefficient
    v = zeros(Nx, 1);        % Advection coefficient (zero in this case)
    
    % Nonlinear reaction term (can be modified)
    g = @(u) 0.01 * u.^2;    % Nonlinear term
    dg = @(u) 0.02 * u;      % Derivative of nonlinear term
    
    % Initialize solution matrix
    UU = zeros(Nx, Nt);
    
    % Finite difference operators
    D1 = diag(ones(Nx-1,1), 1) - diag(ones(Nx-1,1), -1); % First derivative
    D2 = -2*eye(Nx) + diag(ones(Nx-1,1), 1) + diag(ones(Nx-1,1), -1); % Second derivative
    
    % Construct spatial operator matrix
    M = -diag(D1 * k) * D1 - 4 * diag(k) * D2;
    
    % Set up matrices for interior points (boundaries are fixed at zero)
    D3 = eye(Nx-2);
    m_bond = (8 * h2 / dt) * D3 + M(2:end-1, 2:end-1);
    
    % Advection term
    v_bond = 2 * h * diag(v(2:end-1)) * D1(2:end-1, 2:end-1) + ...
             2 * h * diag(v(3:end) - v(1:end-2));
    
    % Combined matrix
    mv_bond = m_bond + v_bond;
    c = (8 * h2 / dt) * D3 - M(2:end-1, 2:end-1) - v_bond;
    
    % Time stepping loop
    for i = 1:Nt-1
        % Evaluate nonlinear terms at current time step
        gi = g(UU(2:end-1, i));
        dgi = dg(UU(2:end-1, i));
        h2dgi = diag(4 * h2 * dgi);
        
        % Construct linear system
        A = mv_bond - h2dgi;
        b1 = 8 * h2 * (0.5 * f(2:end-1) + 0.5 * f(2:end-1) + gi);
        b2 = (c - h2dgi) * UU(2:end-1, i);
        
        % Solve for next time step
        UU(2:end-1, i+1) = A \ (b1 + b2);
    end
end