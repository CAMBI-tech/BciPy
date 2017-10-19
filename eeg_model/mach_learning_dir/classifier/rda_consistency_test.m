clear all
%close all
clear classes

% Tests the results between the python implementation of regularized
% discriminant analysis and currently RSVPKeyboard implementation.
% mse is the mean squared error of all possible parameters to be checked.

mod1 = py.importlib.import_module('function_classifier');
py.reload(mod1);
mod = py.importlib.import_module('demo');
py.reload(mod);

num_experiments = 100;
mse = 0;

% Iterates over sample random experiments
% TODO: Update the procedure to keep experiments statistically independent
% of each other. Right now it is just duplication of the same experiment
for exper = 1:num_experiments
        
    dim_x = randi([150, 200]);
    num_x_p = randi([75, 125]);
    num_x_n = randi([750, 1250]);
    
    xp = randi([1, 15]) + randi([1, 10]) * randn(num_x_p, dim_x);
    xn = randi([1, 15]) + randi([1, 10]) * randn(num_x_n, dim_x);
    x = [xp; xn];
    
    y = [ones( num_x_p,1) ; zeros(num_x_n,1)];
    
    % Get all possible outputs
    tic;
    out_py = py.demo.test_rda(toggleNumpy(x),toggleNumpy(y),toggleNumpy(x));
    time_py(exper) = toc;
    
    tic;
    rdaRSVP = rda();
    rdaRSVP.learn(x.',y.');
    time_rsvp(exper) = toc;
    
    out_rsvp = rdaRSVP.getInvCovariances();
    rsvp_inv_covar = zeros(size(out_rsvp{1},1),size(out_rsvp{1},2),length(out_rsvp));
    for i = 1:length(out_rsvp)
        rsvp_inv_covar(:,:,i) = out_rsvp{i};
    end
    
    out_rsvp = rdaRSVP.getMeans();
    rsvp_means = zeros(size(out_rsvp{1},1),size(out_rsvp{1},2),length(out_rsvp));
    for i = 1:length(out_rsvp)
        rsvp_means(:,:,i) = out_rsvp{i};
    end
    rsvp_means = squeeze(rsvp_means).';
    
    out_rsvp = rdaRSVP.getCovariances();
    rsvp_covs = zeros(size(out_rsvp{1},1),size(out_rsvp{1},2),length(out_rsvp));
    for i = 1:length(out_rsvp)
        rsvp_covs(:,:,i) = out_rsvp{i};
    end
    
    py_cov = toggleNumpy(out_py{1});
    py_inv_covar = toggleNumpy(out_py{2});
    py_means = toggleNumpy(out_py{3});
    
    MSE_means = sum(sum(py_means - rsvp_means).^2);
    MSE_inv_cov = sum(sum(sum(py_inv_covar - rsvp_inv_covar).^2));
    
    % Proba Difference
    rsvp_prob = rdaRSVP.operate(x.');
    py_prob = toggleNumpy(out_py{4});
    
    
    MSE = MSE_means + MSE_inv_cov;
    mse = mse + MSE;
    
end

disp(strcat('Time py :',num2str(mean(time_py))))
disp(strcat('Time rsvp :',num2str(mean(time_rsvp))))
disp(strcat('MSE :',num2str(mse)))

