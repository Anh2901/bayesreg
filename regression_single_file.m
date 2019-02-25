% #########################################################################
% parameters which users need to specify

% location of the csv file, default ''
loc = 'C:\Users\USer\Desktop\WinterScholar_SportAnalytics\Data set\Regression\Yatch\yacht_hydrodynamics.csv';
% add path
addpath 'C:\Users\USer\Desktop\WinterScholar_SportAnalytics\Data set\bayesreg_matlabcentral_1.8';
% output direction to save the results
output_dir = 'C:\Users\USer\Desktop\WinterScholar_SportAnalytics\Experiments';

% the number of iterations to repeat, default 10
iter = 10;

% the number of folds to split for cross-validation, default 10
num_folds = 10;

% whether or not performing truncated power spline, default false
nonlinear = false;

% if performing truncated power spline, how many knots to use, default 4
num_knots = 4;

% specify the model to be used in bayesreg, default gaussian
%model = 'gaussian';

% specify the prior to be used in bayesreg, default lasso
% prior = 'normal';

% #########################################################################
% the main part of the program

% read the file into matlab as a numerical matrix
df = csvread(loc,1);

% if needs to perform truncated power spline
if nonlinear
    df = to_nonlinear(df);
end

variables = df(:, 1:end-1);
% include interaction term 
D = x2fx(variables,'interaction');
% remove constant
D(:,1) = [];

% declare bayesian mean square error and negative log likelihood
bayes_lasso_mspe = zeros(iter*num_folds, 1);
bayes_lasso_neglike = zeros(iter*num_folds, 1);

bayes_horseshoe_mspe = zeros(iter*num_folds, 1);
bayes_horseshoe_neglike = zeros(iter*num_folds, 1);

bayes_horseshoe_plus_mspe = zeros(iter*num_folds, 1);
bayes_horseshoe_plus_neglike = zeros(iter*num_folds, 1);

% declare normal logistic regression auc and negative log likelihood
normal_mspe = zeros(iter*num_folds, 1);
normal_neglike = zeros(iter*num_folds, 1);

% repeat for iter number of iterations
for i = 1:iter
    % generate cross validation partitions
    cv = cvpartition(length(df), 'KFold', num_folds);
    for j = 1:cv.NumTestSets
        % for each iteration in cross validation, get indexes of training 
        % and testing set
        trIdx = cv.training(j);
        teIdx = cv.test(j);
        
        % split traing and testing data according to indexes
        %trdf_X = df(trIdx, 1:end-1);
        trdf_X = D(trIdx, 1:end);
        trdf_y = df(trIdx, end);
        tedf_X = D(teIdx, 1:end);
        tedf_y = df(teIdx, end);
        
        % since bayesreg cannot accept invariate data, delete these columns
        colStay = (var(trdf_X) ~= 0);
        trdf_X = trdf_X(:, colStay);
        tedf_X = tedf_X(:, colStay);
        
        % run bayesreg with lasso prior and show result stats
        [beta, beta0, retval] = bayesreg(trdf_X, trdf_y, 'gaussian','lasso','nsamples',1e4,'burnin',1e4,'thin',5,'display',false);
        [pred_test_y, predstats_lasso] = br_predict(tedf_X, beta, beta0, retval, 'ytest', tedf_y, 'display', false);
    
        
        % append mspe and negative log likelihood into bayesian lists
        bayes_lasso_mspe((i-1)*num_folds + j) = predstats_lasso.mspe;
        bayes_lasso_neglike((i-1)*num_folds + j) = predstats_lasso.neglike;
        %disp('The AUC of the bayesian logistic regression is:')
        %display(predstats.auc)
        %disp('The negative log likelihood of the bayesian logistic regression is:')
        %display(predstats.neglike)
        
        % run bayesreg with horsehoe prior and show result stats
        [beta_hs, beta0_hs, retval_hs] =  bayesreg(trdf_X, trdf_y, 'gaussian','horseshoe','nsamples',1e4,'burnin',1e4,'thin',5,'display',false);
        [pred_test_y_hs, predstats_hs] = br_predict(tedf_X, beta_hs, beta0_hs, retval_hs, 'ytest', tedf_y, 'display', false);
        
        % append mspe and negative log likelihood into bayesian lists
        bayes_horseshoe_mspe((i-1)*num_folds + j) = predstats_hs.mspe;
        bayes_horseshoe_neglike((i-1)*num_folds + j) = predstats_hs.neglike;
        
        % run bayesreg with horsehoe plus prior and show result stats
        [beta_hs_plus, beta0_hs_plus, retval_hs_plus] =  bayesreg(trdf_X, trdf_y, 'gaussian','horseshoe+','nsamples',1e4,'burnin',1e4,'thin',5,'display',false);
        [pred_test_y_hs_plus, predstats_hs_plus] = br_predict(tedf_X, beta_hs_plus, beta0_hs_plus, retval_hs_plus, 'ytest', tedf_y, 'display', false);
        
        % append mspe and negative log likelihood into bayesian lists
        bayes_horseshoe_plus_mspe((i-1)*num_folds + j) = predstats_hs_plus.mspe;
        bayes_horseshoe_plus_neglike((i-1)*num_folds + j) =  predstats_hs_plus.neglike;
        
        
        % run normal lasso on linear regression
        [B,FitInfo] = lassoglm(trdf_X, trdf_y,'normal','CV',3);
        idxLambdaMinDeviance = FitInfo.IndexMinDeviance;
        B0 = FitInfo.Intercept(idxLambdaMinDeviance);
        coef = [B0; B(:,idxLambdaMinDeviance)];
        num_of_coef = size(coef,1);
        training_size = size(tedf_y,1);
        yhat = glmval(coef, tedf_X,'identity');
        
        pred_mspe = mean((yhat - tedf_y).^2);
        
        pred_neglike = cal_neglike_reg(tedf_y, yhat, num_of_coef, training_size);
        %disp('The AUC for the normal logistic regression is:')
        %display(pred_auc)
        %disp('The negative log likelihood for the normal logistic regression is:')
        %display(pred_neglike)
        
        %disp('###########################################################')
        % append auc and negative log likelihood into normal lists
        normal_neglike((i-1)*num_folds + j) = pred_neglike;
        normal_mspe((i-1)*num_folds + j) = pred_mspe;
    end
end

% show the average bayesian auc and negative loglikelihood
%disp('###################################################################')
%disp('###################################################################')

disp("The summarized stats for baysian lasso")
disp([mean(bayes_lasso_mspe), mean(bayes_lasso_neglike)])
disp("The summarized stats for baysian horseshoe")
disp([mean(bayes_horseshoe_mspe), mean(bayes_horseshoe_neglike)])
disp("The summarized stats for baysian horseshoe plus")
disp([mean(bayes_horseshoe_plus_mspe), mean(bayes_horseshoe_plus_neglike)])
disp("The summarized stats for normal linear regression")
disp([mean(normal_mspe), mean(normal_neglike)])
    
