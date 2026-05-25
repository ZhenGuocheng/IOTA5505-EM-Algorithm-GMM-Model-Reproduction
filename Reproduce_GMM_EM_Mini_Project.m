%% reproduce_gmm_em_mini_project_with_update_check.m
% Reproducible GMM--EM experiment for the mini project:
% "From Incomplete Data to Maximum Likelihood: The EM Algorithm"
%
% Author: Guocheng Zhen
% Student ID: 50028129
%
% What this script does:
%   1. Generates synthetic 2D Gaussian-mixture data from known parameters.
%   2. Runs EM using only the observations, not the latent labels.
%   3. Records the observed log-likelihood trace.
%   4. Records the full parameter history across EM iterations.
%   5. Checks, at every iteration, that the numerical M-step parameters
%      agree with an independently recomputed closed-form theoretical M-step.
%   6. Compares final estimates with the ground-truth parameters after
%      label matching.
%   7. Saves tables and figures for direct use in the report.
%
% The implementation is self-contained and does not require the Statistics
% and Machine Learning Toolbox.  It uses custom routines for Gaussian-density
% evaluation and categorical sampling.

clear; clc; close all;

%% ------------------------- User-controlled settings -------------------------
outputDir = 'gmm_em_results';
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Synthetic GMM setting
n = 900;                 % sample size
d = 2;                   % data dimension
K = 3;                   % number of mixture components

% Reproducibility
dataSeed = 103;          % seed for synthetic data generation
initSeed = 1;            % seed for EM initialization

% EM settings
maxIter = 200;
tol = 1e-8;
ridge = 1e-6;            % covariance regularization
numStarts = 1;           % set to >1 to demonstrate initialization effects

% Ground-truth parameters
truePi = [0.35, 0.40, 0.25];

trueMu = [
    -4.0,  0.0;
     0.0,  4.0;
     4.0,  0.0
];

trueSigma = zeros(d, d, K);
trueSigma(:, :, 1) = [0.70,  0.20;  0.20, 0.50];
trueSigma(:, :, 2) = [0.60, -0.15; -0.15, 0.80];
trueSigma(:, :, 3) = [0.90,  0.25;  0.25, 0.70];

%% -------------------------- Generate synthetic data -------------------------
rng(dataSeed, 'twister');
[X, trueLabels] = sampleGMM(n, truePi, trueMu, trueSigma);

dataTable = table((1:n)', X(:,1), X(:,2), trueLabels, ...
    'VariableNames', {'index', 'y1', 'y2', 'true_label'});
writetable(dataTable, fullfile(outputDir, 'synthetic_gmm_data.csv'));

%% ------------------------------ Run EM --------------------------------------
bestRun = struct();
bestRun.finalLogLik = -Inf;

for s = 1:numStarts
    rng(initSeed + s - 1, 'twister');
    theta0 = initializeRandomFromData(X, K, ridge);

    run = runGMMEM(X, K, theta0, maxIter, tol, ridge);
    run.startIndex = s;
    run.startSeed = initSeed + s - 1;

    if run.finalLogLik > bestRun.finalLogLik
        bestRun = run;
    end
end

thetaHat = bestRun.theta;
llTrace = bestRun.llTrace;
thetaHistory = bestRun.thetaHistory;
updateConsistency = bestRun.updateConsistency;
updateParameterComparison = bestRun.updateParameterComparison;
numIter = numel(llTrace) - 1;  % because llTrace includes iteration 0

%% -------------------------- Label matching ----------------------------------
% GMM labels are arbitrary.  We match estimated components to true components
% by minimizing the sum of squared distances between estimated and true means.
[permEstForTrue, estToTrue] = matchComponentsByMeans(thetaHat.mu, trueMu);

matchedPi = thetaHat.pi(permEstForTrue);
matchedMu = thetaHat.mu(permEstForTrue, :);
matchedSigma = thetaHat.Sigma(:, :, permEstForTrue);

% Final responsibilities and hard labels, only for post-hoc diagnostic checks.
[gammaFinal, ~] = eStepResponsibilities(X, thetaHat);
[~, predLabelsEst] = max(gammaFinal, [], 2);
predLabelsMatched = zeros(size(predLabelsEst));
for i = 1:n
    predLabelsMatched(i) = estToTrue(predLabelsEst(i));
end
classificationAccuracy = mean(predLabelsMatched == trueLabels);

%% ----------------------------- Diagnostics ----------------------------------
llDiff = diff(llTrace);
minIncrement = min(llDiff);
maxDeltaPi = max(updateConsistency.delta_pi_inf);
maxDeltaMu = max(updateConsistency.delta_mu_max_l2);
maxDeltaSigma = max(updateConsistency.delta_sigma_max_fro);
maxDeltaOverall = max(updateConsistency.delta_overall);

fprintf('\n============================================================\n');
fprintf('GMM--EM synthetic reproduction experiment\n');
fprintf('============================================================\n');
fprintf('n = %d, d = %d, K = %d\n', n, d, K);
fprintf('Data seed = %d, initialization seed = %d\n', dataSeed, initSeed);
fprintf('Number of random starts = %d\n', numStarts);
fprintf('Best start index = %d, best start seed = %d\n', bestRun.startIndex, bestRun.startSeed);
fprintf('Number of EM iterations = %d\n', numIter);
fprintf('Final observed log-likelihood = %.10f\n', bestRun.finalLogLik);
fprintf('Smallest likelihood increment = %.12e\n', minIncrement);
fprintf('Maximum theory-code update discrepancy for pi = %.12e\n', maxDeltaPi);
fprintf('Maximum theory-code update discrepancy for mu = %.12e\n', maxDeltaMu);
fprintf('Maximum theory-code update discrepancy for Sigma = %.12e\n', maxDeltaSigma);
fprintf('Maximum overall theory-code update discrepancy = %.12e\n', maxDeltaOverall);
fprintf('Post-hoc hard-label accuracy = %.4f\n', classificationAccuracy);

if minIncrement < -1e-7
    warning('Observed log-likelihood is not monotone up to tolerance. Check implementation.');
else
    fprintf('Monotonicity check: passed up to numerical tolerance.\n');
end

if maxDeltaOverall > 1e-10
    warning('Theory-code update consistency discrepancy is larger than 1e-10.');
else
    fprintf('Theory-code update consistency check: passed up to numerical precision.\n');
end

%% ---------------------------- Print tables ----------------------------------
fprintf('\nSelected observed log-likelihood values:\n');
selectedIter = unique([0, 1, 2, 3, 4, 5, 10, 15, 20, numIter]);
selectedIter = selectedIter(selectedIter <= numIter);
selectedLL = llTrace(selectedIter + 1);

for j = 1:numel(selectedIter)
    fprintf('  iter %3d: %.10f\n', selectedIter(j), selectedLL(j));
end

fprintf('\nParameter recovery after label matching:\n');
fprintf('Comp.   true_pi     est_pi        true_mu                    est_mu\n');
for k = 1:K
    fprintf('%3d    %8.4f   %8.4f     (%8.4f,%8.4f)       (%8.4f,%8.4f)\n', ...
        k, truePi(k), matchedPi(k), trueMu(k,1), trueMu(k,2), matchedMu(k,1), matchedMu(k,2));
end

fprintf('\nEstimated covariance matrices after label matching:\n');
for k = 1:K
    fprintf('Component %d:\n', k);
    disp(matchedSigma(:, :, k));
end

%% ----------------------------- Save CSV tables ------------------------------
llTable = table(selectedIter(:), selectedLL(:), ...
    'VariableNames', {'iteration', 'observed_log_likelihood'});
writetable(llTable, fullfile(outputDir, 'selected_log_likelihood_trace.csv'));

fullLLTable = table((0:numIter)', llTrace(:), ...
    'VariableNames', {'iteration', 'observed_log_likelihood'});
writetable(fullLLTable, fullfile(outputDir, 'full_log_likelihood_trace.csv'));

parameterHistoryTable = makeParameterHistoryTable(thetaHistory, llTrace);
writetable(parameterHistoryTable, fullfile(outputDir, 'parameter_history.csv'));

writetable(updateConsistency, fullfile(outputDir, 'update_consistency_check.csv'));
writetable(updateParameterComparison, fullfile(outputDir, 'update_parameter_comparison.csv'));

selectedTransitionFrom = unique([0, 1, 2, 3, 4, 5, 10, 15, numIter - 1]);
selectedTransitionFrom = selectedTransitionFrom(selectedTransitionFrom >= 0 & selectedTransitionFrom < numIter);
selectedUpdateConsistency = updateConsistency(ismember(updateConsistency.iteration_from, selectedTransitionFrom), :);
writetable(selectedUpdateConsistency, fullfile(outputDir, 'selected_update_consistency_check.csv'));

comp = (1:K)';
trueMu1 = trueMu(:,1);
trueMu2 = trueMu(:,2);
estMu1 = matchedMu(:,1);
estMu2 = matchedMu(:,2);
paramTable = table(comp, truePi(:), matchedPi(:), trueMu1, trueMu2, estMu1, estMu2, ...
    'VariableNames', {'component', 'true_pi', 'estimated_pi', ...
                      'true_mu_1', 'true_mu_2', 'estimated_mu_1', 'estimated_mu_2'});
writetable(paramTable, fullfile(outputDir, 'parameter_recovery.csv'));

covRows = [];
for k = 1:K
    row = [k, reshape(trueSigma(:,:,k), 1, []), reshape(matchedSigma(:,:,k), 1, [])];
    covRows = [covRows; row]; %#ok<AGROW>
end
covTable = array2table(covRows, 'VariableNames', ...
    {'component', ...
     'true_S11', 'true_S21', 'true_S12', 'true_S22', ...
     'estimated_S11', 'estimated_S21', 'estimated_S12', 'estimated_S22'});
writetable(covTable, fullfile(outputDir, 'covariance_recovery.csv'));

summaryTable = table(n, d, K, dataSeed, initSeed, numStarts, numIter, ...
    bestRun.finalLogLik, minIncrement, maxDeltaPi, maxDeltaMu, maxDeltaSigma, ...
    maxDeltaOverall, classificationAccuracy, ...
    'VariableNames', {'n','d','K','data_seed','init_seed','num_starts','num_iterations', ...
                      'final_log_likelihood','smallest_likelihood_increment', ...
                      'max_delta_pi_inf','max_delta_mu_l2','max_delta_sigma_fro', ...
                      'max_delta_overall','classification_accuracy'});
writetable(summaryTable, fullfile(outputDir, 'experiment_summary.csv'));

%% ---------------------------- Save LaTeX tables -----------------------------
writeLikelihoodLatex(fullfile(outputDir, 'selected_log_likelihood_table.tex'), ...
    selectedIter, selectedLL);

writeParameterLatex(fullfile(outputDir, 'parameter_recovery_table.tex'), ...
    truePi, matchedPi, trueMu, matchedMu);

writeCovarianceLatex(fullfile(outputDir, 'estimated_covariance_table.tex'), ...
    matchedSigma);

writeUpdateConsistencyLatex(fullfile(outputDir, 'selected_update_consistency_table.tex'), ...
    selectedUpdateConsistency);

%% -------------------------------- Figures -----------------------------------
% Figure 1: synthetic data colored by true latent labels.
figure('Color', 'w');
hold on;
for k = 1:K
    idx = (trueLabels == k);
    scatter(X(idx,1), X(idx,2), 18, 'filled');
end
hold off;
grid on;
axis equal;
xlabel('$y_1$', 'Interpreter', 'latex');
ylabel('$y_2$', 'Interpreter', 'latex');
title('Synthetic GMM data colored by true latent labels', 'Interpreter', 'latex');
legendStrings = arrayfun(@(k) sprintf('True component %d', k), 1:K, 'UniformOutput', false);
legend(legendStrings, 'Location', 'best');
saveFigure(gcf, fullfile(outputDir, 'synthetic_data_true_labels.png'));

% Figure 2: observed log-likelihood trace.
figure('Color', 'w');
plot(0:numIter, llTrace, '-o', 'LineWidth', 1.2, 'MarkerSize', 4);
grid on;
xlabel('EM iteration $t$', 'Interpreter', 'latex');
ylabel('Observed log-likelihood $\ell_o(\theta^{(t)})$', 'Interpreter', 'latex');
title('Observed log-likelihood trace', 'Interpreter', 'latex');
saveFigure(gcf, fullfile(outputDir, 'log_likelihood_trace.png'));

% Figure 3: final hard assignments after EM, matched to true label order.
figure('Color', 'w');
hold on;
for k = 1:K
    idx = (predLabelsMatched == k);
    scatter(X(idx,1), X(idx,2), 18, 'filled');
end
plot(matchedMu(:,1), matchedMu(:,2), 'kx', 'LineWidth', 2.0, 'MarkerSize', 10);
plot(trueMu(:,1), trueMu(:,2), 'ko', 'LineWidth', 1.5, 'MarkerSize', 8);
hold off;
grid on;
axis equal;
xlabel('$y_1$', 'Interpreter', 'latex');
ylabel('$y_2$', 'Interpreter', 'latex');
title('Final EM hard assignments and component means', 'Interpreter', 'latex');
legendStrings = arrayfun(@(k) sprintf('Predicted component %d', k), 1:K, 'UniformOutput', false);
legendStrings{end+1} = 'Estimated means';
legendStrings{end+1} = 'True means';
legend(legendStrings, 'Location', 'best');
saveFigure(gcf, fullfile(outputDir, 'final_em_assignments.png'));

fprintf('\nSaved outputs to folder: %s\n', outputDir);
fprintf('Key files:\n');
fprintf('  - full_log_likelihood_trace.csv\n');
fprintf('  - selected_log_likelihood_trace.csv\n');
fprintf('  - parameter_history.csv\n');
fprintf('  - update_consistency_check.csv\n');
fprintf('  - update_parameter_comparison.csv\n');
fprintf('  - selected_update_consistency_table.tex\n');
fprintf('  - parameter_recovery.csv\n');
fprintf('  - covariance_recovery.csv\n');
fprintf('  - selected_log_likelihood_table.tex\n');
fprintf('  - parameter_recovery_table.tex\n');
fprintf('  - estimated_covariance_table.tex\n');
fprintf('  - synthetic_data_true_labels.png\n');
fprintf('  - log_likelihood_trace.png\n');
fprintf('  - final_em_assignments.png\n');

%% ============================== Local functions =============================

function [X, labels] = sampleGMM(n, piVec, muMat, SigmaArray)
% sampleGMM samples n points from a Gaussian mixture model.
    K = numel(piVec);
    d = size(muMat, 2);
    labels = sampleCategorical(piVec, n);
    X = zeros(n, d);

    for i = 1:n
        k = labels(i);
        L = chol(SigmaArray(:, :, k), 'lower');
        X(i, :) = muMat(k, :) + randn(1, d) * L';
    end
end

function labels = sampleCategorical(piVec, n)
% sampleCategorical samples from categorical probabilities piVec.
    piVec = piVec(:)' / sum(piVec);
    edges = cumsum(piVec);
    edges(end) = 1.0;
    u = rand(n, 1);
    labels = zeros(n, 1);
    for i = 1:n
        labels(i) = find(u(i) <= edges, 1, 'first');
    end
end

function theta = initializeRandomFromData(X, K, ridge)
% initializeRandomFromData initializes means from random data points,
% weights uniformly, and covariances using the full empirical covariance.
    [n, d] = size(X);
    idx = randperm(n, K);

    theta.pi = ones(1, K) / K;
    theta.mu = X(idx, :);

    empiricalCov = cov(X, 1);
    empiricalCov = makeSymmetricPD(empiricalCov, ridge);

    theta.Sigma = zeros(d, d, K);
    for k = 1:K
        theta.Sigma(:, :, k) = empiricalCov;
    end
end

function run = runGMMEM(X, K, theta0, maxIter, tol, ridge)
% runGMMEM runs exact EM for a Gaussian mixture model.
%
% Besides the likelihood trace, this routine records two implementation
% diagnostics:
%   1. thetaHistory stores the parameters after every EM iteration.
%   2. updateConsistency compares the stored numerical M-step output with an
%      independently recomputed closed-form theoretical M-step at the same
%      responsibility matrix.  The discrepancies should be at roundoff level.
    theta = theta0;
    llTrace = zeros(maxIter + 1, 1);
    thetaHistory = cell(maxIter + 1, 1);
    formulaUpdateHistory = cell(maxIter, 1);
    codeUpdateHistory = cell(maxIter, 1);

    updateConsistencyRaw = zeros(maxIter, 6);
    % Columns: iteration_from, iteration_to, delta_pi_inf,
    % delta_mu_max_l2, delta_sigma_max_fro, delta_overall.

    llTrace(1) = computeLogLikelihood(X, theta);
    thetaHistory{1} = theta;

    actualIterations = maxIter;

    for iter = 1:maxIter
        [gamma, ~] = eStepResponsibilities(X, theta);

        % Numerical update used by the EM implementation.
        thetaCode = mStepWeightedGaussian(X, gamma, ridge);

        % Independent reference update obtained by directly substituting the
        % same responsibilities into the closed-form GMM M-step formulas.
        thetaFormula = mStepClosedFormReference(X, gamma, ridge);

        [deltaPi, deltaMu, deltaSigma] = compareThetaUpdates(thetaCode, thetaFormula);
        deltaOverall = max([deltaPi, deltaMu, deltaSigma]);
        updateConsistencyRaw(iter, :) = [iter - 1, iter, deltaPi, deltaMu, deltaSigma, deltaOverall];

        codeUpdateHistory{iter} = thetaCode;
        formulaUpdateHistory{iter} = thetaFormula;

        theta = thetaCode;
        thetaHistory{iter + 1} = theta;
        llTrace(iter + 1) = computeLogLikelihood(X, theta);

        if abs(llTrace(iter + 1) - llTrace(iter)) < tol
            actualIterations = iter;
            break;
        end
    end

    llTrace = llTrace(1:actualIterations + 1);
    thetaHistory = thetaHistory(1:actualIterations + 1);
    codeUpdateHistory = codeUpdateHistory(1:actualIterations);
    formulaUpdateHistory = formulaUpdateHistory(1:actualIterations);
    updateConsistencyRaw = updateConsistencyRaw(1:actualIterations, :);

    updateConsistency = array2table(updateConsistencyRaw, 'VariableNames', ...
        {'iteration_from', 'iteration_to', 'delta_pi_inf', ...
         'delta_mu_max_l2', 'delta_sigma_max_fro', 'delta_overall'});

    updateParameterComparison = makeUpdateParameterComparisonTable(codeUpdateHistory, formulaUpdateHistory);

    run.theta = theta;
    run.llTrace = llTrace;
    run.thetaHistory = thetaHistory;
    run.updateConsistency = updateConsistency;
    run.updateParameterComparison = updateParameterComparison;
    run.finalLogLik = llTrace(end);
end

function ll = computeLogLikelihood(X, theta)
% computeLogLikelihood evaluates the observed-data log-likelihood.
% For a GMM, this is sum_i log(sum_k pi_k N(x_i; mu_k, Sigma_k)).
    [~, logDenom] = eStepResponsibilities(X, theta);
    ll = sum(logDenom);
end

function [gamma, logDenom] = eStepResponsibilities(X, theta)
% eStepResponsibilities computes posterior responsibilities.
    [n, ~] = size(X);
    K = numel(theta.pi);

    logResp = zeros(n, K);
    for k = 1:K
        logResp(:, k) = log(theta.pi(k)) + logGaussianPDF(X, theta.mu(k, :), theta.Sigma(:, :, k));
    end

    logDenom = logsumexpRows(logResp);
    gamma = exp(bsxfun(@minus, logResp, logDenom));
end

function theta = mStepWeightedGaussian(X, gamma, ridge)
% mStepWeightedGaussian performs weighted Gaussian MLE updates.
    [n, d] = size(X);
    K = size(gamma, 2);

    Nk = sum(gamma, 1);
    theta.pi = Nk / n;
    theta.mu = zeros(K, d);
    theta.Sigma = zeros(d, d, K);

    for k = 1:K
        if Nk(k) <= eps
            error('Component %d has effectively zero responsibility. Try another initialization.', k);
        end

        theta.mu(k, :) = (gamma(:, k)' * X) / Nk(k);

        Xc = X - theta.mu(k, :);
        weightedXc = Xc .* gamma(:, k);
        SigmaK = (Xc' * weightedXc) / Nk(k);
        theta.Sigma(:, :, k) = makeSymmetricPD(SigmaK, ridge);
    end
end

function theta = mStepClosedFormReference(X, gamma, ridge)
% mStepClosedFormReference recomputes the theoretical closed-form M-step.
%
% This function intentionally uses explicit loops rather than calling the
% numerical update routine.  It provides an independent implementation check:
% after the E-step produces gamma, the formulas are
%   pi_k     = N_k / n,
%   mu_k     = sum_i gamma_ik x_i / N_k,
%   Sigma_k  = sum_i gamma_ik (x_i-mu_k)(x_i-mu_k)' / N_k.
    [n, d] = size(X);
    K = size(gamma, 2);

    theta.pi = zeros(1, K);
    theta.mu = zeros(K, d);
    theta.Sigma = zeros(d, d, K);

    for k = 1:K
        Nk = sum(gamma(:, k));
        if Nk <= eps
            error('Component %d has effectively zero responsibility in the reference M-step.', k);
        end

        theta.pi(k) = Nk / n;

        numeratorMu = zeros(1, d);
        for i = 1:n
            numeratorMu = numeratorMu + gamma(i, k) * X(i, :);
        end
        theta.mu(k, :) = numeratorMu / Nk;

        SigmaK = zeros(d, d);
        for i = 1:n
            diff = X(i, :) - theta.mu(k, :);
            SigmaK = SigmaK + gamma(i, k) * (diff' * diff);
        end
        SigmaK = SigmaK / Nk;
        theta.Sigma(:, :, k) = makeSymmetricPD(SigmaK, ridge);
    end
end

function [deltaPi, deltaMu, deltaSigma] = compareThetaUpdates(thetaCode, thetaFormula)
% compareThetaUpdates returns aggregate differences between two parameter sets.
    K = numel(thetaCode.pi);

    deltaPi = max(abs(thetaCode.pi(:) - thetaFormula.pi(:)));

    deltaMu = 0;
    for k = 1:K
        diffMu = thetaCode.mu(k, :) - thetaFormula.mu(k, :);
        deltaMu = max(deltaMu, sqrt(sum(diffMu.^2)));
    end

    deltaSigma = 0;
    for k = 1:K
        diffSigma = thetaCode.Sigma(:, :, k) - thetaFormula.Sigma(:, :, k);
        deltaSigma = max(deltaSigma, sqrt(sum(diffSigma(:).^2)));
    end
end

function parameterHistoryTable = makeParameterHistoryTable(thetaHistory, llTrace)
% makeParameterHistoryTable stores the raw EM parameter trajectory.
%
% The table has one row for each pair (iteration, component).  Since this
% experiment is two-dimensional, the covariance entries are written as S11,
% S12, S21, and S22.
    numIterPlusOne = numel(thetaHistory);
    K = numel(thetaHistory{1}.pi);
    numRows = numIterPlusOne * K;

    iteration = zeros(numRows, 1);
    component = zeros(numRows, 1);
    observedLogLikelihood = zeros(numRows, 1);
    piValue = zeros(numRows, 1);
    mu1 = zeros(numRows, 1);
    mu2 = zeros(numRows, 1);
    S11 = zeros(numRows, 1);
    S12 = zeros(numRows, 1);
    S21 = zeros(numRows, 1);
    S22 = zeros(numRows, 1);

    row = 0;
    for t = 0:(numIterPlusOne - 1)
        theta = thetaHistory{t + 1};
        for k = 1:K
            row = row + 1;
            iteration(row) = t;
            component(row) = k;
            observedLogLikelihood(row) = llTrace(t + 1);
            piValue(row) = theta.pi(k);
            mu1(row) = theta.mu(k, 1);
            mu2(row) = theta.mu(k, 2);
            S = theta.Sigma(:, :, k);
            S11(row) = S(1, 1);
            S12(row) = S(1, 2);
            S21(row) = S(2, 1);
            S22(row) = S(2, 2);
        end
    end

    parameterHistoryTable = table(iteration, component, observedLogLikelihood, ...
        piValue, mu1, mu2, S11, S12, S21, S22, ...
        'VariableNames', {'iteration', 'component', 'observed_log_likelihood', ...
                          'pi', 'mu_1', 'mu_2', 'Sigma_11', 'Sigma_12', ...
                          'Sigma_21', 'Sigma_22'});
end

function comparisonTable = makeUpdateParameterComparisonTable(codeUpdateHistory, formulaUpdateHistory)
% makeUpdateParameterComparisonTable stores entrywise theory-code comparisons.
%
% Each row corresponds to one component in one EM transition t -> t+1.
% The ``code'' columns are the parameters actually stored by the EM routine,
% while the ``formula'' columns are obtained by substituting the same gamma
% matrix into the theoretical closed-form updates.
    numTransitions = numel(codeUpdateHistory);
    K = numel(codeUpdateHistory{1}.pi);
    numRows = numTransitions * K;

    iterationFrom = zeros(numRows, 1);
    iterationTo = zeros(numRows, 1);
    component = zeros(numRows, 1);

    codePi = zeros(numRows, 1);
    formulaPi = zeros(numRows, 1);
    absDeltaPi = zeros(numRows, 1);

    codeMu1 = zeros(numRows, 1);
    formulaMu1 = zeros(numRows, 1);
    absDeltaMu1 = zeros(numRows, 1);
    codeMu2 = zeros(numRows, 1);
    formulaMu2 = zeros(numRows, 1);
    absDeltaMu2 = zeros(numRows, 1);

    codeS11 = zeros(numRows, 1);
    formulaS11 = zeros(numRows, 1);
    absDeltaS11 = zeros(numRows, 1);
    codeS12 = zeros(numRows, 1);
    formulaS12 = zeros(numRows, 1);
    absDeltaS12 = zeros(numRows, 1);
    codeS21 = zeros(numRows, 1);
    formulaS21 = zeros(numRows, 1);
    absDeltaS21 = zeros(numRows, 1);
    codeS22 = zeros(numRows, 1);
    formulaS22 = zeros(numRows, 1);
    absDeltaS22 = zeros(numRows, 1);

    row = 0;
    for t = 1:numTransitions
        thetaCode = codeUpdateHistory{t};
        thetaFormula = formulaUpdateHistory{t};
        for k = 1:K
            row = row + 1;
            iterationFrom(row) = t - 1;
            iterationTo(row) = t;
            component(row) = k;

            codePi(row) = thetaCode.pi(k);
            formulaPi(row) = thetaFormula.pi(k);
            absDeltaPi(row) = abs(codePi(row) - formulaPi(row));

            codeMu1(row) = thetaCode.mu(k, 1);
            formulaMu1(row) = thetaFormula.mu(k, 1);
            absDeltaMu1(row) = abs(codeMu1(row) - formulaMu1(row));

            codeMu2(row) = thetaCode.mu(k, 2);
            formulaMu2(row) = thetaFormula.mu(k, 2);
            absDeltaMu2(row) = abs(codeMu2(row) - formulaMu2(row));

            codeS = thetaCode.Sigma(:, :, k);
            formulaS = thetaFormula.Sigma(:, :, k);

            codeS11(row) = codeS(1, 1);
            formulaS11(row) = formulaS(1, 1);
            absDeltaS11(row) = abs(codeS11(row) - formulaS11(row));

            codeS12(row) = codeS(1, 2);
            formulaS12(row) = formulaS(1, 2);
            absDeltaS12(row) = abs(codeS12(row) - formulaS12(row));

            codeS21(row) = codeS(2, 1);
            formulaS21(row) = formulaS(2, 1);
            absDeltaS21(row) = abs(codeS21(row) - formulaS21(row));

            codeS22(row) = codeS(2, 2);
            formulaS22(row) = formulaS(2, 2);
            absDeltaS22(row) = abs(codeS22(row) - formulaS22(row));
        end
    end

    comparisonTable = table(iterationFrom, iterationTo, component, ...
        codePi, formulaPi, absDeltaPi, ...
        codeMu1, formulaMu1, absDeltaMu1, ...
        codeMu2, formulaMu2, absDeltaMu2, ...
        codeS11, formulaS11, absDeltaS11, ...
        codeS12, formulaS12, absDeltaS12, ...
        codeS21, formulaS21, absDeltaS21, ...
        codeS22, formulaS22, absDeltaS22, ...
        'VariableNames', {'iteration_from', 'iteration_to', 'component', ...
                          'code_pi', 'formula_pi', 'abs_delta_pi', ...
                          'code_mu_1', 'formula_mu_1', 'abs_delta_mu_1', ...
                          'code_mu_2', 'formula_mu_2', 'abs_delta_mu_2', ...
                          'code_Sigma_11', 'formula_Sigma_11', 'abs_delta_Sigma_11', ...
                          'code_Sigma_12', 'formula_Sigma_12', 'abs_delta_Sigma_12', ...
                          'code_Sigma_21', 'formula_Sigma_21', 'abs_delta_Sigma_21', ...
                          'code_Sigma_22', 'formula_Sigma_22', 'abs_delta_Sigma_22'});
end

function logp = logGaussianPDF(X, mu, Sigma)
% logGaussianPDF evaluates log N(X; mu, Sigma) row-wise.
    [n, d] = size(X);
    Xc = X - mu;

    Sigma = (Sigma + Sigma') / 2;
    [L, flag] = chol(Sigma, 'lower');
    if flag ~= 0
        Sigma = makeSymmetricPD(Sigma, 1e-8);
        L = chol(Sigma, 'lower');
    end

    sol = Xc / L';
    maha = sum(sol.^2, 2);
    logdet = 2 * sum(log(diag(L)));

    logp = -0.5 * (d * log(2*pi) + logdet + maha);
    if any(~isfinite(logp)) || numel(logp) ~= n
        error('Non-finite Gaussian log-density encountered.');
    end
end

function y = logsumexpRows(A)
% logsumexpRows computes log(sum(exp(A),2)) stably.
    m = max(A, [], 2);
    y = m + log(sum(exp(bsxfun(@minus, A, m)), 2));
end

function S = makeSymmetricPD(S, ridge)
% makeSymmetricPD symmetrizes S and adds enough ridge to make it positive definite.
    S = (S + S') / 2;
    d = size(S, 1);

    minEig = min(eig(S));
    if minEig <= ridge
        S = S + (ridge - minEig + ridge) * eye(d);
    else
        S = S + ridge * eye(d);
    end

    S = (S + S') / 2;
end

function [permEstForTrue, estToTrue] = matchComponentsByMeans(estMu, trueMu)
% matchComponentsByMeans matches estimated components to true components.
%
% Output:
%   permEstForTrue(j) = estimated-component index matched to true component j.
%   estToTrue(k)      = true-component index matched to estimated component k.
    K = size(trueMu, 1);
    allPerms = perms(1:K);

    bestScore = Inf;
    bestPerm = allPerms(1, :);

    for r = 1:size(allPerms, 1)
        perm = allPerms(r, :);
        score = 0;
        for j = 1:K
            diff = estMu(perm(j), :) - trueMu(j, :);
            score = score + sum(diff.^2);
        end
        if score < bestScore
            bestScore = score;
            bestPerm = perm;
        end
    end

    permEstForTrue = bestPerm;
    estToTrue = zeros(1, K);
    for j = 1:K
        estToTrue(permEstForTrue(j)) = j;
    end
end

function saveFigure(figHandle, filePath)
% saveFigure saves a figure robustly across MATLAB versions.
    try
        exportgraphics(figHandle, filePath, 'Resolution', 300);
    catch
        saveas(figHandle, filePath);
    end
end

function writeLikelihoodLatex(filePath, iter, ll)
% writeLikelihoodLatex writes a small booktabs LaTeX table.
    fid = fopen(filePath, 'w');
    if fid < 0
        error('Cannot open %s for writing.', filePath);
    end

    fprintf(fid, '%% Auto-generated by reproduce_gmm_em_mini_project.m\n');
    fprintf(fid, '\\begin{tabular}{rr}\n');
    fprintf(fid, '\\toprule\n');
    fprintf(fid, 'Iteration $t$ & $\\ell_o(\\theta^{(t)})$ \\\\\n');
    fprintf(fid, '\\midrule\n');
    for i = 1:numel(iter)
        fprintf(fid, '%d & %.10f \\\\\n', iter(i), ll(i));
    end
    fprintf(fid, '\\bottomrule\n');
    fprintf(fid, '\\end{tabular}\n');

    fclose(fid);
end

function writeParameterLatex(filePath, truePi, estPi, trueMu, estMu)
% writeParameterLatex writes a parameter-recovery LaTeX table.
    K = numel(truePi);

    fid = fopen(filePath, 'w');
    if fid < 0
        error('Cannot open %s for writing.', filePath);
    end

    fprintf(fid, '%% Auto-generated by reproduce_gmm_em_mini_project.m\n');
    fprintf(fid, '\\begin{tabular}{ccccc}\n');
    fprintf(fid, '\\toprule\n');
    fprintf(fid, 'Comp. & $\\pi_k^\\star$ & $\\widehat\\pi_k$ & $\\mu_k^\\star$ & $\\widehat\\mu_k$ \\\\\n');
    fprintf(fid, '\\midrule\n');
    for k = 1:K
        fprintf(fid, '%d & %.4f & %.4f & $(%.4f,%.4f)$ & $(%.4f,%.4f)$ \\\\\n', ...
            k, truePi(k), estPi(k), trueMu(k,1), trueMu(k,2), estMu(k,1), estMu(k,2));
    end
    fprintf(fid, '\\bottomrule\n');
    fprintf(fid, '\\end{tabular}\n');

    fclose(fid);
end

function writeCovarianceLatex(filePath, estSigma)
% writeCovarianceLatex writes estimated covariance matrices as LaTeX display equations.
    K = size(estSigma, 3);

    fid = fopen(filePath, 'w');
    if fid < 0
        error('Cannot open %s for writing.', filePath);
    end

    fprintf(fid, '%% Auto-generated by reproduce_gmm_em_mini_project.m\n');
    fprintf(fid, '\\[\n');
    fprintf(fid, '\\begin{split}\n');

    for k = 1:K
        S = estSigma(:, :, k);
        if k < K
            ending = ',\\qquad';
        else
            ending = '.';
        end
        fprintf(fid, '\\widehat\\Sigma_%d&=\\begin{pmatrix} %.4f&%.4f\\\\ %.4f&%.4f\\end{pmatrix}%s\n', ...
            k, S(1,1), S(1,2), S(2,1), S(2,2), ending);
    end

    fprintf(fid, '\\end{split}\n');
    fprintf(fid, '\\]\n');

    fclose(fid);
end


function writeUpdateConsistencyLatex(filePath, selectedUpdateConsistency)
% writeUpdateConsistencyLatex writes selected update-consistency diagnostics.
    fid = fopen(filePath, 'w');
    if fid < 0
        error('Cannot open %s for writing.', filePath);
    end

    fprintf(fid, '%% Auto-generated by reproduce_gmm_em_mini_project.m\n');
    fprintf(fid, '\\begin{tabular}{rrrrr}\n');
    fprintf(fid, '\\toprule\n');
    fprintf(fid, 'Transition & $\\Delta_\\pi$ & $\\Delta_\\mu$ & $\\Delta_\\Sigma$ & Overall \\\\\n');
    fprintf(fid, '\\midrule\n');

    for i = 1:height(selectedUpdateConsistency)
        fprintf(fid, '$%d\\to%d$ & %.3e & %.3e & %.3e & %.3e \\\\\n', ...
            selectedUpdateConsistency.iteration_from(i), ...
            selectedUpdateConsistency.iteration_to(i), ...
            selectedUpdateConsistency.delta_pi_inf(i), ...
            selectedUpdateConsistency.delta_mu_max_l2(i), ...
            selectedUpdateConsistency.delta_sigma_max_fro(i), ...
            selectedUpdateConsistency.delta_overall(i));
    end

    fprintf(fid, '\\bottomrule\n');
    fprintf(fid, '\\end{tabular}\n');

    fclose(fid);
end
