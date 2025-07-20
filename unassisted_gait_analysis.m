% =========================================================================

% Description:
% - v3.7: Implemented normalized GRF for rgait detection using a
%   relative threshold.
%
% REQUIRES: Statistics and Machine Learning Toolbox, Signal Processing Toolbox
% =========================================================================
clear; clc; close all;

%% ------------------- USER CONFIGURATION ---------------------------------
baseDataDir = 'Z:\AnkleStudy\participantData\unassistedTreadmillWalking\20602_Sub10_5June';
outputDir = 'C:\Users\msingh25\Desktop\github_gpu\us_processing_for_neuromuscularsys\matlab_processed_results';

% --- Normalized threshold for GRF ---
% This value is relative (0-1). 0.15 means 15% of the peak force.
config.grf_stance_threshold_norm = 0.25; 

config.emg_muscles = {'sEMG_ta_raw', 'sEMG_latgas_raw', 'sEMG_medgas_raw', 'sEMG_sol_raw'};
config.pf_muscles = {'sEMG_latgas_envelope', 'sEMG_medgas_envelope', 'sEMG_sol_envelope'};
config.vaf_threshold = 0.93;
%% ------------------------------------------------------------------------

fprintf('Starting MAT File Processing and Synergy Analysis (v3.7)...\n');
timestamp = datestr(now, 'yyyymmdd-HHMMSS');
[~, subjectName, ~] = fileparts(baseDataDir);
masterOutputDir = fullfile(outputDir, sprintf('mat_results_%s_%s', subjectName, timestamp));
if ~exist(masterOutputDir, 'dir'), mkdir(masterOutputDir); end
fprintf('All results will be saved in: %s\n', masterOutputDir);

matFiles = dir(fullfile(baseDataDir, 't*_*.*.mat'));
if isempty(matFiles), error('No .mat files found.'); end
fprintf('Found %d trials to analyze.\n', length(matFiles));

for i = 1:length(matFiles)
    trialPath = fullfile(matFiles(i).folder, matFiles(i).name);
    [~, trialName, ~] = fileparts(trialPath);
    
    fprintf('\n============================================================\n');
    fprintf('Processing Trial %d/%d: %s\n', i, length(matFiles), matFiles(i).name);
    
    trialOutputDir = fullfile(masterOutputDir, trialName);
    if ~exist(trialOutputDir, 'dir'), mkdir(trialOutputDir); end
    
    loadedData = load(trialPath);
    simulinkData = loadedData.data;
    
    [time, grf] = getSignalByName(simulinkData, 'grf_left_lpf');
    if isempty(time), continue; end
    
    fs = 1 / (time(2) - time(1));
    processedTable = table(time);
    
    emgEnvelopes = [];
    emgLabels = {};
    for m = 1:length(config.emg_muscles)
        muscleName = config.emg_muscles{m};
        [~, rawEMG] = getSignalByName(simulinkData, muscleName);
        if ~isempty(rawEMG)
            envelope = processEMG(rawEMG, fs);
            envelopeColName = strrep(muscleName, '_raw', '_envelope');
            processedTable.(envelopeColName) = envelope;
            emgEnvelopes = [emgEnvelopes, envelope];
            cleanLabel = strrep(strrep(muscleName, 'sEMG_', ''), '_raw', '');
            emgLabels{end+1} = cleanLabel;
        end
    end

    processedTable = runAdvancedGaitDetection(processedTable, grf, config);
    plotGaitPhaseComparison(processedTable, config, trialOutputDir, trialName);
    
    plotNormalizedActivation(processedTable, emgLabels, config, trialOutputDir, trialName);
    
    if size(emgEnvelopes, 2) > 1
        runComparativeSynergyAnalysis(emgEnvelopes, emgLabels, time, fs, trialOutputDir, trialName, config.vaf_threshold);
    end
    
    csvPath = fullfile(trialOutputDir, [trialName, '_processed_mat_data.csv']);
    writetable(processedTable, csvPath);
    fprintf('\nProcessed data saved to: %s\n', csvPath);
end

fprintf('\n############################################################\n');
fprintf('MATLAB PROCESSING COMPLETE\n');
fprintf('############################################################\n');


% =========================================================================
% ======================== HELPER FUNCTIONS ===============================
% =========================================================================

function [time, data] = getSignalByName(simulinkData, signalName)
    time = []; data = [];
    try
        element = getElement(simulinkData, signalName);
        time = element.Values.Time(:);
        data = element.Values.Data(:);
    catch ME
        fprintf('--- WARNING: Signal "%s" not found. Error: %s\n', signalName, ME.message);
    end
end

function processedEMG = processEMG(rawEMG, fs)
    rawEMG_col = double(rawEMG(:));
    nyq = fs / 2;
    [b, a] = butter(4, [20 450] / nyq, 'bandpass');
    emg_bp = filtfilt(b, a, rawEMG_col);
    emg_rect = abs(emg_bp);
    [b, a] = butter(4, 6 / nyq, 'low');
    processedEMG = filtfilt(b, a, emg_rect);
end

function dataTable = runAdvancedGaitDetection(dataTable, grf, config)
    fprintf('Running advanced gait phase detection...\n');
    n_points = height(dataTable);
    
    % --- NEW: Normalize GRF signal to 0-1 range ---
    grf_norm = (grf - min(grf)) / (max(grf) - min(grf));
    dataTable.grf_norm = grf_norm;
    
    % Method 1: GRF-Based Detection using NORMALIZED signal
    gait_phase_grf = repmat(categorical({'Swing'}), n_points, 1);
    stance_indices = find(grf_norm > config.grf_stance_threshold_norm);
    if ~isempty(stance_indices)
        [~, heel_strike_locs] = findpeaks(grf_norm, 'MinPeakHeight', config.grf_stance_threshold_norm, 'MinPeakDistance', 200);
        gait_phase_grf(stance_indices) = categorical({'MidStance'});
        gait_phase_grf(heel_strike_locs) = categorical({'HeelStrike'});
        stance_end_indices = find(diff(grf_norm > config.grf_stance_threshold_norm) == -1);
        for k = 1:length(stance_end_indices)
           pushoff_window = max(1, stance_end_indices(k)-100):stance_end_indices(k);
           gait_phase_grf(pushoff_window) = categorical({'PushOff'});
        end
    end
    dataTable.gait_phase_grf = gait_phase_grf;

    % Method 2 & 3 remain the same but can be more robust now
    gait_phase_emg = repmat(categorical({'Swing'}), n_points, 1);
    ta_norm = dataTable.sEMG_ta_envelope / max(dataTable.sEMG_ta_envelope);
    pf_group_norm = mean(dataTable{:, config.pf_muscles}, 2) / max(mean(dataTable{:, config.pf_muscles}, 2));
    is_swing = ta_norm > 0.3 & pf_group_norm < 0.25;
    is_stance = pf_group_norm > 0.3 & ta_norm < 0.25;
    gait_phase_emg(is_swing) = categorical({'Swing'});
    gait_phase_emg(is_stance) = categorical({'MidStance'});
    dataTable.gait_phase_emg = gait_phase_emg;
    
    gait_phase_fused = dataTable.gait_phase_grf;
    stance_periods = find(gait_phase_fused ~= 'Swing');
    if ~isempty(stance_periods)
        [~, pf_peaks] = findpeaks(pf_group_norm(stance_periods), 'MinPeakHeight', 0.4, 'MinPeakDistance', 100);
        pushoff_locs_in_stance = stance_periods(pf_peaks);
        for k = 1:length(pushoff_locs_in_stance)
            pushoff_window = max(1, pushoff_locs_in_stance(k)-50):min(n_points, pushoff_locs_in_stance(k)+50);
            gait_phase_fused(pushoff_window(gait_phase_fused(pushoff_window) ~= 'Swing')) = categorical({'PushOff'});
        end
    end
    dataTable.gait_phase_fused = gait_phase_fused;
end

function plotGaitPhaseComparison(dataTable, config, outputDir, trialName)
    hFig = figure('Visible', 'off', 'Position', [100 100 1200 600]);
    phase_types = {'gait_phase_grf', 'gait_phase_emg', 'gait_phase_fused'};
    titles = {'GRF-Based Phases', 'EMG-Based Phases', 'Fused Phases'};
    colors = struct('Swing', [0.8 0.9 1], 'HeelStrike', [1 0.8 0.8], 'MidStance', [0.8 1 0.8], 'PushOff', [1 1 0.8]);
    
    for i = 1:3
        subplot(3,1,i);
        hold on;
        phases = unique(dataTable.(phase_types{i}), 'stable');
        legend_handles = gobjects(length(phases), 1);
        for p = 1:length(phases)
            phase_name = char(phases(p));
            if ~isfield(colors, phase_name), continue; end
            indices = find(dataTable.(phase_types{i}) == phase_name);
            if ~isempty(indices)
                blocks = find(diff(indices) > 1);
                starts = [indices(1); indices(blocks+1)];
                ends = [indices(blocks); indices(end)];
                for k = 1:length(starts)
                    h = area([dataTable.time(starts(k)) dataTable.time(ends(k))], [1 1], 'FaceColor', colors.(phase_name), 'EdgeColor', 'none', 'BaseValue', -1);
                    if k == 1, legend_handles(p) = h; else, set(h, 'HandleVisibility', 'off'); end
                end
            end
        end
        % --- PLOT NORMALIZED GRF AND THRESHOLD ---
        plot(dataTable.time, dataTable.grf_norm, 'k-', 'LineWidth', 1);
        yline(config.grf_stance_threshold_norm, 'r--', 'LineWidth', 2);
        hold off;
        title(titles{i}); ylim([-0.1 1.1]); xlim([dataTable.time(1) dataTable.time(end)]);
        ylabel('Normalized GRF');
        valid_legend_items = isgraphics(legend_handles) & arrayfun(@(x) ~isempty(x.DisplayName), legend_handles);
        legend(legend_handles(valid_legend_items), 'Location', 'northeast');
        if i<3, xticklabels([]); end
    end
    xlabel('Time (s)');
    sgtitle(['Gait Phase Detection Comparison for ', strrep(trialName, '_', '\_')]);
    saveas(hFig, fullfile(outputDir, [trialName, '_gait_phase_comparison.png']));
    close(hFig);
    fprintf('Gait phase comparison plot saved.\n');
end

function plotNormalizedActivation(dataTable, emgLabels, config, outputDir, trialName)
    fprintf('Plotting normalized muscle activation...\n');
    isStance = dataTable.gait_phase_fused == 'MidStance' | dataTable.gait_phase_fused == 'PushOff' | dataTable.gait_phase_fused == 'HeelStrike';
    cycle_starts = find(diff([0; isStance]) == 1);
    if isempty(cycle_starts) || length(cycle_starts) < 2
        fprintf('--- WARNING: Not enough gait cycles found to plot activation patterns.\n');
        return; 
    end
    
    pf_muscle_cols = config.pf_muscles;
    dataTable.Plantarflexors_envelope = mean(dataTable{:, pf_muscle_cols}, 2);
    dataTable.Plantarflexors_envelope_norm = (dataTable.Plantarflexors_envelope - min(dataTable.Plantarflexors_envelope)) / (max(dataTable.Plantarflexors_envelope) - min(dataTable.Plantarflexors_envelope));
    plot_labels = [emgLabels, {'Plantarflexors'}];

    for m = 1:length(plot_labels)
        muscle_label = plot_labels{m};
        if strcmp(muscle_label, 'Plantarflexors')
            norm_col_name = 'Plantarflexors_envelope_norm';
        else
            norm_col_name = ['sEMG_', muscle_label, '_envelope_norm'];
        end
        
        if ~ismember(norm_col_name, dataTable.Properties.VariableNames)
             env_col_name = strrep(norm_col_name, '_norm', '');
             dataTable.(norm_col_name) = (dataTable.(env_col_name) - min(dataTable.(env_col_name))) / (max(dataTable.(env_col_name)) - min(dataTable.(env_col_name)));
        end
        
        norm_time = linspace(0, 100, 101);
        resampled_cycles = [];
        for k = 1:length(cycle_starts)-1
            cycle_indices = cycle_starts(k):cycle_starts(k+1)-1;
            if isempty(cycle_indices), continue; end
            cycle_time = dataTable.time(cycle_indices);
            cycle_emg = dataTable.(norm_col_name)(cycle_indices);
            cycle_time_norm = (cycle_time - cycle_time(1)) / (cycle_time(end) - cycle_time(1)) * 100;
            resampled_cycles(k, :) = interp1(cycle_time_norm, cycle_emg, norm_time);
        end
        
        if isempty(resampled_cycles), continue; end
        mean_activation = mean(resampled_cycles, 1);
        std_activation = std(resampled_cycles, 0, 1);
        
        hFig = figure('Visible', 'off', 'Position', [100 100 800 500]);
        hold on;
        plot(norm_time, mean_activation, 'r-', 'LineWidth', 2.5, 'DisplayName', 'Mean Activation');
        fill([norm_time, fliplr(norm_time)], [mean_activation-std_activation, fliplr(mean_activation+std_activation)], 'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'DisplayName', 'Std. Dev.');
        hold off;
        
        title(['Normalized Activation: ', strrep(muscle_label, '_', '\_'), ' for ', strrep(trialName, '_', '\_')]);
        xlabel('Gait Cycle (%)'); ylabel('Normalized EMG Amplitude');
        xlim([0 100]); ylim([0 1]); grid on; legend;
        saveas(hFig, fullfile(outputDir, [trialName, '_activation_', muscle_label, '.png']));
        close(hFig);
    end
    fprintf('Activation plots saved.\n');
end

function runComparativeSynergyAnalysis(emgMatrix, emgLabels, time, fs, outputDir, trialName, vafThreshold)
    fprintf('\n--- Running Comparative Synergy Analysis ---\n');
    emgMatrixNorm = emgMatrix ./ max(emgMatrix, [], 1);
    maxSynergies = size(emgMatrixNorm, 2);
    
    [W_nmf, H_nmf, n_nmf, vaf_nmf, rmse_nmf] = performNMF(emgMatrixNorm, maxSynergies, vafThreshold);
    fprintf('NMF: Selected %d synergies. Final VAF = %.4f, RMSE = %.4f\n', n_nmf, vaf_nmf, rmse_nmf);
    
    [W_pca, H_pca, n_pca, vaf_pca, rmse_pca] = performPCA(emgMatrixNorm, maxSynergies, vafThreshold);
    fprintf('PCA: Selected %d components. Final VAF = %.4f, RMSE = %.4f\n', n_pca, vaf_pca, rmse_pca);
    
    [W_dmd, H_dmd, n_dmd] = performDMD(emgMatrixNorm, fs, maxSynergies);
    fprintf('DMD: Selected %d dynamic modes.\n', n_dmd);

    plotSynergyComparison(W_nmf, H_nmf, W_pca, H_pca, W_dmd, H_dmd, emgLabels, time, outputDir, trialName);
    
    synergyDataPath = fullfile(outputDir, [trialName, '_synergy_data.mat']);
    save(synergyDataPath, 'W_nmf', 'H_nmf', 'W_pca', 'H_pca', 'W_dmd', 'H_dmd', 'emgLabels');
    fprintf('Synergy matrix data saved to: %s\n', synergyDataPath);
end

function [W, H, n, final_vaf, final_rmse] = performNMF(matrix, max_n, vaf_thresh)
    vaf = zeros(1, max_n);
    for i = 1:max_n
        [W_temp, H_temp] = nnmf(matrix, i, 'Replicates', 5, 'Options', statset('Display','off'));
        vaf(i) = 1 - sum((matrix - W_temp*H_temp).^2, 'all') / sum(matrix.^2, 'all');
    end
    n = find(vaf >= vaf_thresh, 1, 'first');
    if isempty(n), n = max_n; end
    [W, H] = nnmf(matrix, n, 'Replicates', 10, 'Options', statset('Display','off'));
    reconstructed = W * H;
    final_vaf = vaf(n);
    final_rmse = sqrt(mean((matrix - reconstructed).^2, 'all'));
end

function [W, H, n, final_vaf, final_rmse] = performPCA(matrix, max_n, vaf_thresh)
    [coeff, score, ~, ~, explained] = pca(matrix);
    vaf = cumsum(explained) / sum(explained);
    n = find(vaf >= vaf_thresh, 1, 'first');
    if isempty(n), n = max_n; end
    W = score(:, 1:n);
    H = coeff(:, 1:n)';
    reconstructed = W * H;
    final_vaf = vaf(n);
    final_rmse = sqrt(mean((matrix - reconstructed).^2, 'all'));
end

function [W_dmd, H_dmd, n_dmd] = performDMD(matrix, fs, max_n)
    X1 = matrix(1:end-1, :)';
    X2 = matrix(2:end, :)';
    [U, S, V] = svd(X1, 'econ');
    A_tilde = U' * X2 * V / S;
    [eigVecs, ~] = eig(A_tilde);
    dmd_modes = X2 * V / S * eigVecs;
    n_dmd = max_n;
    H_dmd = dmd_modes(:, 1:n_dmd)';
    W_dmd = pinv(dmd_modes) * matrix';
    W_dmd = W_dmd(1:n_dmd, :)';
end

function plotSynergyComparison(W_nmf, H_nmf, W_pca, H_pca, W_dmd, H_dmd, emgLabels, time, outputDir, trialName)
    n_nmf = size(W_nmf, 2); n_pca = size(W_pca, 2); n_dmd = size(W_dmd, 2);
    hFig = figure('Visible', 'off', 'Position', [100 100 1200 900]);
    
    subplot(3,1,1);
    bar(H_nmf'); title('NMF Synergy Weights');
    set(gca, 'xtick', 1:length(emgLabels), 'xticklabel', emgLabels); xtickangle(45);
    ylabel('Weighting'); grid on;
    legend(arrayfun(@(x) sprintf('Synergy %d', x), 1:n_nmf, 'UniformOutput', false));
    
    subplot(3,1,2);
    bar(abs(H_pca')); title('PCA Synergy Weights (Absolute Value)');
    set(gca, 'xtick', 1:length(emgLabels), 'xticklabel', emgLabels); xtickangle(45);
    ylabel('Weighting'); grid on;
    legend(arrayfun(@(x) sprintf('Component %d', x), 1:n_pca, 'UniformOutput', false));

    subplot(3,1,3);
    bar(abs(real(H_dmd'))); title('DMD Synergy Weights (Absolute Real Part)');
    set(gca, 'xtick', 1:length(emgLabels), 'xticklabel', emgLabels); xtickangle(45);
    ylabel('Weighting'); grid on;
    legend(arrayfun(@(x) sprintf('Mode %d', x), 1:n_dmd, 'UniformOutput', false));
    
    sgtitle(['Synergy Weights Comparison for ', strrep(trialName, '_', '\_')]);
    saveas(hFig, fullfile(outputDir, [trialName, '_synergy_weights_comparison.png']));
    close(hFig);
    fprintf('Synergy weights comparison plot saved.\n');
    
    hFig = figure('Visible', 'off', 'Position', [100 100 1600 800]);
    max_rows = max([n_nmf, n_pca, n_dmd]);
    for n = 1:max_rows
        if n <= n_nmf
            subplot(max_rows, 3, 3*(n-1)+1);
            plot(time, W_nmf(:,n), 'r-'); grid on; xlim([time(1) time(end)]);
            title(['NMF Activation ' num2str(n)]);
        end
        if n <= n_pca
            subplot(max_rows, 3, 3*(n-1)+2);
            plot(time, W_pca(:,n), 'b-'); grid on; xlim([time(1) time(end)]);
            title(['PCA Activation ' num2str(n)]);
        end
        if n <= n_dmd
            subplot(max_rows, 3, 3*(n-1)+3);
            plot(time, real(W_dmd(:,n)), 'g-'); grid on; xlim([time(1) time(end)]);
            title(['DMD Activation ' num2str(n)]);
        end
    end
    sgtitle(['Synergy Activations Comparison for ', strrep(trialName, '_', '\_')]);
    saveas(hFig, fullfile(outputDir, [trialName, '_synergy_activations_comparison.png']));
    close(hFig);
    fprintf('Synergy activations comparison plot saved.\n');
end
