% beamformToImages_GPU.m

%% Start
clear
close all
clc

%% <<-- GPU CHECK -->>
% Check if a compatible GPU is available and select it.
if gpuDeviceCount < 1
    error('No compatible GPU detected. This script requires a GPU and the Parallel Computing Toolbox.');
end
gpu = gpuDevice(1);
fprintf('GPU Detected: %s\n', gpu.Name);


%% Import
folder = 'Z:\AnkleStudy\participantData\unassistedTreadmillWalking\20602_Sub17_16June\t8_0.8';
dataFilesPrefix = 'KMPC_';
imageFolder = fullfile(folder, 'images_gpu'); % Separate folder for GPU output
vidFile = fullfile(imageFolder, 'video_gpu.mp4');

if ~exist(imageFolder, 'dir')
   mkdir(imageFolder);
end

%% Declarations
% Ultrasound parameters
soundSpeed = 1540.0 * 1e3; % (mm/s)
fs = 20.0 * 1e6;          % (Hz)
elements = 128;
samples = 1264;
minDepth = 1.7112;        % (mm)
maxDepth = 50;            % (mm)
pitch = 0.46;             % (mm)

% Measurement parameters
files = 60;
frames_per_file = 496;
videoFrameRate = 60;      % Use a standard video frame rate

% Planewave beamforming parameters
fnum = 3;
apo = 'hamming';

%% Calculations
% These calculations are done on the CPU first
lat_cpu = ((0:elements-1) - (elements-1)/2) * pitch;
maxSample = round(maxDepth * 2 / soundSpeed * fs) + 1;
axial_cpu = minDepth + (0:maxSample-1) * (soundSpeed / fs / 2);

%% <<-- Move Coordinate Data to GPU -->>
% Transfer coordinate arrays to the GPU once before the loop starts.
lat = gpuArray(lat_cpu);
axial = gpuArray(axial_cpu);

%% Main Processing Loop
vWriter = VideoWriter(vidFile, 'MPEG-4');
vWriter.FrameRate = videoFrameRate;
open(vWriter);

totalFrameCount = 0;

% <<-- ADDITION: Start the timer -->>
tic; 

for f = 0:(files - 1)
    
    fileName = sprintf('%s%d.psrf', dataFilesPrefix, f);
    filePath = fullfile(folder, fileName);
    
    fprintf('Processing file: %s\n', filePath);
    
    if ~exist(filePath, 'file')
        warning('File not found: %s. Skipping.', filePath);
        continue;
    end
    
    fid = fopen(filePath, 'r');
    if fid == -1
        warning('Could not open file: %s. Skipping.', filePath);
        continue;
    end
    
    rawDataLinear = fread(fid, 'int16=>double');
    fclose(fid);
    
    expected_elements = frames_per_file * elements * samples;
    if numel(rawDataLinear) ~= expected_elements
        warning('File %s has unexpected size. Expected %d, got %d. Skipping.', ...
                fileName, expected_elements, numel(rawDataLinear));
        continue;
    end
    
    rawData_cpu = reshape(rawDataLinear, [samples, elements, frames_per_file]);

    for i = 1:frames_per_file
        totalFrameCount = totalFrameCount + 1;
        
        % Get the current frame's data from the CPU matrix
        currentFrame_cpu = rawData_cpu(:, :, i);

        %% <<-- GPU ACCELERATION HAPPENS HERE -->>
        % 1. Move the single frame of raw data to the GPU
        rawData_gpu = gpuArray(currentFrame_cpu);
        
        % 2. Call the GPU-optimized beamforming function
        IQEnvelope_gpu = planewaveBeamform_gpu(rawData_gpu, lat, axial, fnum, pitch, apo);
        
        % 3. Perform log compression on the GPU
        imgLog_gpu = 20 * log10(abs(IQEnvelope_gpu));
        
        % 4. Gather the final image from the GPU back to the CPU for display/saving
        imgLog_cpu = gather(imgLog_gpu);
        
        % --- Post-processing and saving (on CPU) ---
        dynamicRange = 40;
        maxVal = max(imgLog_cpu(:));
        cLim = [maxVal - dynamicRange, maxVal];
        
        fig = figure('Visible', 'off');
        imagesc(lat_cpu, axial_cpu, imgLog_cpu, cLim);
        axis image; colormap(gray); colorbar;
        title(sprintf('Frame %d', totalFrameCount));

        imagePath = fullfile(imageFolder, sprintf('frame_%05d.jpg', totalFrameCount));
        print(fig, imagePath, '-djpeg', '-r150');
        close(fig);

        frameImage = imread(imagePath);
        writeVideo(vWriter, frameImage);
        
        fprintf('Processed Frame: %d / %d\n', totalFrameCount, files * frames_per_file);
    end
end

close(vWriter);
disp('GPU Video processing complete.');

% <<-- ADDITION: Stop the timer and display the elapsed time -->>
elapsedTime = toc;
fprintf('Total processing time: %.2f seconds.\n', elapsedTime);


%%
function [envOut] = planewaveBeamform_gpu(rawData, lat, axial, fnum, pitch, apotype)
% planewaveBeamform_gpu - Fully vectorized GPU implementation
%
% This function performs all calculations on the GPU using matrix operations
% to achieve high parallelism and speed.

    [nSamples, nElements] = size(rawData);
    nLines = nElements;
    dAx = mean(diff(axial));

    % --- Dynamic Aperture (on GPU) ---
    d = axial / fnum;
    d = round(d / pitch);
    d(d < 1) = 1;
    d(d > nElements) = nElements;

    % --- Apodization Window (calculated on GPU) ---
    % Create apodization windows for all possible aperture sizes
    apo_lookup = cell(1, nElements);
    switch apotype
        case 'hamming'
            for k = 1:nElements, apo_lookup{k} = gpuArray(hamming(k)); end
        case 'hanning'
            for k = 1:nElements, apo_lookup{k} = gpuArray(hanning(k)); end
        case 'none'
            for k = 1:nElements, apo_lookup{k} = gpuArray(ones(k, 1)); end
    end
    
    % --- Vectorized Delay-and-Sum ---
    % Pre-allocate the final RF data matrix on the GPU
    rfData = gpuArray.zeros(length(axial), nLines);

    % Create grids of coordinates using bsxfun for efficient matrix expansion.
    % bsxfun is a powerful tool for vectorization.
    axial_grid = bsxfun(@plus, axial', zeros(1, nElements));
    lat_grid = bsxfun(@plus, lat, zeros(length(axial), 1));
    
    % Loop through each scanline (this outer loop is small)
    for lcv = 1:nLines
        
        % Calculate receive delays for ALL points in the image relative to ALL elements
        % This creates a [nAxial x nElements] matrix of delays.
        delay_rx = sqrt(axial_grid.^2 + (lat_grid - lat(lcv)).^2);
        delay_total_mm = delay_rx + axial_grid; % Add transmit delay
        delay_samples = round(delay_total_mm / (dAx * 2));

        % Create a mask for out-of-bounds samples
        mask = (delay_samples > nSamples) | (delay_samples < 1);
        delay_samples(mask) = 1; % Set invalid indices to 1 to prevent error

        % --- Vectorized Data Pull ---
        % Create a linear index to pull all required data points from rawData in one go
        % This is the most critical vectorization step.
        linear_indices = sub2ind(size(rawData), delay_samples, ...
            bsxfun(@plus, zeros(size(delay_samples,1),1), 1:nElements));
        
        summed_data = rawData(linear_indices);
        summed_data(mask) = 0; % Zero out the contributions from invalid delays

        % --- Vectorized Apodization (Advanced) ---
        % A full dynamic apodization is complex to vectorize perfectly.
        % Here, we use a fixed aperture based on the f-number at the center depth.
        center_idx = round(length(axial)/2);
        aperture_size = d(center_idx);
        apo_win = apo_lookup{aperture_size};
        
        % Find which elements fall into the aperture for this scanline
        start_el = max(1, lcv - floor(aperture_size/2));
        end_el = min(nElements, start_el + aperture_size - 1);
        
        % Apply the window to the valid elements
        temp_win = gpuArray.zeros(1, nElements);
        temp_win(start_el:end_el) = apo_win(1:length(start_el:end_el));
        
        % Apply apodization and sum across all elements in a single matrix operation
        rfData(:, lcv) = sum(summed_data .* temp_win, 2);
    end
    
    % Perform Hilbert transform on the GPU
    envOut = hilbert(rfData);
end
