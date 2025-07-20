function [envOut, rfData, lat, axial] = planewaveBeamform(rawData, lat, axial, fnum, pitch, apotype)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Jaesok Yu
% 19 October 2016
% Planewave beamforming
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Inputs
% rawData = raw US data
% lat = lateral elements array
% axial = axial samples array
% fNum = between 3 and 10
% pitch = distance between adjacent elements
% apotype = apodization window type (hamming, hanning, or none)

%% Outputs
% envOut = envelope
% rfData = radio frequency data
% lat = lateral elements array
% axial = axial samples array

%% Calculations
[nSamples, nElements] = size(rawData);
nLines = nElements;

dLat = mean(diff(lat)); % spacing between elements (mm)
dAx = mean(diff(axial)); % spacing between samples (mm)

if isempty(pitch)
    pitch = dLat; % (mm)
end

ch1_pos = lat;

% find number of elements at every given depth
d = axial / fnum;
d = round(d / pitch); % converting from mm to elements

mask = d < 1;
d(mask) = 1;

mask = d > nElements;
d(mask) = nElements;

% make the apodization window
nLat = length(lat);
nAx = length(axial);
apoWindow = zeros(nAx, nElements, nLat);
apoLookup = struct([]);

switch apotype
    case 'hamming'
        for k = 1:nElements
            apoLookup(k).window = hamming(k);
        end
        
    case 'hanning'
        for k = 1:nElements
            apoLookup(k).window = hanning(k);
        end
    case 'none'
        for k = 1:nElements
            apoLookup(k).window = ones(1,k);
        end
end

for k = 1:nLat
    
    % find the middle element
    [~, middle] = min(abs(ch1_pos - lat(k)));
    
    nLeft = round((d - 1) / 2);
    indexL = middle - nLeft;
    
    nRight = d - 1 - nLeft;
    indexR = middle + nRight;
    
    mask = indexL < 1;
    indexR(mask) = indexR(mask) + 1 - indexL(mask);
    
    mask = indexR > nElements; % 128, 64
    indexL(mask) = indexL(mask) - indexR(mask) + nElements; % 128, 64
    
    mask = indexL < 1;
    indexL(mask) = 1;
    
    mask = indexR > nElements; % 128, 64;
    indexR(mask) = nElements; % 128, 64
    
    for j = 1:nAx
        len = length(indexL(j):indexR(j));
        apoWindow(j, indexL(j):indexR(j), k) ...
            = apoLookup(len).window;
    end
end

% allocate space
rfData = zeros(nAx, nLines);

%loop through the transmit lines
for lcv = 1:nLines
    
    data = rawData;
    sumOut = zeros(nAx, 1);
    
    % loop through each receive channel
    for k = 1:nElements

        % calculate position of the current channel wrt scanline of interest
        rel_dist = ch1_pos(k) - lat(lcv);
        
        if abs(rel_dist) < (pitch / 4)
            rel_dist = 0;
        end
        
        rel_dist = rel_dist * ones(1, nAx);
        
        
        % transmit distance      
        dx = 0; % (mm)
        delay_tx = (dx * ones(1, nAx)  + axial); % (mm)
        
        % receive distance
        delay_rx = sqrt(axial.^2 + rel_dist.^2);
        
        % total delay
        delay = (delay_tx + delay_rx) / (dAx * 2);
        indexD = round(delay);    

        % find delays that are larger than range of samples
        mask = indexD > nSamples | indexD < 1;
        
        % pull the samples of interest and add it the delays
        apo_temp = apoWindow(:, k, lcv);
        sumOut(~mask) = sumOut(~mask) + data(indexD(~mask), k).* apo_temp(~mask);
    end
    
    rfData(:, lcv) = sumOut;
end

envOut = hilbert(rfData);

% env_log = 20*log10(abs(env_out)./max(abs(env_out(:))));

% figure(1);
% imagesc(lat,axial,env_log, [-50 0]);
% colorbar;
% colormap(gray);
% axis image;

