%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Mayank Singh
% 12 October 2023
% Planewave beamforming
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [envOut, rfData, lat, axial] = planewaveBeamform(rawData, lat, axial, fnum, pitch, apotype)
%PLANWAVEBEAMFORM GPU‑accelerated planewave beamforming
%
% Inputs:
%   rawData : [nSamples × nElements] RF data
%   lat     : [1 × nElements] lateral positions (mm)
%   axial   : [nAxial × 1] depth samples (mm)
%   fnum    : compounding factor (3–10)
%   pitch   : element pitch (mm), if empty uses mean(diff(lat))
%   apotype : 'hamming' | 'hanning' | 'none'
%
% Outputs:
%   envOut  : Hilbert envelope of size [nAxial × nLines]
%   rfData  : beamformed RF data
%   lat, axial returned (gathered back if GPU used)

%% — Move everything onto GPU if avaiable
if gpuDeviceCount > 0
    rawData = gpuArray(rawData);
    lat     = gpuArray(lat);
    axial   = gpuArray(axial);
end

%% — Basic geometry
[nSamp, nElem] = size(rawData);
nLines         = nElem;
dLat           = mean(diff(lat));
dAx            = mean(diff(axial));
if isempty(pitch), pitch = dLat; end

%% — Channels per depth (rounded and clamped)
d = round((axial / fnum) / pitch);
d = min(max(d,1), nElem);   % ensure 1 ≤ d ≤ nElem

%% — Precompute apodization windows once
apo = cell(1,nElem);
for k = 1:nElem
    switch apotype
      case 'hamming', apo{k} = hamming(d(k));
      case 'hanning', apo{k} = hanning(d(k));
      otherwise      apo{k} = ones(d(k),1);
    end
    if gpuDeviceCount>0, apo{k} = gpuArray(apo{k}); end
end

%% — Beamforming loop (still recursive, but all math on GPU)
rfData = zeros(numel(axial), nLines, 'like', rawData);
for lineIdx = 1:nLines
    sumOut = zeros(numel(axial),1, 'like', rawData);
    for ch = 1:nElem
        % compute sample‐by‐sample delay
        relDist = abs(lat(ch) - lat(lineIdx));
        if relDist < pitch/4, relDist = 0; end
        delays = round( (axial + sqrt(axial.^2 + relDist^2)) / (2*dAx) );
        delays = min(max(delays,1), nSamp);   % clamp indices

        % pull samples and apply apodization
        idx = sub2ind([nSamp nElem], delays, ch*ones(size(delays)));
        sumOut = sumOut + rawData(idx) .* apo{ch};
    end
    rfData(:,lineIdx) = sumOut;
end

%% — Envelope via Hilbert transform
envOut = hilbert(rfData);

%% — Bring results back to CPU
if gpuDeviceCount > 0
    envOut = gather(envOut);
    rfData = gather(rfData);
    lat     = gather(lat);
    axial   = gather(axial);
end
end
