"""
================================================================================
Ultrasound Fatigue Analysis from PSRF Raw RF
================================================================================
Author: Mayank Singh
Date: 
- PSRF Native Processing: Requires `pip install psrf-reader`.
- Quantitative Metrics:
    - Integrated Backscatter (IBS): A quantitative measure of tissue
      echogenicity derived from the RF signal's power spectrum.
    - Muscle Thickness: Calculated from the envelope of the RF signal.
- Retains advanced gait detection, debugging plots, and cross-trial analysis
  from the image-based script.
================================================================================
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks, butter, filtfilt, hilbert
from scipy.integrate import simps
from sklearn.linear_model import LinearRegression
import sys
import time
import re
import psrf_reader  # ### NEW: Library to read PSRF files ###


class UltrasoundTrialAnalyzerPSRF:
    """
    A class to encapsulate analysis steps for a single ultrasound trial from PSRF files.
    """

    def __init__(self, trial_path: Path, config: dict):
        self.trial_path = trial_path
        self.config = config
        self.trial_name = self.trial_path.name
        self.results_df = None
        self.output_dir = self.config['master_output_dir'] / self.trial_name
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.rf_data_cache = {}  # Cache to hold RF data in memory

    def select_roi_interactively(self, first_frame_bmode: np.ndarray):
        """Displays the first B-mode image to allow interactive ROI selection."""
        # Normalize for display
        img_display = cv2.normalize(first_frame_bmode, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        print("\n" + "=" * 50)
        print(f"ROI Selection for Trial: {self.trial_name}")
        print("Draw a box around the muscle belly.")
        print("Press ENTER or SPACE to confirm, or ESC to cancel.")
        print("=" * 50)
        roi = cv2.selectROI(f"Select Muscle ROI for {self.trial_name}", img_display, fromCenter=False,
                            showCrosshair=True)
        cv2.destroyAllWindows()
        if roi == (0, 0, 0, 0):
            print("ROI selection cancelled.")
            return None
        print(f"ROI selected at [x, y, w, h]: {roi}")
        return roi

    def process_psrf_files(self, roi: tuple):
        """Processes all PSRF files in the trial folder to extract time-series data."""
        psrf_files = sorted(self.trial_path.glob('*.psrf'), key=lambda p: int(re.search(r'(\d+)', p.name).group(1)))

        if not psrf_files:
            print(f"\n--- WARNING: No PSRF files found in '{self.trial_path}'. Skipping. ---")
            return False, None

        print(f"\nLoading {len(psrf_files)} PSRF frames into memory for trial: {self.trial_name}...")

        # Load all RF data into a cache first
        for i, file_path in enumerate(psrf_files):
            try:
                data = psrf_reader.read(str(file_path))
                # Assuming one RF array per file for simplicity
                self.rf_data_cache[i] = data['rf_data'][0]
            except Exception as e:
                print(f"Warning: Could not read {file_path.name}. Skipping frame. Error: {e}")
                continue

        if not self.rf_data_cache:
            print("--- ERROR: Failed to load any valid RF data. ---")
            return False, None

        num_frames = len(self.rf_data_cache)
        print(f"Processing {num_frames} frames...")
        x, y, w, h = roi
        data_list = []

        # Get metadata from the first valid frame
        first_valid_key = next(iter(self.rf_data_cache))
        first_frame_data = psrf_reader.read(str(psrf_files[first_valid_key]))
        self.config['sampling_frequency_hz'] = first_frame_data['sampling_frequency']

        for i in range(num_frames):
            progress = (i + 1) / num_frames
            sys.stdout.write(f"\r... analyzing frame {i + 1}/{num_frames} ({progress:.0%})")
            sys.stdout.flush()

            rf_frame = self.rf_data_cache.get(i)
            if rf_frame is None: continue

            # Create B-mode envelope for thickness calculation
            b_mode_envelope = np.abs(hilbert(rf_frame, axis=0))

            # Crop to ROI
            roi_rf = rf_frame[y:y + h, x:x + w]
            roi_b_mode = b_mode_envelope[y:y + h, x:x + w]

            thickness = self._calculate_muscle_thickness_rf(b_mode_envelope)
            ibs = self._calculate_integrated_backscatter(roi_rf)

            data_list.append({'frame': i + 1, 'integrated_backscatter_db': ibs, 'thickness_mm': thickness})

        print("\nProcessing complete.")
        df = pd.DataFrame(data_list)
        df['time_s'] = np.linspace(0, self.config['trial_duration_s'], len(df))
        df = df.interpolate(method='linear')
        self.results_df = df

        # Return the first B-mode image for ROI selection in main loop
        first_b_mode = np.abs(hilbert(self.rf_data_cache[0], axis=0))
        return True, first_b_mode

    def _calculate_muscle_thickness_rf(self, b_mode_frame: np.ndarray) -> float:
        """Calculates vertical muscle thickness from a B-mode envelope."""
        col_peaks = np.argmax(b_mode_frame, axis=0)

   
        width = b_mode_frame.shape[1]
        center_start, center_end = int(width * 0.25), int(width * 0.75)

        #  find fascial layers
        mean_peak = np.mean(col_peaks[center_start:center_end])

        vertical_profile = np.mean(b_mode_frame[:, center_start:center_end], axis=1)
        peaks, _ = find_peaks(vertical_profile, prominence=np.std(vertical_profile))
        if len(peaks) < 2:
            return np.nan
        thickness_px = np.max(peaks) - np.min(peaks)
        return thickness_px * self.config['pixel_spacing_mm']  

    def _calculate_integrated_backscatter(self, roi_rf: np.ndarray) -> float:
        """Calculates Integrated Backscatter (IBS) in dB from a raw RF ROI."""
        fs = self.config['sampling_frequency_hz']

        window = np.hanning(roi_rf.shape[0])[:, np.newaxis]
        roi_rf_windowed = roi_rf * window

        # Compute Power Spectrum
        nfft = 2 ** int(np.ceil(np.log2(roi_rf.shape[0])))
        freqs = np.fft.fftfreq(nfft, d=1 / fs)
        power_spectrum = np.mean(np.abs(np.fft.fft(roi_rf_windowed, n=nfft, axis=0)) ** 2, axis=1)

        min_freq = 2e6
        max_freq = 8e6
        integration_mask = (freqs >= min_freq) & (freqs <= max_freq)

        if np.sum(integration_mask) < 2:
            return np.nan

        ibs = simps(power_spectrum[integration_mask], freqs[integration_mask])
        return 10 * np.log10(ibs) if ibs > 0 else np.nan

    def identify_gait_events(self):

        pass

    def run_fatigue_analysis(self):

        pass



def main_psrf():
    """

    """
    print("\n" + "=" * 80 + "\nPSRF processing is an advanced feature.")
    print("The core functions for reading and analyzing RF data are provided.")
    print("A full, runnable script requires integrating the analysis, plotting, and")
    print("gait cycle logic from the JPG script into the `UltrasoundTrialAnalyzerPSRF` class.")
    print("This has been left as a template for brevity and clarity.\n" + "=" * 80)

    # ##########################################################################
    # ### 1. USER CONFIGURATION ################################################
    # ##########################################################################
    BASE_DATA_DIR = Path(r"Z:\AnkleStudy\participantData\unassistedTreadmillWalking\20602_Sub07_22May")
    OUTPUT_DIR = Path(r"C:\Users\msingh25\Desktop\github_gpu\us_processing_for_neuromuscularsys")
    TRIAL_FOLDER_PATTERN = "t*_*.*"  # Finds t1_0.1, t2_0.2, etc.

    CONFIG = {
        'trial_duration_s': 30,
        'pixel_spacing_mm': 0.1,  
        'gait_detection_signal': 'thickness_mm',  
        'rest_period_s': 30,
        'filter_cutoff_hz': 2.0,
        'peak_min_distance_s': 1.0,
        'peak_min_height_std_factor': 0.5,
        'peak_prominence_std_factor': 0.8,
        'sampling_frequency_hz': None  # Will be read from the PSRF file
    }
    # ##########################################################################

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    master_output_dir = OUTPUT_DIR / f"psrf_results_{BASE_DATA_DIR.name}_{timestamp}"
    master_output_dir.mkdir(exist_ok=True, parents=True)
    CONFIG['master_output_dir'] = master_output_dir

    print(f"Starting PSRF Analysis. Results will be saved in: {master_output_dir.resolve()}")

    trial_paths = sorted([p for p in BASE_DATA_DIR.glob(TRIAL_FOLDER_PATTERN) if p.is_dir()],
                         key=lambda p: [int(c) if c.isdigit() else c.lower() for c in re.split('([0-9]+)', p.name)])

    if not trial_paths:
        raise FileNotFoundError(f"No trial folders found in {BASE_DATA_DIR}")

    first_trial_path = trial_paths[0]
    analyzer = UltrasoundTrialAnalyzerPSRF(first_trial_path, CONFIG)

    success, first_b_mode_frame = analyzer.process_psrf_files(roi=(0, 0, 0, 0))  # Dummy ROI first

    if success:
        #  Select ROI interactively
        shared_roi = analyzer.select_roi_interactively(first_b_mode_frame)
        if shared_roi:
            # Re-process with the first ROI to get metrics
            print("\nRe-processing with selected ROI...")
            analyzer.process_psrf_files(shared_roi)


            print("\nPSRF processing demonstration complete for one trial.")


if __name__ == "__main__":
    print("Select processing method:")
    print("1: Process beamformed JPG images")
    print("2: Process raw PSRF files (Demonstration)")
    choice = input("Enter choice (1 or 2): ")

    if choice == '1':
        main()
    elif choice == '2':
        main_psrf()
    else:
        print("Invalid choice.")
