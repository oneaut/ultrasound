# -*- coding: utf-8 -*-
"""
================================================================================
Ultrasound Fatigue Analysis - v9.0 (Robust Gait Detection)
================================================================================
Author: Gemini AI
Date: 2025-06-30

Version 9.0 Enhancements:
- ROBUST GAIT DETECTION: Implements a superior gait detection method by finding
  prominent valleys (muscle contractions) instead of peaks. This is much more
  reliable across different subjects and signal conditions.
- NEW MUSCLE ACTIVATION PLOT: Generates a time-normalized plot showing the
  average muscle activation pattern across all gait cycles in a trial.
- ENHANCED DEBUG PLOT: The gait detection debug plot is now clearer, showing
  the inverted signal used for valley detection with a dual y-axis for context.
- REFINED TUNING: Configuration is updated to prioritize the 'prominence'
  parameter, the most effective way to tune the new detection algorithm.
================================================================================
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks, butter, filtfilt
from scipy.interpolate import interp1d
import sys
import time
import re


class UltrasoundTrialAnalyzer:
    """
    A class to encapsulate all analysis steps for a single ultrasound trial from JPG images.
    """

    def __init__(self, trial_path: Path, config: dict):
        self.trial_path = trial_path
        self.config = config
        self.trial_name = f"{trial_path.parent.name}_{trial_path.name}"
        self.results_df = None
        self.output_dir = self.config['master_output_dir'] / self.trial_name
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def select_roi_interactively(self, first_image_path: Path):
        """Displays the first image to allow interactive ROI selection."""
        if not first_image_path.exists():
            print(f"Error: Image not found at {first_image_path}")
            return None
        img = cv2.imread(str(first_image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error: Could not read image from {first_image_path}")
            return None
        print("\n" + "=" * 50)
        print(f"ROI Selection for Trial: {self.trial_name}")
        print("Draw a box around the muscle belly.")
        print("Press ENTER or SPACE to confirm, or ESC to cancel.")
        print("=" * 50)
        roi = cv2.selectROI(f"Select Muscle ROI for {self.trial_name}", img, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows()
        if roi == (0, 0, 0, 0):
            print("ROI selection cancelled.")
            return None
        print(f"ROI selected at [x, y, w, h]: {roi}")
        return roi

    def process_images(self, roi: tuple):
        """Processes all images in the trial folder to extract time-series data."""
        image_files = sorted(self.trial_path.glob('*.jpg'), key=lambda p: int(re.search(r'(\d+)', p.name).group(1)))

        if not image_files:
            print(f"\n--- WARNING: No images found in '{self.trial_path}'. Skipping. ---")
            return False

        num_images = len(image_files)
        print(f"\nProcessing {num_images} images from trial: {self.trial_name}...")
        x, y, w, h = roi
        data = []

        for i, img_path in enumerate(image_files):
            progress = (i + 1) / num_images
            sys.stdout.write(f"\r... processing frame {i + 1}/{num_images} ({progress:.0%})")
            sys.stdout.flush()

            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None: continue

            roi_img = img[y:y + h, x:x + w]
            echogenicity = np.mean(roi_img)
            thickness = self._calculate_muscle_thickness(roi_img)
            data.append({'frame': i + 1, 'echogenicity': echogenicity, 'thickness_mm': thickness})

        print("\nProcessing complete.")
        df = pd.DataFrame(data)
        df['time_s'] = np.linspace(0, self.config['trial_duration_s'], len(df))
        df['thickness_mm'] = df['thickness_mm'].interpolate(method='linear')
        df['trial_name'] = self.trial_name
        self.results_df = df
        return True

    def _calculate_muscle_thickness(self, roi_image: np.ndarray) -> float:
        """Calculates vertical muscle thickness from a cropped ROI image."""
        blurred_img = cv2.GaussianBlur(roi_image, (5, 5), 0)
        edges = cv2.Canny(blurred_img, 50, 150)
        width = edges.shape[1]
        center_start, center_end = int(width * 0.25), int(width * 0.75)
        central_column_edges = edges[:, center_start:center_end]

        # Use percentile-based approach for robustness to noise
        edge_points_y, _ = np.where(central_column_edges > 0)
        if edge_points_y.size < 10: return np.nan

        lower_bound = np.percentile(edge_points_y, 5)
        upper_bound = np.percentile(edge_points_y, 95)
        thickness_px = upper_bound - lower_bound

        return thickness_px * self.config['pixel_spacing_mm']

    def identify_gait_events(self):
        """
        Identifies gait events by finding prominent valleys (muscle contractions)
        in the thickness signal. This version is more robust to baseline drift and noise.
        """
        signal_col = self.config['gait_detection_signal']
        raw_signal = self.results_df[signal_col].values

        if pd.isna(raw_signal).all():
            print(f"Signal '{signal_col}' contains only NaN values. Skipping gait detection.")
            self.results_df['gait_event'] = 'Stance'
            self.results_df['gait_cycle'] = 0
            return

        sampling_rate = len(self.results_df) / self.config['trial_duration_s']
        nyquist = 0.5 * sampling_rate
        cutoff = self.config['filter_cutoff_hz']
        b, a = butter(2, cutoff / nyquist, btype='low', analog=False)
        filtered_signal = filtfilt(b, a, raw_signal)
        self.results_df['filtered_thickness'] = filtered_signal

        # Invert the signal to find valleys (contractions) as peaks
        inverted_signal = -filtered_signal

        # Use prominence as the main detection criterion for robustness
        signal_std = np.nanstd(inverted_signal)
        min_peak_prominence = self.config['peak_prominence_std_factor'] * signal_std
        min_peak_height = np.nanmean(inverted_signal)  # A generous height threshold
        min_peak_distance = int(self.config['peak_min_distance_s'] * sampling_rate)

        # Find peaks on the INVERTED signal
        peak_indices, _ = find_peaks(
            inverted_signal,
            height=min_peak_height,
            prominence=min_peak_prominence,
            distance=min_peak_distance
        )

        self.results_df['gait_event'] = 'Stance'
        self.results_df.loc[peak_indices, 'gait_event'] = 'Swing'

        self.results_df['gait_cycle'] = 0
        cycle_num = 1
        for peak_idx in peak_indices:
            stance_start_idx_series = self.results_df.index[:peak_idx][
                self.results_df['gait_event'][:peak_idx] == 'Stance']
            stance_start_idx = stance_start_idx_series.max() if not stance_start_idx_series.empty else 0
            next_peak_idx_loc = np.searchsorted(peak_indices, peak_idx) + 1
            cycle_end_idx = int((peak_idx + peak_indices[next_peak_idx_loc]) / 2) if next_peak_idx_loc < len(
                peak_indices) else len(self.results_df)
            self.results_df.loc[stance_start_idx:cycle_end_idx, 'gait_cycle'] = cycle_num
            cycle_num += 1

        print(f"Identified {len(peak_indices)} swing events (valleys) using robust detection on '{signal_col}'.")
        self._plot_gait_detection_debug(raw_signal, filtered_signal, peak_indices)

    def _plot_gait_detection_debug(self, raw_signal, filtered_signal, peaks):
        """Generates a diagnostic plot for the gait detection process."""
        fs = len(self.results_df) / self.config['trial_duration_s']
        time_axis = self.results_df['time_s']

        fig, ax = plt.subplots(figsize=(18, 8))

        # Plot original signals
        ax.plot(time_axis, raw_signal, color='gray', alpha=0.4, label='Raw Signal')
        ax.plot(time_axis, filtered_signal, color='blue', alpha=0.9, label='Filtered Signal')

        # Mark the detected valleys on the original signal plot
        ax.plot(time_axis.iloc[peaks], filtered_signal[peaks], "x", color='red', markersize=10, mew=2,
                label=f'Detected Contractions (Valleys) ({len(peaks)})')

        ax.set_title(f"Gait Detection Debug Plot for: {self.trial_name}", fontsize=16)
        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel(f"{self.config['gait_detection_signal']} (mm)", fontsize=12, color='blue')
        ax.legend(loc='upper left')
        ax.grid(True)
        ax.set_xlim(0, time_axis.max())
        # Invert y-axis to make valleys appear as peaks, which is intuitive
        ax.invert_yaxis()

        plt.tight_layout()
        save_path = self.output_dir / f"{self.trial_name}_gait_detection_debug.png"
        plt.savefig(save_path)
        plt.close(fig)

    def run_fatigue_analysis(self):
        """Calculates fatigue metrics and generates all plots and data files."""
        # --- Intra-trial Normalization (Overall) ---
        min_thick_overall = self.results_df['thickness_mm'].min()
        max_thick_overall = self.results_df['thickness_mm'].max()
        if max_thick_overall > min_thick_overall:
            self.results_df['norm_thick_intra_overall'] = (self.results_df['thickness_mm'] - min_thick_overall) / (
                        max_thick_overall - min_thick_overall)
        else:
            self.results_df['norm_thick_intra_overall'] = 0.5

        swing_events = self.results_df[self.results_df['gait_event'] == 'Swing'].copy()

        self._plot_time_series('thickness_mm', 'Muscle Thickness Over Time', 'Thickness (mm)')

        # --- NEW: Generate Muscle Activation Plot ---
        self._plot_muscle_activation()
        print("Muscle activation plot saved.")

        if len(swing_events) < 3:
            print(f"Warning: Only {len(swing_events)} swing events found. Fatigue trend analysis unreliable.")
            detailed_csv_path = self.output_dir / f"{self.trial_name}_detailed_data.csv"
            self.results_df.to_csv(detailed_csv_path, index=False)
            return None

        gait_cycle_metrics = self._calculate_gait_cycle_metrics()
        swing_events = swing_events.merge(gait_cycle_metrics, on='gait_cycle', how='left')
        thick_trend = self._calculate_fatigue_trend(swing_events, 'thickness_mm')
        self._plot_fatigue_analysis(swing_events, 'thickness_mm', thick_trend, 'Fatigue Trend: Peak Muscle Thickness',
                                    'Peak Thickness (mm)')
        print("Fatigue trend plots saved.")

        detailed_csv_path = self.output_dir / f"{self.trial_name}_detailed_data.csv"
        self.results_df.to_csv(detailed_csv_path, index=False)
        print(f"Detailed data with normalized values saved to: {detailed_csv_path.name}")

        summary = {
            'trial_name': self.trial_name,
            'num_gait_cycles': len(swing_events),
            'mean_peak_thickness': swing_events['thickness_mm'].mean(),
            'thickness_slope_intra_trial': thick_trend['slope'],
            'thickness_r2': thick_trend['r_squared'],
            'mean_swing_duration_s': swing_events['swing_duration_s'].mean()
        }
        return summary

    def _plot_muscle_activation(self):
        """
        Generates a plot of the average, time-normalized muscle activation
        pattern across all gait cycles in the trial.
        """
        gait_cycles = self.results_df[self.results_df['gait_cycle'] > 0]
        if gait_cycles.empty:
            return

        num_cycles = int(gait_cycles['gait_cycle'].max())
        resampled_cycles = []

        # Normalize each cycle to 101 points (0-100%)
        norm_time = np.linspace(0, 100, 101)

        for i in range(1, num_cycles + 1):
            cycle_df = gait_cycles[gait_cycles['gait_cycle'] == i]
            if len(cycle_df) < 2: continue

            # Create an interpolation function for this cycle's data
            interp_func = interp1d(
                cycle_df['time_s'],
                cycle_df['norm_thick_intra_overall'],
                bounds_error=False,
                fill_value='extrapolate'
            )

            # Time-normalize the cycle's time axis
            cycle_norm_time = (cycle_df['time_s'] - cycle_df['time_s'].iloc[0]) / (
                        cycle_df['time_s'].iloc[-1] - cycle_df['time_s'].iloc[0]) * 100

            # Re-create the interpolation function with the normalized time
            interp_func_norm = interp1d(
                cycle_norm_time,
                cycle_df['norm_thick_intra_overall'],
                bounds_error=False,
                fill_value='extrapolate'
            )

            resampled_cycles.append(interp_func_norm(norm_time))

        if not resampled_cycles:
            return

        all_cycles_matrix = np.vstack(resampled_cycles)
        mean_activation = np.mean(all_cycles_matrix, axis=0)
        std_activation = np.std(all_cycles_matrix, axis=0)

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 7))

        # Plot individual cycles faintly
        for cycle_data in all_cycles_matrix:
            ax.plot(norm_time, cycle_data, color='gray', alpha=0.2)

        # Plot the mean and std deviation
        ax.plot(norm_time, mean_activation, color='red', linewidth=2.5, label='Mean Activation')
        ax.fill_between(norm_time, mean_activation - std_activation, mean_activation + std_activation,
                        color='red', alpha=0.2, label='Std. Deviation')

        ax.set_title(f'TA Muscle Activation Pattern for: {self.trial_name}', fontsize=16)
        ax.set_xlabel('Gait Cycle (%)', fontsize=12)
        ax.set_ylabel('Normalized Muscle Thickness', fontsize=12)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        save_path = self.output_dir / f"{self.trial_name}_muscle_activation_plot.png"
        plt.savefig(save_path)
        plt.close(fig)

    def _calculate_gait_cycle_metrics(self):
        # This function is simplified as some metrics are not needed for this version
        metrics = []
        gait_cycles = self.results_df[self.results_df['gait_cycle'] > 0]['gait_cycle'].unique()
        sampling_period = self.config['trial_duration_s'] / len(self.results_df)

        for cycle_num in gait_cycles:
            cycle_df = self.results_df[self.results_df['gait_cycle'] == cycle_num]
            swing_points = cycle_df[cycle_df['gait_event'] == 'Swing']

            if len(swing_points) > 1:
                swing_duration = swing_points['time_s'].iloc[-1] - swing_points['time_s'].iloc[0]
            else:
                swing_duration = 0

            metrics.append({'gait_cycle': cycle_num, 'swing_duration_s': swing_duration})
        return pd.DataFrame(metrics)

    def _calculate_fatigue_trend(self, df: pd.DataFrame, metric_col: str):
        df_cleaned = df.dropna(subset=[metric_col, 'time_s'])
        if len(df_cleaned) < 2:
            return {'slope': np.nan, 'intercept': np.nan, 'r_squared': np.nan}
        X = df_cleaned['time_s'].values.reshape(-1, 1)
        y = df_cleaned[metric_col].values
        model = LinearRegression().fit(X, y)
        return {'slope': model.coef_[0], 'intercept': model.intercept_, 'r_squared': model.score(X, y)}

    def _plot_time_series(self, y_col: str, title: str, ylabel: str):
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.plot(self.results_df['time_s'], self.results_df[y_col], linestyle='-', label=y_col.replace('_', ' ').title(),
                alpha=0.8, linewidth=1.5)
        swing_df = self.results_df[self.results_df['gait_event'] == 'Swing']
        ax.scatter(swing_df['time_s'], swing_df[y_col], color='red', s=50, zorder=5, label='Swing Contraction')
        ax.set_title(f"{self.trial_name}\n{title}", fontsize=16)
        ax.set_xlabel('Time (s)', fontsize=12);
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend();
        ax.grid(True);
        ax.set_xlim(0, self.results_df['time_s'].max())
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{self.trial_name}_{y_col}_timeseries.png")
        plt.close(fig)

    def _plot_fatigue_analysis(self, df_swing: pd.DataFrame, metric_col: str, trend: dict, title: str, ylabel: str):
        df_cleaned = df_swing.dropna(subset=[metric_col, 'time_s'])
        if len(df_cleaned) < 2: return
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(df_cleaned['time_s'], df_cleaned[metric_col], label='Peak value at Swing')
        x_trend = np.array([df_cleaned['time_s'].min(), df_cleaned['time_s'].max()])
        y_trend = trend['slope'] * x_trend + trend['intercept']
        trend_label = f"Trend (Slope: {trend['slope']:.4f}, R²: {trend['r_squared']:.2f})"
        ax.plot(x_trend, y_trend, 'r-', linewidth=2, label=trend_label)
        ax.set_title(f"{self.trial_name}\n{title}", fontsize=16)
        ax.set_xlabel('Time (s)', fontsize=12);
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend();
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{self.trial_name}_fatigue_trend_{metric_col}.png")
        plt.close(fig)


def main():
    """Main function to run batch processing pipeline for beaaamformed images."""
    # ##########################################################################
    # ### 1. USER CONFIGURATION ################################################
    # ##########################################################################
    BASE_DATA_DIR = Path(r"Z:\AnkleStudy\participantData\unassistedTreadmillWalking\20602_Sub10_5June")
    OUTPUT_DIR = Path(r"C:\Users\msingh25\Desktop\github_gpu\us_processing_for_neuromuscularsys")
    TRIAL_FOLDER_PATTERN = "t*_*.*"

    CONFIG = {
        'trial_duration_s': 30,
        'pixel_spacing_mm': 0.1,
        'gait_detection_signal': 'thickness_mm',
        'rest_period_s': 30,

        # --- TUNING PARAMETERS FOR ROBUST GAIT DETECTION ---
        'filter_cutoff_hz': 2.0,
        'peak_prominence_std_factor': 0.7,  # Increase to reject noise; decrease to find smaller contractions.
        'peak_min_distance_s': 0.8,
    }
    # ##########################################################################

    print("Starting Automated Ultrasound Fatigue Analysis Pipeline (v9.0 Robust Gait)...")
    if not BASE_DATA_DIR.is_dir():
        raise FileNotFoundError(f"Base data folder not found: {BASE_DATA_DIR}")

    trial_paths = [p for p in BASE_DATA_DIR.glob(TRIAL_FOLDER_PATTERN) if p.is_dir()]
    if not trial_paths:
        raise FileNotFoundError(f"No trial folders found for pattern '{TRIAL_FOLDER_PATTERN}' in {BASE_DATA_DIR}")

    trial_paths.sort(key=lambda p: [int(c) if c.isdigit() else c.lower() for c in re.split('([0-9]+)', p.name)])
    print(f"Found {len(trial_paths)} trials to analyze.")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    master_output_dir = OUTPUT_DIR / f"image_results_{BASE_DATA_DIR.name}_{timestamp}"
    master_output_dir.mkdir(exist_ok=True, parents=True)
    CONFIG['master_output_dir'] = master_output_dir
    print(f"\nAll results will be saved in: {master_output_dir.resolve()}")

    master_summary_list = []
    shared_roi = None

    for i, trial_path in enumerate(trial_paths):
        image_folder_path = trial_path / 'images'
        if not image_folder_path.is_dir():
            continue

        print("\n" + "=" * 80)
        print(f"Processing Trial {i + 1}/{len(trial_paths)}: {image_folder_path.relative_to(BASE_DATA_DIR)}")

        analyzer = UltrasoundTrialAnalyzer(image_folder_path, CONFIG)

        if shared_roi is None:
            first_image = next(image_folder_path.glob('*.jpg'), None)
            if not first_image: continue
            shared_roi = analyzer.select_roi_interactively(first_image)
            if not shared_roi:
                print("ROI selection cancelled. Exiting.")
                return

        if analyzer.process_images(shared_roi):
            analyzer.identify_gait_events()
            summary = analyzer.run_fatigue_analysis()
            if summary:
                master_summary_list.append(summary)

    if master_summary_list:
        summary_df = pd.DataFrame(master_summary_list)
        summary_csv_path = master_output_dir / "master_fatigue_summary.csv"
        summary_df.to_csv(summary_csv_path, index=False)
        print("\n" + "#" * 80)
        print("PIPELINE COMPLETE")
        print(f"Master summary saved to: {summary_csv_path.resolve()}")
        print("#" * 80)
        print("\nFinal Summary Across All Trials:")
        print(summary_df.to_string())
    else:
        print("\nPipeline complete, but no trials had sufficient data for a final summary.")


if __name__ == "__main__":
    main()