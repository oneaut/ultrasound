# -*- coding: utf-8 -*-
"""

=====================================================

- **Gait Detection:** Implements a low-pass filter and peak detection thresholds (height, prominence) to eliminate false
  positives caused by noise.
- **Plot:** plot for each
  trial, showing the raw vs. filtered signal and the thresholds used for peak
  detection.
- **Reusable ROI:** Select the ROI once for the entire batch run.
- **Cross-Trial Analysis:** Models cumulative fatigue across an entire session.

"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks, butter, filtfilt 
import sys
import time
import re


class UltrasoundTrialAnalyzer:
    

    def __init__(self, trial_path: Path, config: dict):
        """
        Initializes the analyzer with the path to the trial data and configuration.
        """
        self.trial_path = trial_path
        self.config = config
        self.trial_name = trial_path.name
        self.results_df = None

        self.output_dir = self.config['master_output_dir'] / self.trial_name
        self.output_dir.mkdir(exist_ok=True)

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
        """Processes all images to extract time-series data."""
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
        self.results_df = df
        return True

    def _calculate_muscle_thickness(self, roi_image: np.ndarray) -> float:
        """Calculates vertical muscle thickness from an ROI image."""
        blurred_img = cv2.GaussianBlur(roi_image, (5, 5), 0)
        edges = cv2.Canny(blurred_img, 50, 150)
        width = edges.shape[1]
        center_start, center_end = int(width * 0.25), int(width * 0.75)
        central_column_edges = edges[:, center_start:center_end]
        edge_points_y, _ = np.where(central_column_edges > 0)
        if edge_points_y.size < 2: return np.nan
        thickness_px = np.max(edge_points_y) - np.min(edge_points_y)
        return thickness_px * self.config['pixel_spacing_mm']

    # ### MODIFIED ###: gait detection function
    def identify_gait_events(self):
        """
        Identifies gait events using a peak detection.
        """
        signal_col = self.config['gait_detection_signal']
        raw_signal = self.results_df[signal_col].values

        if pd.isna(raw_signal).all():
            print(f"Signal '{signal_col}' contains only NaN values. Skipping gait detection.")
            self.results_df['gait_event'] = 'Stance'
            return

        sampling_rate = len(self.results_df) / self.config['trial_duration_s']

        # 1. Apply a low-pass Butterworth filter to remove noise
        nyquist = 0.5 * sampling_rate
        cutoff = self.config['filter_cutoff_hz']
        b, a = butter(2, cutoff / nyquist, btype='low', analog=False)
        filtered_signal = filtfilt(b, a, raw_signal)

        # 2. Calculate  peak detection parameters based on the *filtered* signal
        signal_mean = np.nanmean(filtered_signal)
        signal_std = np.nanstd(filtered_signal)

        # Minimum height: Must be significantly above the mean
        min_peak_height = signal_mean + self.config['peak_min_height_std_factor'] * signal_std

        # Minimum prominence: Must stand out from neighbors by a significant amount
        min_peak_prominence = self.config['peak_prominence_std_factor'] * signal_std

        # Minimum distance between peaks in samples
        min_peak_distance = int(self.config['peak_min_distance_s'] * sampling_rate)

        # 3. Find peaks using the new robust parameters
        peak_indices, properties = find_peaks(
            filtered_signal,
            height=min_peak_height,
            prominence=min_peak_prominence,
            distance=min_peak_distance
        )

        self.results_df['gait_event'] = 'Stance'
        self.results_df.loc[peak_indices, 'gait_event'] = 'Swing'

        print(f"Identified {len(peak_indices)} swing events using robust detection on '{signal_col}'.")

        # 4. Generate and save a debug plot
        self._plot_gait_detection_debug(
            raw_signal,
            filtered_signal,
            peak_indices,
            min_peak_height,
            min_peak_prominence,
            sampling_rate
        )

    # plotting gait detection
    def _plot_gait_detection_debug(self, raw_signal, filtered_signal, peaks, height_thresh, prominence_thresh, fs):
        """
        """
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(18, 8))
        time_axis = np.arange(len(raw_signal)) / fs

        # Plot signals
        ax.plot(time_axis, raw_signal, color='gray', alpha=0.5, linewidth=1.0, label='Raw Signal')
        ax.plot(time_axis, filtered_signal, color='blue', alpha=0.9, linewidth=1.5, label='Filtered Signal')

        # Plot detected peaks
        ax.plot(time_axis[peaks], filtered_signal[peaks], "x", color='red', markersize=10, mew=2,
                label=f'Detected Peaks ({len(peaks)})')

        # Plot threshold lines for context
        ax.axhline(y=height_thresh, color='green', linestyle='--', linewidth=1.5,
                   label=f'Min Height Threshold: {height_thresh:.2f}')

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        param_text = (
            f"Parameters:\n"
            f" - Min Distance: {self.config['peak_min_distance_s']}s\n"
            f" - Height Factor: {self.config['peak_min_height_std_factor']} (std)\n"
            f" - Prominence Factor: {self.config['peak_prominence_std_factor']} (std)\n"
            f" - Prominence Value: {prominence_thresh:.2f}"
        )
        ax.text(0.95, 0.95, param_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right', bbox=props)

        ax.set_title(f"Gait Detection Debug Plot for: {self.trial_name}", fontsize=16)
        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel(self.config['gait_detection_signal'], fontsize=12)
        ax.legend(loc='lower left')
        ax.grid(True)
        ax.set_xlim(0, time_axis[-1])

        plt.tight_layout()
        save_path = self.output_dir / f"{self.trial_name}_gait_detection_debug.png"
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Gait detection debug plot saved to: {save_path.name}")

    def run_fatigue_analysis(self):
        """Calculates fatigue metrics"""
        swing_events = self.results_df[self.results_df['gait_event'] == 'Swing'].copy()

        self._plot_time_series('thickness_mm', 'Muscle Thickness Over Time', 'Thickness (mm)')
        self._plot_time_series('echogenicity', 'Echogenicity Over Time', 'Mean Pixel Intensity')
        print("Time-series plots saved.")

        detailed_csv_path = self.output_dir / f"{self.trial_name}_detailed_data.csv"
        self.results_df.to_csv(detailed_csv_path, index=False)
        print(f"Detailed data saved to: {detailed_csv_path.name}")

        if len(swing_events) < 3:
            print(f"Warning: Only {len(swing_events)} swing events found. Fatigue trend analysis may be unreliable.")
            return {
                'trial_name': self.trial_name,
                'num_gait_cycles': len(swing_events),
                'mean_peak_thickness': swing_events['thickness_mm'].mean() if not swing_events.empty else np.nan,
                'mean_peak_echogenicity': swing_events['echogenicity'].mean() if not swing_events.empty else np.nan,
                'thickness_slope_intra_trial': np.nan,
                'thickness_r2': np.nan,
                'echogenicity_slope_intra_trial': np.nan,
                'echogenicity_r2': np.nan,
                'fatigue_index_thickness': np.nan,
                'fatigue_index_echogenicity': np.nan
            }

        thick_trend = self._calculate_fatigue_trend(swing_events, 'thickness_mm')
        echo_trend = self._calculate_fatigue_trend(swing_events, 'echogenicity')
        self._plot_fatigue_analysis(swing_events, 'thickness_mm', thick_trend, 'Fatigue Trend: Peak Muscle Thickness',
                                    'Peak Thickness (mm)')
        self._plot_fatigue_analysis(swing_events, 'echogenicity', echo_trend, 'Fatigue Trend: Peak Echogenicity',
                                    'Peak Echogenicity')
        print("Fatigue trend plots saved.")

        fatigue_idx_thick = self._calculate_normalized_fatigue_index(swing_events, 'thickness_mm',
                                                                     higher_is_fatigued=False)
        fatigue_idx_echo = self._calculate_normalized_fatigue_index(swing_events, 'echogenicity',
                                                                    higher_is_fatigued=True)

        summary = {
            'trial_name': self.trial_name,
            'num_gait_cycles': len(swing_events),
            'mean_peak_thickness': swing_events['thickness_mm'].mean(),
            'mean_peak_echogenicity': swing_events['echogenicity'].mean(),
            'thickness_slope_intra_trial': thick_trend['slope'],
            'thickness_r2': thick_trend['r_squared'],
            'echogenicity_slope_intra_trial': echo_trend['slope'],
            'echogenicity_r2': echo_trend['r_squared'],
            'fatigue_index_thickness': fatigue_idx_thick,
            'fatigue_index_echogenicity': fatigue_idx_echo
        }
        return summary

    def _calculate_fatigue_trend(self, df: pd.DataFrame, metric_col: str):
        X = df['time_s'].values.reshape(-1, 1)
        y = df[metric_col].values
        model = LinearRegression().fit(X, y)
        return {'slope': model.coef_[0], 'intercept': model.intercept_, 'r_squared': model.score(X, y)}

    def _calculate_normalized_fatigue_index(self, df: pd.DataFrame, metric_col: str, higher_is_fatigued: bool):
        if len(df) < 6:  # Need at least 3 at start and 3 at end
            return np.nan
        metric_start = df[metric_col].iloc[:3].mean()
        metric_end = df[metric_col].iloc[-3:].mean()
        if higher_is_fatigued:
            return min(1.0, metric_start / metric_end) if metric_end != 0 else np.nan
        else:
            return min(1.0, metric_end / metric_start) if metric_start != 0 else np.nan

    def _plot_time_series(self, y_col: str, title: str, ylabel: str):
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.plot(self.results_df['time_s'], self.results_df[y_col], linestyle='-', label=y_col.replace('_', ' ').title(),
                alpha=0.8, linewidth=1.5)
        swing_df = self.results_df[self.results_df['gait_event'] == 'Swing']
        ax.scatter(swing_df['time_s'], swing_df[y_col], color='red', s=50, zorder=5, label='Swing Peak')
        ax.set_title(f"{self.trial_name}\n{title}", fontsize=16)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend()
        ax.grid(True)
        ax.set_xlim(0, self.results_df['time_s'].max())
        plt.tight_layout()
        save_path = self.output_dir / f"{self.trial_name}_{y_col}_timeseries.png"
        plt.savefig(save_path)
        plt.close(fig)

    def _plot_fatigue_analysis(self, df_swing: pd.DataFrame, metric_col: str, trend: dict, title: str, ylabel: str):
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(df_swing['time_s'], df_swing[metric_col], label='Peak value at Swing')
        x_trend = np.array([df_swing['time_s'].min(), df_swing['time_s'].max()])
        y_trend = trend['slope'] * x_trend + trend['intercept']
        trend_label = f"Trend (Slope: {trend['slope']:.4f}, R²: {trend['r_squared']:.2f})"
        ax.plot(x_trend, y_trend, 'r-', linewidth=2, label=trend_label)
        ax.set_title(f"{self.trial_name}\n{title}", fontsize=16)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        save_path = self.output_dir / f"{self.trial_name}_fatigue_trend_{metric_col}.png"
        plt.savefig(save_path)
        plt.close(fig)


def plot_cross_trial_trend(summary_df: pd.DataFrame, output_dir: Path):
    """
    Plots the trend of a metric across multiple trials.
    """
    if len(summary_df) < 2:
        print("Not enough trials to plot a cross-trial trend.")
        return 0, 0

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    metric_col = 'mean_peak_thickness'
    time_col = 'experiment_midpoint_s'

    # Ensure data is sorted by time for plotting
    summary_df = summary_df.sort_values(by=time_col)

    ax.plot(summary_df[time_col], summary_df[metric_col], 'o-', markersize=8, label='Mean Peak Thickness per Trial')

    X = summary_df[time_col].values.reshape(-1, 1)
    y = summary_df[metric_col].values
    model = LinearRegression().fit(X, y)
    slope = model.coef_[0]
    r_squared = model.score(X, y)

    x_trend = np.array([summary_df[time_col].min(), summary_df[time_col].max()])
    y_trend = model.predict(x_trend.reshape(-1, 1))
    trend_label = f'Overall Trend (Slope: {slope:.5f} mm/sec, R²: {r_squared:.2f})'
    ax.plot(x_trend, y_trend, 'r--', linewidth=2, label=trend_label)

    ax.set_title('Cross-Trial Fatigue Analysis: Muscle Thickness', fontsize=16, fontweight='bold')
    ax.set_xlabel('Total Experiment Time (s)', fontsize=12)
    ax.set_ylabel('Mean Peak Muscle Thickness (mm)', fontsize=12)
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    save_path = output_dir / "master_cross_trial_fatigue_trend.png"
    plt.savefig(save_path)
    plt.close(fig)
    print(f"\nCross-trial trend plot saved to: {save_path.name}")
    return slope, r_squared


def main():
    """
    Main function to run the automated batch processing pipeline.
    """
    # ##########################################################################
    # ### 1. USER CONFIGURATION ################################################
    # ##########################################################################

    PROJECT_ROOT = Path(r"C:\Users\msingh25\Pictures\may29")
    TRIAL_FOLDER_PATTERN = "images_*"

    CONFIG = {
        'trial_duration_s': 30,
        'pixel_spacing_mm': 0.1,
        'gait_detection_signal': 'thickness_mm',  # 'thickness_mm' or 'echogenicity'
        'rest_period_s': 60,

        # Gait Detection Parameters
        # --- How to Tune These Parameters ---
        # - If you are getting too few peaks (missing gait cycles):
        #   - Try DECREASING 'peak_min_height_std_factor' or 'peak_prominence_std_factor'.
        # - If you are still getting too many false peaks (from noise):
        #   - Try INCREASING 'peak_prominence_std_factor' or 'peak_min_distance_s'.
        #   - Try DECREASING 'filter_cutoff_hz' to smooth the signal more.

        # Low-pass filter cutoff frequency in Hz. Lower values mean more smoothing.
        # Good for removing jitter in slow walking. Typical human motion is < 10 Hz.
        'filter_cutoff_hz': 2.0,

        # The minimum distance between detected peaks in seconds. Prevents detecting
        # multiple false peaks within a single true gait cycle.
        'peak_min_distance_s': 1.0,  # Increased from 0.5s for slow walking

        # How many standard deviations above the mean a peak must be.
        # This acts as an absolute minimum threshold to qualify as a peak.
        'peak_min_height_std_factor': 0.5,

        # How many standard deviations a peak must 'stick out' from its surroundings.
        # This is very effective at rejecting small, noisy peaks.
        'peak_prominence_std_factor': 0.8,
    }

    # ##########################################################################
    # ### 2. SCRIPT EXECUTION ##################################################
    # ##########################################################################

    print("Starting Automated Ultrasound Fatigue Analysis Pipeline (v7.0)...")
    if not PROJECT_ROOT.is_dir():
        raise FileNotFoundError(f"Project root folder not found: {PROJECT_ROOT}")

    def natural_sort_key(path):
        return [int(c) if c.isdigit() else c.lower() for c in re.split('([0-9]+)', path.name)]

    trial_paths = sorted(list(PROJECT_ROOT.glob(TRIAL_FOLDER_PATTERN)), key=natural_sort_key)
    if not trial_paths:
        raise FileNotFoundError(f"No trial folders found for pattern '{TRIAL_FOLDER_PATTERN}' in {PROJECT_ROOT}")

    print(f"Found {len(trial_paths)} trials to analyze in numerical order.")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    master_output_dir = PROJECT_ROOT / f"results_v7_{timestamp}"
    master_output_dir.mkdir(exist_ok=True)
    CONFIG['master_output_dir'] = master_output_dir

    print("\n" + "*" * 60)
    print(f"All results will be saved in: {master_output_dir.resolve()}")
    print("*" * 60)

    master_summary_list = []
    shared_roi = None
    cumulative_time = 0.0

    for i, trial_path in enumerate(trial_paths):
        print("\n" + "=" * 80)
        print(f"Processing Trial {i + 1}/{len(trial_paths)}: {trial_path.name}")
        print("=" * 80)

        analyzer = UltrasoundTrialAnalyzer(trial_path, CONFIG)

        if shared_roi is None:
            first_image = next(trial_path.glob('*.jpg'), None)
            if not first_image:
                print(f"--- WARNING: No images found in '{trial_path.name}'. Skipping. ---")
                continue
            shared_roi = analyzer.select_roi_interactively(first_image)
            if not shared_roi:
                print("ROI selection cancelled. Exiting.")
                return

        if analyzer.process_images(shared_roi):
            analyzer.identify_gait_events()
            summary = analyzer.run_fatigue_analysis()
            if summary:
                summary['experiment_midpoint_s'] = cumulative_time + (CONFIG['trial_duration_s'] / 2)
                master_summary_list.append(summary)

        cumulative_time += CONFIG['trial_duration_s'] + CONFIG['rest_period_s']

    if master_summary_list:
        summary_df = pd.DataFrame(master_summary_list).dropna(subset=['thickness_slope_intra_trial'])

        if not summary_df.empty:
            overall_slope, overall_r2 = plot_cross_trial_trend(summary_df, master_output_dir)
            summary_df['overall_experimental_slope'] = overall_slope
            summary_csv_path = master_output_dir / "master_fatigue_summary.csv"
            summary_df.to_csv(summary_csv_path, index=False)

            print("\n" + "#" * 80)
            print("PIPELINE COMPLETE")
            print(f"A master summary of all trials has been saved to:")
            print(f"{summary_csv_path.resolve()}")
            print("#" * 80)
            print("\nFinal Summary Across All Trials:")
            print(summary_df.to_string())
        else:
            print("\nPipeline complete, but no trials had sufficient data for a final summary.")


if __name__ == "__main__":
    main()
