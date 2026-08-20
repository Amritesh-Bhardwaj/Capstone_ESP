import sys
import serial
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

# --- CONFIGURATION ---
SERIAL_PORT = '/dev/cu.usbserial-5B530174971'  # Your specific port
BAUD_RATE = 460800
SUBCARRIER_INDEX = 44  # Which subcarrier to plot individually (0-63)
WINDOW_SIZE = 100      # Number of time steps to show in the plot
SMOOTHING_WINDOW = 5   # Window for moving average smoothing
# ---------------------

def moving_average(data, window):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')

def live_plot():
    # Setup the Serial Connection
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"Connected to {SERIAL_PORT}")
    except Exception as e:
        print(f"Error opening port: {e}")
        print("Please check your SERIAL_PORT and ensure the ESP32 is connected.")
        return

    # Data Buffers
    # Buffer for the heatmap: (TIME_STEPS, NUM_SUBCARRIERS)
    # Most ESP32 CSI has 64 subcarriers (some are zeroed out)
    heatmap_buffer = deque([np.zeros(64)] * WINDOW_SIZE, maxlen=WINDOW_SIZE)
    single_sub_buffer = deque([0.0] * WINDOW_SIZE, maxlen=WINDOW_SIZE)

    # Setup the Graph
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 2]})
    fig.canvas.manager.set_window_title('ESP32 CSI - Human Activity Recognition Monitor')

    # Top Plot: Individual Subcarrier Amplitude
    line, = ax1.plot(range(WINDOW_SIZE), [0]*WINDOW_SIZE, 'r-', linewidth=1.5, label=f'Subcarrier {SUBCARRIER_INDEX}')
    smoothed_line, = ax1.plot([], [], 'b-', linewidth=2, label='Smoothed (Moving Avg)')
    movement_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, color='green', fontweight='bold', fontsize=12, verticalalignment='top')
    ax1.set_ylim(0, 60)
    ax1.set_xlim(0, WINDOW_SIZE)
    ax1.set_title(f"CSI Amplitude - Human Activity Signal", fontsize=14)
    ax1.set_ylabel("Amplitude")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Bottom Plot: Waterfall (Heatmap) of all Subcarriers
    initial_heatmap = np.zeros((WINDOW_SIZE, 64)).T
    im = ax2.imshow(initial_heatmap, aspect='auto', cmap='viridis', origin='lower', vmin=0, vmax=50)
    ax2.set_title("CSI Spectrogram (All Subcarriers)", fontsize=14)
    ax2.set_xlabel("Time (Samples)")
    ax2.set_ylabel("Subcarrier Index")
    fig.colorbar(im, ax=ax2, label='Amplitude')

    plt.tight_layout()

    print("Waiting for CSI Data... (Press Ctrl+C to stop)")
    print("Tip: Fast changes in the heatmap often indicate human movement (HAR).")

    try:
        while True:
            # Read a line from the ESP32
            line_bytes = ser.readline()
            try:
                line_str = line_bytes.decode('utf-8', errors='ignore').strip()
            except:
                continue

            # Look for the Magic Word
            if "CSI_DATA" in line_str:
                try:
                    # Parse the text line: "CSI_DATA,STA,Mac,RSSI,..."
                    parts = line_str.split(',')
                    # The array is usually the last part, wrapped in brackets [ ]
                    raw_array = parts[-1].replace('[', '').replace(']', '').strip()
                    csi_values = [int(x) for x in raw_array.split(' ') if x != '']

                    if len(csi_values) < 2:
                        continue

                    # Calculate amplitudes for all available subcarriers
                    # CSI data is pairs of (Imag, Real) or (Real, Imag)
                    amplitudes = []
                    for i in range(0, len(csi_values) - 1, 2):
                        v1 = csi_values[i]
                        v2 = csi_values[i+1]
                        amplitudes.append(np.sqrt(v1**2 + v2**2))
                    
                    if not amplitudes:
                        continue

                    # Pad or truncate to 64 subcarriers for consistent heatmap
                    amp_np = np.array(amplitudes)
                    if len(amp_np) > 64:
                        amp_np = amp_np[:64]
                    elif len(amp_np) < 64:
                        amp_np = np.pad(amp_np, (0, 64 - len(amp_np)), 'constant')

                    # Update Buffers
                    heatmap_buffer.append(amp_np)
                    if len(amp_np) > SUBCARRIER_INDEX:
                        single_sub_buffer.append(amp_np[SUBCARRIER_INDEX])
                    else:
                        single_sub_buffer.append(0.0)

                    # Update Line Plot
                    y_data = list(single_sub_buffer)
                    line.set_ydata(y_data)
                    
                    # Apply smoothing
                    smoothed = moving_average(y_data, SMOOTHING_WINDOW)
                    smoothed_line.set_data(range(len(y_data) - len(smoothed), len(y_data)), smoothed)

                    # Movement Detection (HAR feature)
                    # Calculate variance of the last 10 samples to detect activity
                    if len(y_data) >= 10:
                        variance = np.var(y_data[-10:])
                        if variance > 5.0:  # Threshold for movement
                            movement_text.set_text("MOVEMENT DETECTED")
                            movement_text.set_color('red')
                        else:
                            movement_text.set_text("STATIONARY")
                            movement_text.set_color('green')

                    # Update Heatmap
                    # Transpose buffer to get (Subcarriers, Time) for imshow
                    heatmap_data = np.array(heatmap_buffer).T
                    im.set_data(heatmap_data)

                    # Dynamic scaling for better visibility
                    if np.max(amp_np) > ax1.get_ylim()[1]:
                        ax1.set_ylim(0, np.max(amp_np) * 1.2)

                    fig.canvas.draw()
                    fig.canvas.flush_events()
                        
                except Exception as e:
                    # print(f"Parse error: {e}") # Debugging
                    pass 

    except KeyboardInterrupt:
        print("\nStopping...")
        ser.close()
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    live_plot()
