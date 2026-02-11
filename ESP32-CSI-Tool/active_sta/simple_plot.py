import sys
import serial
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

# --- CONFIGURATION ---
SERIAL_PORT = '/dev/cu.usbserial-57460201261'  # Your specific port
BAUD_RATE = 115200
SUBCARRIER_INDEX = 44  # Which subcarrier to plot (0-63)
# ---------------------

def live_plot():
    # Setup the Serial Connection
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"Connected to {SERIAL_PORT}")
    except Exception as e:
        print(f"Error opening port: {e}")
        return

    # Setup the Graph
    plt.ion()
    fig, ax = plt.subplots()
    data_buffer = deque([0] * 100, maxlen=100)
    line, = ax.plot(data_buffer, 'r-')
    ax.set_ylim(0, 50)  # Adjust vertical scale as needed
    ax.set_title(f"CSI Amplitude (Subcarrier {SUBCARRIER_INDEX})")

    print("Waiting for CSI Data... (Press Ctrl+C to stop)")
    print("If graph stays flat, press the EN button on ESP32 once.")

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

                    # Extract Complex Numbers (Real, Imaginary, Real, Imaginary...)
                    # We just want the amplitude of one subcarrier
                    if len(csi_values) > SUBCARRIER_INDEX * 2:
                        real = csi_values[SUBCARRIER_INDEX * 2]
                        imag = csi_values[SUBCARRIER_INDEX * 2 + 1]
                        amplitude = np.sqrt(real**2 + imag**2)

                        # Update Graph
                        data_buffer.append(amplitude)
                        line.set_ydata(data_buffer)
                        fig.canvas.draw()
                        fig.canvas.flush_events()
                        
                except Exception as e:
                    pass # Ignore bad packets

    except KeyboardInterrupt:
        print("\nStopping...")
        ser.close()

if __name__ == "__main__":
    live_plot()
