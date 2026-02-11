import sys
import serial
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

# --- CONFIGURATION ---
SERIAL_PORT = '/dev/cu.usbserial-57460201261' 
BAUD_RATE = 115200
SUBCARRIER_INDEX = 44  
HISTORY_SIZE = 200     # Increased to see more "history" on screen
# ---------------------

def fast_live_plot():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        ser.reset_input_buffer() # Clear any old "laggy" data
        print(f"Connected to {SERIAL_PORT}")
    except Exception as e:
        print(f"Error opening port: {e}")
        return

    # Setup the Graph
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6)) # Make the window larger
    
    # Data buffer
    data_buffer = deque([0] * HISTORY_SIZE, maxlen=HISTORY_SIZE)
    
    # create the initial line
    line, = ax.plot(range(HISTORY_SIZE), data_buffer, 'r-', linewidth=1.5)
    
    # --- LABELS & AESTHETICS ---
    ax.set_title(f"Real-Time CSI Amplitude (Subcarrier {SUBCARRIER_INDEX})", fontsize=14, fontweight='bold')
    ax.set_xlabel("Time (Packets)", fontsize=12)
    ax.set_ylabel("Signal Amplitude", fontsize=12)
    ax.set_ylim(0, 80)        # Adjust this if your waves are taller
    ax.set_xlim(0, HISTORY_SIZE-1)
    ax.grid(True, linestyle='--', alpha=0.6) # Add a grid for readability

    # --- OPTIMIZATION: Cache the background ---
    # We draw the static parts (axes, labels) once, save them, 
    # and just paste them back every frame.
    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(ax.bbox)

    print("Plotting started... Press Ctrl+C to stop.")

    try:
        while True:
            # Non-blocking read (helps prevent UI freeze)
            if ser.in_waiting:
                line_bytes = ser.readline()
                try:
                    line_str = line_bytes.decode('utf-8', errors='ignore').strip()
                    
                    if "CSI_DATA" in line_str:
                        # Parse Data
                        parts = line_str.split(',')
                        raw_array = parts[-1].replace('[', '').replace(']', '').strip()
                        csi_values = [int(x) for x in raw_array.split(' ') if x != '']

                        if len(csi_values) > SUBCARRIER_INDEX * 2:
                            real = csi_values[SUBCARRIER_INDEX * 2]
                            imag = csi_values[SUBCARRIER_INDEX * 2 + 1]
                            amplitude = np.sqrt(real**2 + imag**2)

                            # Update Data
                            data_buffer.append(amplitude)

                            # --- THE FAST DRAWING PART ---
                            # 1. Restore the saved background (erases the old line)
                            fig.canvas.restore_region(background)
                            
                            # 2. Update the line data
                            line.set_ydata(data_buffer)
                            
                            # 3. Draw ONLY the line
                            ax.draw_artist(line)
                            
                            # 4. Blit the result to screen
                            fig.canvas.blit(ax.bbox)
                            fig.canvas.flush_events()

                except ValueError:
                    pass 

    except KeyboardInterrupt:
        print("\nStopping...")
        ser.close()
        plt.close()

if __name__ == "__main__":
    fast_live_plot()