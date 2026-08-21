import serial, time, sys

out_file = sys.argv[1] if len(sys.argv) > 1 else 'record.csv'
def get_serial():
    s = serial.Serial('/dev/cu.usbserial-5B530174971', 460800, timeout=0.5)
    s.dtr = False
    s.rts = True
    time.sleep(0.1)
    s.rts = False
    time.sleep(0.5)
    return s

s = get_serial()
print(f"Recording 5 minutes to {out_file}...")
duration = 300
start = time.time()

with open(out_file, 'w') as f:
    while True:
        elapsed = time.time() - start
        if elapsed > duration:
            break
        
        progress = int((elapsed / duration) * 50)
        sys.stdout.write(f"\r[{'=' * progress}{' ' * (50 - progress)}] {int(elapsed)}/300s ")
        sys.stdout.flush()
        
        try:
            line = s.readline().decode(errors='ignore')
            if 'CSI_DATA' in line and 'EE:60' in line.upper():
                f.write(line)
        except serial.SerialException:
            # The dreaded Mac USB serial hang! Auto-recover.
            sys.stdout.write("\n[USB Disconnect Detected. Auto-recovering ESP32...]\n")
            sys.stdout.flush()
            try:
                s.close()
            except:
                pass
            time.sleep(1)
            s = get_serial()
            
print(f"\nDone! Saved {out_file}.")
