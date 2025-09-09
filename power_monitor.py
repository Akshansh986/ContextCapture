import subprocess
import time
import threading


is_system_awake = True
power_monitor_running = True
power_state_lock = threading.Lock()


def monitor_power_state():
    """
    Monitor macOS power state changes (sleep/wake) in a separate thread
    """
    global is_system_awake, power_monitor_running
    
    print("🔋 Starting power state monitor...")
    
    while power_monitor_running:
        try:
            result = subprocess.run(['pmset', '-g', 'ps'], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                output = result.stdout.lower()
                
                with power_state_lock:
                    if not is_system_awake:
                        is_system_awake = True
                        print("💡 System woke up - resuming monitoring")
            else:
                with power_state_lock:
                    if is_system_awake:
                        is_system_awake = False
                        print("😴 System appears to be sleeping - pausing monitoring")
                        
        except subprocess.TimeoutExpired:
            with power_state_lock:
                if is_system_awake:
                    is_system_awake = False
                    print("😴 System timeout detected - pausing monitoring")
        except Exception as e:
            print(f"Power monitor error: {e}")
            
        time.sleep(10)


def monitor_system_events():
    """
    Alternative method: Monitor macOS system events for sleep/wake
    """
    global is_system_awake, power_monitor_running
    
    try:
        cmd = ['log', 'stream', '--predicate', 'subsystem == "com.apple.kernel" AND category == "PM"', '--level', 'info']
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        
        print("🔋 Monitoring system power events...")
        
        for line in iter(process.stdout.readline, ''):
            if not power_monitor_running:
                break
                
            line = line.strip().lower()
            
            if 'sleep' in line or 'going to sleep' in line:
                with power_state_lock:
                    if is_system_awake:
                        is_system_awake = False
                        print("😴 System going to sleep - pausing monitoring")
            elif 'wake' in line or 'waking up' in line or 'wake from' in line:
                with power_state_lock:
                    if not is_system_awake:
                        is_system_awake = True
                        print("💡 System waking up - resuming monitoring")
                        
    except Exception as e:
        print(f"System event monitor error: {e}")
        monitor_power_state()


def is_awake():
    """
    Check if system is currently awake
    """
    with power_state_lock:
        return is_system_awake


def stop_monitoring():
    """
    Stop power monitoring
    """
    global power_monitor_running
    power_monitor_running = False
