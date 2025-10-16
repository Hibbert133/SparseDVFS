import os
import time
import subprocess
import pandas as pd
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import logging
import glob
from tqdm import tqdm
import matplotlib
import seaborn as sns
import threading

matplotlib.use('Agg')  # Must be set before importing pyplot
# --- Configure logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dvfs_profiler.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# --- Auto-detect Jetson device paths ---
def detect_gpu_device():
    """Auto-detect GPU device path"""
    devfreq_path = "/sys/class/devfreq/"
    if os.path.exists(devfreq_path):
        for d in os.listdir(devfreq_path):
            full_path = os.path.join(devfreq_path, d)
            if os.path.isdir(full_path) and "gpu" in d.lower():
                return full_path
    return None


def detect_emc_device():
    """Auto-detect EMC (memory) device path"""
    devfreq_path = "/sys/class/devfreq/"
    if os.path.exists(devfreq_path):
        for d in os.listdir(devfreq_path):
            full_path = os.path.join(devfreq_path, d)
            if os.path.isdir(full_path) and ("emc" in d.lower() or "gv" in d or "dla" in d):
                return full_path
    return None


# Detect device paths
GPU_DEVICE_PATH = detect_gpu_device()
EMC_DEVICE_PATH = detect_emc_device()
# Validate path existence
if not GPU_DEVICE_PATH:
    logger.error("Cannot detect GPU device path. Please check if running on Jetson device.")
else:
    logger.info(f"Detected GPU device path: {GPU_DEVICE_PATH}")
if not EMC_DEVICE_PATH:
    logger.warning("Cannot detect EMC (memory) device path, memory frequency control will be disabled.")
    EMC_DEVICE_PATH = None
else:
    logger.info(f"Detected EMC (memory) device path: {EMC_DEVICE_PATH}")
# --- Jetson Orin Nano frequency control paths ---
# CPU frequency control
CPU_FREQ_PATH = "/sys/devices/system/cpu/cpufreq/policy0"
CPU_ONLINE_PATH = "/sys/devices/system/cpu"
CPU_MIN_FREQ = f"{CPU_FREQ_PATH}/scaling_min_freq"
CPU_MAX_FREQ = f"{CPU_FREQ_PATH}/scaling_max_freq"
CPU_GOVERNOR = f"{CPU_FREQ_PATH}/scaling_governor"
CPU_AVAILABLE_FREQS = f"{CPU_FREQ_PATH}/scaling_available_frequencies"
# GPU frequency control
if GPU_DEVICE_PATH:
    GPU_MIN_FREQ = f"{GPU_DEVICE_PATH}/min_freq"
    GPU_MAX_FREQ = f"{GPU_DEVICE_PATH}/max_freq"
    GPU_AVAILABLE_FREQS = f"{GPU_DEVICE_PATH}/available_frequencies"
    GPU_CUR_FREQ = f"{GPU_DEVICE_PATH}/cur_freq"
else:
    GPU_MIN_FREQ = GPU_MAX_FREQ = GPU_AVAILABLE_FREQS = GPU_CUR_FREQ = None
# EMC (memory) frequency control
if EMC_DEVICE_PATH:
    EMC_MIN_FREQ = f"{EMC_DEVICE_PATH}/min_freq"
    EMC_MAX_FREQ = f"{EMC_DEVICE_PATH}/max_freq"
    EMC_AVAILABLE_FREQS = f"{EMC_DEVICE_PATH}/available_frequencies"
    EMC_CUR_FREQ = f"{EMC_DEVICE_PATH}/cur_freq"
else:
    EMC_MIN_FREQ = EMC_MAX_FREQ = EMC_AVAILABLE_FREQS = EMC_CUR_FREQ = None


# --- Detect device's supported frequency ranges ---
def get_available_frequencies():
    """Get device's supported frequency ranges"""
    available = {
        "cpu": [],
        "gpu": [],
        "emc": []
    }
    # Get CPU available frequencies
    if os.path.exists(CPU_AVAILABLE_FREQS):
        try:
            with open(CPU_AVAILABLE_FREQS, 'r') as f:
                freqs = f.read().strip().split()
                available["cpu"] = sorted([int(x) for x in freqs])
        except Exception as e:
            logger.error(f"Cannot read CPU available frequencies: {str(e)}")
    # Get GPU available frequencies
    if GPU_AVAILABLE_FREQS and os.path.exists(GPU_AVAILABLE_FREQS):
        try:
            with open(GPU_AVAILABLE_FREQS, 'r') as f:
                freqs = f.read().strip().split()
                available["gpu"] = sorted([int(x) for x in freqs])
        except Exception as e:
            logger.error(f"Cannot read GPU available frequencies: {str(e)}")
    # Get EMC available frequencies
    if EMC_AVAILABLE_FREQS and os.path.exists(EMC_AVAILABLE_FREQS):
        try:
            with open(EMC_AVAILABLE_FREQS, 'r') as f:
                freqs = f.read().strip().split()
                available["emc"] = sorted([int(x) for x in freqs])
        except Exception as e:
            logger.error(f"Cannot read EMC available frequencies: {str(e)}")
    return available


# Get device's supported frequencies
AVAILABLE_FREQUENCIES = get_available_frequencies()
logger.info(f"Device supported frequency ranges:")
logger.info(f"  CPU: {AVAILABLE_FREQUENCIES['cpu']}")
logger.info(f"  GPU: {AVAILABLE_FREQUENCIES['gpu']}")
logger.info(f"  EMC: {AVAILABLE_FREQUENCIES['emc']}")


# --- Frequency Configuration Management ---
class FrequencyManager:
    def __init__(self):
        self.original_settings = {
            "cpu_min": 0,
            "cpu_max": 0,
            "gpu_min": 0,
            "gpu_max": 0,
            "emc_min": 0,
            "emc_max": 0
        }
        self.save_current_settings()

    def save_current_settings(self):
        """Save current frequency settings for restoration"""
        self.original_settings = {
            "cpu_min": int(read_file(CPU_MIN_FREQ) or 0),
            "cpu_max": int(read_file(CPU_MAX_FREQ) or 0),
            "gpu_min": int(read_file(GPU_MIN_FREQ) or 0) if GPU_DEVICE_PATH else 0,
            "gpu_max": int(read_file(GPU_MAX_FREQ) or 0) if GPU_DEVICE_PATH else 0,
            "emc_min": int(read_file(EMC_MIN_FREQ) or 0) if EMC_DEVICE_PATH else 0,
            "emc_max": int(read_file(EMC_MAX_FREQ) or 0) if EMC_DEVICE_PATH else 0
        }
        logger.info(f"Original frequency settings saved: {self.original_settings}")

    def restore_original(self):
        """Restore original frequency settings"""
        logger.info("Restoring original frequency settings...")
        set_frequency(
            self.original_settings["cpu_min"],
            self.original_settings["cpu_max"],
            self.original_settings["gpu_min"],
            self.original_settings["gpu_max"],
            self.original_settings["emc_min"],
            self.original_settings["emc_max"]
        )

    def get_current_profile(self):
        """Detect current frequency profile"""
        current = {
            "cpu_min": int(read_file(CPU_MIN_FREQ) or 0),
            "cpu_max": int(read_file(CPU_MAX_FREQ) or 0),
            "gpu_min": int(read_file(GPU_MIN_FREQ) or 0) if GPU_DEVICE_PATH else 0,
            "gpu_max": int(read_file(GPU_MAX_FREQ) or 0) if GPU_DEVICE_PATH else 0,
            "emc_min": int(read_file(EMC_MIN_FREQ) or 0) if EMC_DEVICE_PATH else 0,
            "emc_max": int(read_file(EMC_MAX_FREQ) or 0) if EMC_DEVICE_PATH else 0
        }
        return f"CPU_{current['cpu_max'] / 1000:.0f}_GPU_{current['gpu_max'] / 1000000:.0f}"


class PowerMonitor:
    """
    A class to run power measurement in a separate thread to ensure
    power is sampled *during* the workload execution.
    """

    def __init__(self, sample_interval=0.02):
        self._stop_event = threading.Event()
        self._thread = None
        self.power_samples = []
        self.sample_interval = sample_interval

    def _monitor_power(self):
        """Continuously measure power and add to list"""
        while not self._stop_event.is_set():
            power = read_power_watt()
            if power > 0:
                self.power_samples.append(power)
            time.sleep(self.sample_interval)

    def start(self):
        """Start the power monitoring thread"""
        self.power_samples = []
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_power)
        self._thread.start()
        logger.debug("Power monitor started.")

    def stop(self):
        """Stop monitoring and return the average power"""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        logger.debug(f"Power monitor stopped. Collected {len(self.power_samples)} samples.")
        if not self.power_samples:
            logger.warning("No power samples were collected.")
            return -1.0
        return np.mean(self.power_samples)


def read_file(path):
    """Safely read file content"""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, 'r') as f:
            return f.read().strip()
    except Exception:
        return None


def write_file(path, value):
    """Safely write to file"""
    if not path or not os.path.exists(path):
        return False
    try:
        with open(path, 'w') as f:
            f.write(str(value))
        return True
    except Exception as e:
        logger.error(f"Failed to write to {path}: {str(e)}")
        return False


def set_cpu_governor(governor="userspace"):
    """Set CPU governor to userspace mode (allows manual frequency control)"""
    if not os.path.exists(CPU_GOVERNOR):
        logger.error("CPU governor file not found")
        return False
    current_gov = read_file(CPU_GOVERNOR)
    if current_gov and current_gov != governor:
        if not write_file(CPU_GOVERNOR, governor):
            logger.error("Cannot set CPU governor to userspace mode")
            return False
    return True


def get_current_frequencies():
    """Get current CPU/GPU/EMC frequencies in MHz"""
    cpu_freq_mhz, gpu_freq_mhz, emc_freq_mhz = -1, -1, -1
    # Get CPU frequency
    cpu_freq = read_file(f"{CPU_FREQ_PATH}/scaling_cur_freq")
    if cpu_freq:
        cpu_freq_mhz = int(cpu_freq) / 1000  # Convert kHz to MHz
    # Get GPU frequency
    if GPU_DEVICE_PATH:
        gpu_freq = read_file(GPU_CUR_FREQ)
        if gpu_freq:
            gpu_freq_mhz = int(gpu_freq) / 1000000  # Convert Hz to MHz
    # Get EMC frequency
    if EMC_DEVICE_PATH:
        emc_freq = read_file(EMC_CUR_FREQ)
        if emc_freq:
            emc_freq_mhz = int(emc_freq) / 1000000  # Convert Hz to MHz
    return cpu_freq_mhz, gpu_freq_mhz, emc_freq_mhz


def set_frequency(cpu_min, cpu_max, gpu_min, gpu_max, emc_min, emc_max):
    """Set specific frequency configuration"""
    success = True
    # Set CPU frequencies
    if not os.path.exists(CPU_MIN_FREQ) or not os.path.exists(CPU_MAX_FREQ):
        logger.error("CPU frequency control paths not found")
        return False
    if not write_file(CPU_MIN_FREQ, cpu_min):
        success = False
    if not write_file(CPU_MAX_FREQ, cpu_max):
        success = False
    # Set GPU frequencies (if available)
    if GPU_DEVICE_PATH:
        if not write_file(GPU_MIN_FREQ, gpu_min):
            success = False
        if not write_file(GPU_MAX_FREQ, gpu_max):
            success = False
    # Set EMC frequencies (if available)
    if EMC_DEVICE_PATH:
        if not write_file(EMC_MIN_FREQ, emc_min):
            success = False
        if not write_file(EMC_MAX_FREQ, emc_max):
            success = False
    # Wait for frequencies to stabilize
    time.sleep(0.5)
    return success


# --- Power Measurement Module (Fixed for Orin Nano) ---
def measure_power_with_jtop():
    """Measure power using jetson-stats library for accurate measurements"""
    try:
        # Try to import jtop
        try:
            import jtop
        except ImportError:
            logger.debug("jtop module not installed. Please run 'pip install jetson-stats'")
            return -1.0
        # Initialize jtop
        try:
            jetson = jtop.jtop()
        except Exception as e:
            logger.debug(f"Failed to initialize jtop: {str(e)}")
            return -1.0
        # Open the connection
        try:
            jetson.start()
        except Exception as e:
            error_msg = str(e)
            logger.debug(f"jtop service not running: {error_msg}")
            # Attempt to restart jtop service if it's not running
            if "not active" in error_msg or "Connection refused" in error_msg:
                try:
                    logger.info("Attempting to restart jtop.service...")
                    subprocess.run(["sudo", "systemctl", "restart", "jtop.service"],
                                   check=True, capture_output=True, text=True)
                    time.sleep(2)  # Give service time to restart
                    # Try to initialize again
                    jetson = jtop.jtop()
                    jetson.start()
                    logger.info("Successfully restarted jtop.service")
                except Exception as restart_error:
                    logger.debug(f"Failed to restart jtop.service: {str(restart_error)}")
                    return -1.0
            else:
                return -1.0
        # Read power measurements
        try:
            # Check if the power attribute exists
            if not hasattr(jetson, 'power'):
                logger.debug("jtop object does not have 'power' attribute")
                return -1.0
            power_data = jetson.power
            total_power_w = 0.0
            valid_power_found = False  # Track if any valid power data was found
            # Handle your specific device format
            for rail_name, rail_data in power_data.items():
                if isinstance(rail_data, dict):
                    # Directly check for 'power' key first (this is your device's format)
                    if 'power' in rail_data:
                        power_value = rail_data['power']
                        logger.debug(f"Found 'power' key for {rail_name}: {power_value}mW")
                        # Convert to watts and accumulate
                        total_power_w += power_value / 1000.0
                        valid_power_found = True
                        continue
                    # Fallback to other possible keys
                    for key in ['avg_power', 'cur_power', 'power1', 'avg', 'cur']:
                        if key in rail_data:
                            power_value = rail_data[key]
                            logger.debug(f"Found '{key}' key for {rail_name}: {power_value}mW")
                            total_power_w += power_value / 1000.0
                            valid_power_found = True
                            break
                else:
                    logger.debug(f"Unexpected data format for rail {rail_name}: {type(rail_data)}")
            # If no valid power data was collected, return -1
            if not valid_power_found:
                logger.debug("No valid power data collected from any rail")
                return -1.0
            logger.debug(f"Total measured power: {total_power_w:.2f}W from {len(power_data)} rails")
            return total_power_w
        finally:
            # Always close the connection
            try:
                if 'jetson' in locals() and jetson is not None:
                    jetson.close()
            except:
                pass
    except Exception as e:
        logger.debug(f"jtop power measurement failed: {str(e)}")
        return -1.0


def get_power_from_nvidia_path():
    """
    Fallback method for power measurement if jetson-stats is not available
    """
    # Base path - based on your information
    base_path = "/sys/bus/i2c/drivers/ina3221/1-0040"
    # Find all hwmon directories
    hwmon_paths = glob.glob(f"{base_path}/hwmon/hwmon*")
    if not hwmon_paths:
        logger.debug(f"No hwmon directory found: {base_path}/hwmon/hwmon*")
        return -1.0
    logger.debug(f"Found {len(hwmon_paths)} hwmon directories: {hwmon_paths}")
    # Try each hwmon directory
    for hwmon_path in hwmon_paths:
        # Three power rails - try multiple possible filenames
        rail_files = [
            ("VDD_GPU_SOC", "in0_input", "power0_input"),
            ("VDD_CPU_CV", "in1_input", "power1_input"),
            ("VIN_SYS_5V0", "in2_input", "power2_input")
        ]
        total_power_mw = 0
        power_readings = 0
        for rail_name, in_file, power_file in rail_files:
            # Try in*_input
            in_path = f"{hwmon_path}/{in_file}"
            if os.path.exists(in_path):
                try:
                    with open(in_path, 'r') as f:
                        power_mw = int(f.read().strip())
                        total_power_mw += power_mw
                        power_readings += 1
                        logger.debug(f"Read power from {in_path} ({rail_name}): {power_mw}mW")
                        continue
                except Exception as e:
                    logger.debug(f"Cannot read from {in_path} ({rail_name}): {str(e)}")
            # Try power*_input
            power_path = f"{hwmon_path}/{power_file}"
            if os.path.exists(power_path):
                try:
                    with open(power_path, 'r') as f:
                        power_mw = int(f.read().strip())
                        total_power_mw += power_mw
                        power_readings += 1
                        logger.debug(f"Read power from {power_path} ({rail_name}): {power_mw}mW")
                        continue
                except Exception as e:
                    logger.debug(f"Cannot read from {power_path} ({rail_name}): {str(e)}")
        # If at least one rail was read successfully, return sum
        if power_readings > 0:
            return total_power_mw
    # Try alternative path - directly accessing 1-0040 directory
    for rail_num in range(3):
        path = f"{base_path}/in{rail_num}_input"
        try:
            with open(path, 'r') as f:
                power_mw = int(f.read().strip())
                logger.debug(f"Read power from {path}: {power_mw}mW")
                return power_mw
        except Exception as e:
            logger.debug(f"Cannot read from {path}: {str(e)}")
            continue
    logger.debug("All power measurement attempts failed. Please confirm your device paths.")
    return -1.0


def read_power_watt():
    """Read total system power using jtop for accuracy"""
    # First try using jtop for accurate measurements
    power_w = measure_power_with_jtop()
    # If jtop fails, try the file-based method
    if power_w <= 0:
        power_mw = get_power_from_nvidia_path()
        if power_mw > 0:
            power_w = power_mw / 1000.0  # Convert to watts
    if power_w <= 0:
        logger.warning("Power measurement failed. Using fallback value of 0.5W")
        return 0.5  # Fallback value to prevent division by zero
    return power_w


# --- DNN Model and Profiling Module ---
def calculate_sparsity(tensor):
    if not isinstance(tensor, torch.Tensor):
        return 0.0
    num_non_zeros = torch.count_nonzero(tensor).item()
    total_elements = tensor.numel()
    return 1.0 - (num_non_zeros / total_elements) if total_elements > 0 else 0.0


def get_layer_input_tensor(model, layer_name, input_tensor):
    """Get input tensor for specified layer"""
    intermediate_tensor = None

    def hook_fn(module, input):
        nonlocal intermediate_tensor
        if isinstance(input, tuple):
            input = input[0]
        intermediate_tensor = input.clone().detach()

    target_module = dict(model.named_modules()).get(layer_name)
    if target_module is None:
        logger.error(f"Layer '{layer_name}' not found")
        return None
    handle = target_module.register_forward_pre_hook(hook_fn)
    try:
        with torch.no_grad():
            model(input_tensor)
    except Exception as e:
        logger.error(f"Model execution failed: {str(e)}")
        handle.remove()
        return None
    handle.remove()
    return intermediate_tensor


def get_all_layers(model, model_type="cnn"):
    """Get all executable layers in the model with different strategies for CNNs and Transformers"""
    all_layers = []
    if model_type == "cnn":
        # CNN-specific layer extraction
        for name, module in model.named_modules():
            if name == "":
                continue
            if any(x in name.lower() for x in ["bn", "pool", "downsample"]):
                continue
            if isinstance(module, torch.nn.Sequential):
                continue
            if "fc" in name or "avgpool" in name:
                continue
            all_layers.append(name)
    elif model_type == "transformer":
        # Transformer-specific layer extraction
        for name, module in model.named_modules():
            if name == "":
                continue
            if "mlp" in name or "attn" in name or "norm" in name or "head" in name:
                all_layers.append(name)
    # Limit to 5 layers for testing
    return all_layers[:]


def profile_model(model, model_name, model_type, target_layers, frequency_combinations, input_tensor):
    """Profile model under different frequency configurations with concurrent power measurement"""
    results = []
    freq_manager = FrequencyManager()
    original_profile = freq_manager.get_current_profile()
    logger.info(f"Detected current frequency configuration: {original_profile}")
    # Set CPU governor to userspace (for manual frequency control)
    if not set_cpu_governor():
        logger.error("Cannot set CPU governor to userspace mode, frequency control may not work")
    total_combinations = len(target_layers) * len(frequency_combinations)
    current_combination = 0
    # Ensure initial state stability
    time.sleep(1)
    logger.info("Initial state stabilization complete")
    # Create progress bar
    pbar = tqdm(total=total_combinations, desc=f"Profiling {model_name}")
    for layer_name in target_layers:
        logger.info(f"\nPreparing input tensor for layer '{layer_name}'...")
        try:
            layer_input = get_layer_input_tensor(model, layer_name, input_tensor)
        except Exception as e:
            logger.error(f"Error getting input tensor: {str(e)}")
            continue
        if layer_input is None:
            logger.warning(f"Cannot get input tensor for layer '{layer_name}', skipping")
            continue
        target_module = dict(model.named_modules()).get(layer_name)
        if target_module is None:
            logger.warning(f"Layer '{layer_name}' not found, skipping")
            continue
        for freq_combo in frequency_combinations:
            current_combination += 1
            cpu_freq, gpu_freq = freq_combo
            cpu_min = cpu_max = cpu_freq
            gpu_min = gpu_max = gpu_freq
            logger.info(f"\n[{current_combination}/{total_combinations}] Profiling: "
                        f"Layer='{layer_name}', CPU={cpu_freq / 1000:.0f}MHz, GPU={gpu_freq / 1000000:.0f}MHz")
            try:
                # Attempt to set frequency configuration
                if not set_frequency(cpu_min, cpu_max, gpu_min, gpu_max, 0, 0):
                    logger.error(f"  > Cannot set frequency configuration, skipping")
                    pbar.update(1)
                    continue
                # Additional stabilization time
                time.sleep(0.5)
                cpu_freq_mhz, gpu_freq_mhz, emc_freq_mhz = get_current_frequencies()
                logger.info(
                    f"  > Configuration set. CPU Freq: {cpu_freq_mhz:.1f}MHz, GPU Freq: {gpu_freq_mhz:.2f}MHz, EMC: {emc_freq_mhz:.2f}MHz")
                torch.cuda.synchronize()
                # Warm-up runs
                warmup_runs = 5
                for _ in range(warmup_runs):
                    _ = target_module(layer_input)
                torch.cuda.synchronize()
                # --- Concurrent Power and Latency Measurement ---
                min_profile_runs = 5
                latencies = []
                # Start power monitor
                power_monitor = PowerMonitor(sample_interval=0.02)  # 50Hz sampling rate
                power_monitor.start()
                time.sleep(0.1)  # Give monitor thread a moment to start
                # Perform measurement runs
                for _ in range(min_profile_runs):
                    iter_start = time.perf_counter()
                    output_tensor = target_module(layer_input)
                    torch.cuda.synchronize()
                    iter_end = time.perf_counter()
                    latencies.append(iter_end - iter_start)
                # Stop power monitor and get average power
                avg_power_w = power_monitor.stop()
                if not latencies or avg_power_w <= 0:
                    logger.warning(
                        f"  > Invalid measurement: Not enough valid latency/power readings. Power: {avg_power_w:.2f}W")
                    pbar.update(1)
                    continue
                # Calculate averages
                latency_s = np.mean(latencies)
                energy_j = avg_power_w * latency_s
                logger.info(f"  > Measurement successful: Avg latency: {latency_s * 1000:.2f}ms, "
                            f"Avg power: {avg_power_w:.2f}W, Energy: {energy_j:.4f}J")
                sparsity = calculate_sparsity(output_tensor)
                logger.info(f"  > Sparsity: {sparsity:.2%}")
                results.append({
                    "model": model_name,
                    "model_type": model_type,
                    "layer_name": layer_name,
                    "cpu_freq": cpu_freq_mhz,  # Store in MHz already
                    "gpu_freq": gpu_freq_mhz,  # Store in MHz already
                    "latency_ms": latency_s * 1000,
                    "avg_power_watt": avg_power_w,
                    "energy_joule": energy_j,
                    "sparsity": sparsity
                })
            except Exception as e:
                logger.error(f"  > Profiling failed: {str(e)}")
            finally:
                pbar.update(1)
    # Restore original configuration
    logger.info(f"\nProfiling complete, restoring original frequency configuration {original_profile}...")
    freq_manager.restore_original()
    # Final verification
    time.sleep(1)
    cpu_freq_mhz, gpu_freq_mhz, emc_freq_mhz = get_current_frequencies()
    logger.info(
        f"Restored frequencies: CPU={cpu_freq_mhz:.1f}MHz, GPU={gpu_freq_mhz:.2f}MHz, EMC={emc_freq_mhz:.2f}MHz")
    pbar.close()
    return pd.DataFrame(results)


# --- Data Visualization Module ---
def setup_plot_style():
    """Set up global plot style with Arial font and black text"""
    plt.rcParams.update({
        'font.family': 'Arial',
        'text.color': 'black',
        'axes.labelcolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'legend.fontsize': 14,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.format': 'png',
        'savefig.bbox': 'tight'
    })


def plot_sparsity_distribution(df, model_name):
    """Plot sparsity distribution across layers as a bar chart"""
    if df.empty:
        logger.warning(f"No data for model {model_name}")
        return
    setup_plot_style()
    # Get unique layers and their sparsity (in order of appearance)
    layers = []
    sparsities = []
    # Use the first occurrence of each layer for ordering
    for layer_name in df['layer_name'].unique():
        if layer_name not in layers:
            layer_data = df[df['layer_name'] == layer_name]
            avg_sparsity = layer_data['sparsity'].mean()
            layers.append(layer_name)
            sparsities.append(avg_sparsity)
    # Create the bar plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(layers)), sparsities, color='#0972BA', alpha=0.7)
    plt.xlabel('Layer Index', fontsize=12)
    plt.ylabel('Sparsity', fontsize=12)
    plt.title(f'Sparsity Distribution Across Layers - {model_name}', fontsize=14)
    plt.xticks(range(len(layers)), layers, rotation=45, fontsize=8)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    # Save and show
    filename = f"sparsity_distribution_{model_name}.png"
    plt.savefig(filename)
    logger.info(f"Sparsity distribution plot saved to: {filename}")
    plt.close()


def plot_sparsity_performance(df, model_name):
    """Plot sparsity vs. latency and efficiency for each layer (sorted by index)"""
    if df.empty:
        logger.warning(f"No data for model {model_name}")
        return
    setup_plot_style()
    # Prepare data: average values for each layer (in order of appearance)
    layer_data = []
    for layer_name in df['layer_name'].unique():
        layer_df = df[df['layer_name'] == layer_name]
        avg_sparsity = layer_df['sparsity'].mean()
        avg_latency = layer_df['latency_ms'].mean()
        avg_efficiency = 1 / layer_df['energy_joule'].mean()
        layer_data.append({
            'layer': layer_name,
            'sparsity': avg_sparsity,
            'latency': avg_latency,
            'efficiency': avg_efficiency
        })
    # Sort layers by their index (order of appearance) rather than sparsity
    # The layers are already in order of appearance from the unique() call
    # Create the plot
    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()
    # Plot latency (bar chart)
    layers = [d['layer'] for d in layer_data]
    sparsities = [d['sparsity'] for d in layer_data]
    latencies = [d['latency'] for d in layer_data]
    bars = ax1.bar(range(len(layers)), latencies, color='#0972BA', alpha=0.7)
    ax1.set_xlabel('Layer Index', fontsize=12)
    ax1.set_ylabel('Latency (ms)', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='black')
    # Add sparsity as data labels on the x-axis
    x_labels = [f"{layer}(sparsity:{sparsity:.1%})" for layer, sparsity in zip(layers, sparsities)]
    plt.xticks(range(len(layers)), x_labels, rotation=45, fontsize=8)
    # Create secondary axis for efficiency
    ax2 = ax1.twinx()
    efficiencies = [d['efficiency'] for d in layer_data]
    line = ax2.plot(range(len(layers)), efficiencies, 'o-', color='black', linewidth=2, markersize=4,
                    markerfacecolor='white', markeredgewidth=2, markeredgecolor='black')
    ax2.set_ylabel('Efficiency (inferences per joule)', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='black')
    plt.title(f'Layer Sparsity vs. Performance - {model_name}', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    # Save and show
    filename = f"sparsity_performance_{model_name}.png"
    plt.savefig(filename)
    logger.info(f"Sparsity-performance plot saved to: {filename}")
    plt.close()


def plot_frequency_combination_performance(df, model_name):
    """Plot performance across different frequency combinations for each layer (individual plots)"""
    if df.empty:
        logger.warning(f"No data for model {model_name}")
        return
    setup_plot_style()
    # Get unique layers
    layers = df['layer_name'].unique()
    for layer_name in layers:
        layer_df = df[df['layer_name'] == layer_name]
        # Group by CPU and GPU frequency
        grouped = layer_df.groupby(['cpu_freq', 'gpu_freq'])
        # Calculate average values for each combination
        avg_data = []
        for (cpu, gpu), group in grouped:
            avg_latency = group['latency_ms'].mean()
            avg_power = group['avg_power_watt'].mean()
            avg_data.append({
                'latency': avg_latency,
                'power': avg_power,
                'cpu': cpu,
                'gpu': gpu
            })
        # Sort by CPU frequency (primary) and GPU frequency (secondary)
        avg_data.sort(key=lambda x: (x['cpu'], x['gpu']))
        # Create plot for this layer
        plt.figure(figsize=(16, 8))  # Increased width for better x-axis labels
        ax1 = plt.gca()
        # Extract data for plotting
        cpu_freqs = [d['cpu'] for d in avg_data]
        gpu_freqs = [d['gpu'] for d in avg_data]
        latencies = [d['latency'] for d in avg_data]
        powers = [d['power'] for d in avg_data]

        # FIXED: Create frequency labels with proper precision and format
        # Convert to integers and format as "CPU/GPUMHz"
        freq_labels = [f"{int(cpu)}/{int(gpu)}MHz" for cpu, gpu in zip(cpu_freqs, gpu_freqs)]

        # Plot latency (bar chart)
        bars = ax1.bar(range(len(freq_labels)), latencies, color='#0972BA', alpha=0.7)
        ax1.set_title(f'Layer: {layer_name} - CPU/GPU Frequency Performance', fontsize=16)
        ax1.set_xlabel('CPU/GPU Frequency (MHz)', fontsize=14)
        ax1.set_ylabel('Latency (ms)', fontsize=14)
        ax1.tick_params(axis='y', labelcolor='black')

        # Set x-ticks to frequency combinations with improved layout
        ax1.set_xticks(range(len(freq_labels)))
        ax1.set_xticklabels(freq_labels, rotation=45, fontsize=8, ha='right')

        # Create secondary axis for power
        ax2 = ax1.twinx()
        line = ax2.plot(range(len(freq_labels)), powers, 'o-', color='black', linewidth=2, markersize=4,
                        markerfacecolor='white', markeredgewidth=2, markeredgecolor='black')
        ax2.set_ylabel('Power (W)', fontsize=14)
        ax2.tick_params(axis='y', labelcolor='black')

        # Add grid
        ax1.grid(axis='y', linestyle='--', alpha=0.7)

        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='#0972BA', lw=4, label='Latency (ms)'),
            Line2D([0], [0], color='black', marker='o', lw=2, label='Power (W)')
        ]
        ax1.legend(handles=legend_elements, loc='upper right', fontsize=12)

        # Adjust layout to prevent label clipping
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # Make more space for x-axis labels

        # Save and show
        filename = f"frequency_performance_{model_name}_{layer_name.replace('.', '_')}.png"
        plt.savefig(filename)
        logger.info(f"Frequency-performance plot saved to: {filename}")
        plt.close()

        # Generate heatmaps for this layer
        plot_latency_heatmap(df, model_name, layer_name)
        plot_power_heatmap(df, model_name, layer_name)


# --- Heatmap Visualization Functions ---
def plot_latency_heatmap(df, model_name, layer_name):
    """Plot heatmap of latency across different CPU and GPU frequency combinations for a specific layer"""
    if df.empty:
        logger.warning(f"No data for model {model_name} and layer {layer_name}")
        return
    setup_plot_style()

    # Filter data for the specific layer
    layer_df = df[df['layer_name'] == layer_name]

    # Group by CPU and GPU frequency
    grouped = layer_df.groupby(['cpu_freq', 'gpu_freq'])

    # Prepare data for heatmap
    freq_data = []
    for (cpu, gpu), group in grouped:
        latency = group['latency_ms'].mean()
        freq_data.append({
            'cpu': cpu,
            'gpu': gpu,
            'latency': latency
        })

    if not freq_data:
        logger.warning(f"No frequency data for layer {layer_name}")
        return

    # Create pivot table for heatmap
    df_pivot = pd.DataFrame(freq_data)
    heatmap_data = df_pivot.pivot(index='gpu', columns='cpu', values='latency')

    # Sort frequencies
    sorted_cpu = sorted(heatmap_data.columns)
    sorted_gpu = sorted(heatmap_data.index, reverse=True)  # Reverse for top-to-bottom in heatmap

    # Calculate aspect ratio based on data dimensions
    num_cols = len(sorted_cpu)
    num_rows = len(sorted_gpu)
    # MODIFIED: Reduce base size to make cells smaller
    base_size = 0.25  # Base size per cell in inches
    # Calculate figure dimensions to maintain square cells
    width = base_size * num_cols + 1.5  # Add extra space for labels and colorbar
    height = base_size * num_rows + 1.5  # Add extra space for labels and colorbar

    # Create figure with calculated dimensions
    plt.figure(figsize=(width, height))

    # MODIFIED: Create heatmap with square cells and no annotations
    ax = sns.heatmap(
        heatmap_data.reindex(sorted_gpu).reindex(columns=sorted_cpu),
        cmap='viridis',
        annot=False,  # MODIFIED: Remove numerical values
        linewidths=.5,
        cbar_kws={'label': 'Latency (ms)'},
        square=True,  # This ensures each cell is square
        vmin=heatmap_data.min().min(),  # Set minimum value for color scale
        vmax=heatmap_data.max().max()  # Set maximum value for color scale
    )

    # Format the plot
    plt.title(f'Latency Heatmap - {model_name} Layer: {layer_name}', fontsize=16)
    plt.xlabel('CPU Frequency (MHz)', fontsize=14)
    plt.ylabel('GPU Frequency (MHz)', fontsize=14)

    # Set tick labels with proper formatting
    plt.xticks(ticks=np.arange(len(sorted_cpu)) + 0.5,
               labels=[f"{freq:.0f}" for freq in sorted_cpu],
               rotation=45,
               fontsize=9)

    plt.yticks(ticks=np.arange(len(sorted_gpu)) + 0.5,
               labels=[f"{freq:.0f}" for freq in sorted_gpu],
               rotation=0,
               fontsize=9)

    # Add grid lines between cells (already included by square=True, but we can make them more visible)
    for i in range(len(sorted_gpu) + 1):
        ax.axhline(i, color='k', linewidth=0.5)
    for j in range(len(sorted_cpu) + 1):
        ax.axvline(j, color='k', linewidth=0.5)

    # Ensure equal aspect ratio for the plot
    ax.set_aspect('equal')
    plt.tight_layout()

    # Save and show
    filename = f"latency_heatmap_{model_name}_{layer_name.replace('.', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    logger.info(f"Latency heatmap saved to: {filename}")
    plt.close()


def plot_power_heatmap(df, model_name, layer_name):
    """Plot heatmap of power consumption across different CPU and GPU frequency combinations for a specific layer"""
    if df.empty:
        logger.warning(f"No data for model {model_name} and layer {layer_name}")
        return
    setup_plot_style()

    # Filter data for the specific layer
    layer_df = df[df['layer_name'] == layer_name]

    # Group by CPU and GPU frequency
    grouped = layer_df.groupby(['cpu_freq', 'gpu_freq'])

    # Prepare data for heatmap
    freq_data = []
    for (cpu, gpu), group in grouped:
        power = group['avg_power_watt'].mean()
        freq_data.append({
            'cpu': cpu,
            'gpu': gpu,
            'power': power
        })

    if not freq_data:
        logger.warning(f"No frequency data for layer {layer_name}")
        return

    # Create pivot table for heatmap
    df_pivot = pd.DataFrame(freq_data)
    heatmap_data = df_pivot.pivot(index='gpu', columns='cpu', values='power')

    # Sort frequencies
    sorted_cpu = sorted(heatmap_data.columns)
    sorted_gpu = sorted(heatmap_data.index, reverse=True)  # Reverse for top-to-bottom in heatmap

    # Calculate aspect ratio based on data dimensions
    num_cols = len(sorted_cpu)
    num_rows = len(sorted_gpu)
    # MODIFIED: Reduce base size to make cells smaller
    base_size = 0.5  # Base size per cell in inches
    # Calculate figure dimensions to maintain square cells
    width = base_size * num_cols + 2.5  # Add extra space for labels and colorbar
    height = base_size * num_rows + 1.5  # Add extra space for labels and colorbar

    # Create figure with calculated dimensions
    plt.figure(figsize=(width, height))

    # MODIFIED: Create heatmap with square cells and no annotations
    ax = sns.heatmap(
        heatmap_data.reindex(sorted_gpu).reindex(columns=sorted_cpu),
        cmap='plasma',
        annot=False,  # MODIFIED: Remove numerical values
        linewidths=.5,
        cbar_kws={'label': 'Power (W)'},
        square=True,  # This ensures each cell is square
        vmin=heatmap_data.min().min(),  # Set minimum value for color scale
        vmax=heatmap_data.max().max()  # Set maximum value for color scale
    )

    # Format the plot
    plt.title(f'Power Consumption Heatmap - {model_name} Layer: {layer_name}', fontsize=16)
    plt.xlabel('CPU Frequency (MHz)', fontsize=14)
    plt.ylabel('GPU Frequency (MHz)', fontsize=14)

    # Set tick labels with proper formatting
    plt.xticks(ticks=np.arange(len(sorted_cpu)) + 0.5,
               labels=[f"{freq:.0f}" for freq in sorted_cpu],
               rotation=45,
               fontsize=9)

    plt.yticks(ticks=np.arange(len(sorted_gpu)) + 0.5,
               labels=[f"{freq:.0f}" for freq in sorted_gpu],
               rotation=0,
               fontsize=9)

    # Add grid lines between cells
    for i in range(len(sorted_gpu) + 1):
        ax.axhline(i, color='k', linewidth=0.5)
    for j in range(len(sorted_cpu) + 1):
        ax.axvline(j, color='k', linewidth=0.5)

    # Ensure equal aspect ratio for the plot
    ax.set_aspect('equal')
    plt.tight_layout()

    # Save and show
    filename = f"power_heatmap_{model_name}_{layer_name.replace('.', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    logger.info(f"Power heatmap saved to: {filename}")
    plt.close()


def plot_cpu_performance(df, model_name, fixed_gpu_freq=None):
    """
    Plot performance when CPU frequency varies and GPU frequency is fixed.
    If fixed_gpu_freq is None, uses the median GPU frequency.
    """
    if df.empty:
        logger.warning(f"No data for model {model_name}")
        return
    setup_plot_style()
    # If fixed_gpu_freq is not specified, use the median GPU frequency
    if fixed_gpu_freq is None:
        # Find the closest GPU frequency to the median
        gpu_freqs = df['gpu_freq'].unique()
        if len(gpu_freqs) == 0:
            logger.error(f"No GPU frequency data available for {model_name}")
            return
        # Use the most common value instead of median
        fixed_gpu_freq = df['gpu_freq'].mode()[0]
        logger.info(f"Using most common GPU frequency: {fixed_gpu_freq:.2f}MHz")
    else:
        # Ensure we're using a value that exists in the data
        gpu_freqs = df['gpu_freq'].unique()
        if fixed_gpu_freq not in gpu_freqs:
            # Find the closest existing value
            closest = min(gpu_freqs, key=lambda x: abs(x - fixed_gpu_freq))
            logger.info(f"Using closest existing GPU frequency: {closest:.2f}MHz (requested: {fixed_gpu_freq:.2f}MHz)")
            fixed_gpu_freq = closest
    # Filter data to only include rows with the fixed GPU frequency
    gpu_df = df[df['gpu_freq'] == fixed_gpu_freq]
    if gpu_df.empty:
        logger.warning(f"No data with fixed GPU frequency {fixed_gpu_freq:.2f}MHz")
        return
    # Get unique layers
    layers = gpu_df['layer_name'].unique()
    # Create individual plots for each layer
    for layer_name in layers:
        layer_df = gpu_df[gpu_df['layer_name'] == layer_name]
        # Group by CPU frequency
        grouped = layer_df.groupby('cpu_freq')
        # Calculate average values for each CPU frequency
        cpu_freqs = []
        latencies = []
        powers = []
        for cpu, group in grouped:
            cpu_freqs.append(cpu)
            latencies.append(group['latency_ms'].mean())
            powers.append(group['avg_power_watt'].mean())
        # Sort by CPU frequency
        sorted_indices = np.argsort(cpu_freqs)
        cpu_freqs = [cpu_freqs[i] for i in sorted_indices]
        latencies = [latencies[i] for i in sorted_indices]
        powers = [powers[i] for i in sorted_indices]
        # Create plot
        plt.figure(figsize=(10, 6))
        ax1 = plt.gca()
        # Plot latency (bar chart)
        bars = ax1.bar(range(len(cpu_freqs)), latencies, color='#0972BA', alpha=0.7)
        ax1.set_title(f'Layer: {layer_name} (GPU={fixed_gpu_freq:.2f}MHz)', fontsize=14)
        ax1.set_xlabel('CPU Frequency (MHz)', fontsize=12)
        ax1.set_ylabel('Latency (ms)', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='black')
        # Format x-tick labels with correct precision
        xtick_labels = [f"{cpu:.1f}" for cpu in cpu_freqs]
        ax1.set_xticks(range(len(cpu_freqs)))
        ax1.set_xticklabels(xtick_labels, rotation=45, fontsize=10)
        # Create secondary axis for power
        ax2 = ax1.twinx()
        line = ax2.plot(range(len(cpu_freqs)), powers, 'o-', color='black', linewidth=2, markersize=4,
                        markerfacecolor='white', markeredgewidth=2, markeredgecolor='black')
        ax2.set_ylabel('Power (W)', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='black')
        # Add grid
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='#0972BA', lw=4, label='Latency (ms)'),
            Line2D([0], [0], color='black', marker='o', lw=2, label='Power (W)')
        ]
        ax1.legend(handles=legend_elements, loc='upper right', fontsize=10)
        plt.tight_layout()
        # Save and show
        filename = f"cpu_performance_{model_name}_{layer_name.replace('.', '_')}_gpu{int(fixed_gpu_freq)}.png"
        plt.savefig(filename)
        logger.info(f"CPU-only performance plot saved to: {filename}")
        plt.close()


def plot_gpu_performance(df, model_name, fixed_cpu_freq=None):
    """
    Plot performance when GPU frequency varies and CPU frequency is fixed.
    If fixed_cpu_freq is None, uses the median CPU frequency.
    """
    if df.empty:
        logger.warning(f"No data for model {model_name}")
        return
    setup_plot_style()
    # If fixed_cpu_freq is not specified, use the median CPU frequency
    if fixed_cpu_freq is None:
        # Use the most common value instead of median
        fixed_cpu_freq = df['cpu_freq'].mode()[0]
        logger.info(f"Using most common CPU frequency: {fixed_cpu_freq:.1f}MHz")
    else:
        # Ensure we're using a value that exists in the data
        cpu_freqs = df['cpu_freq'].unique()
        if fixed_cpu_freq not in cpu_freqs:
            # Find the closest existing value
            closest = min(cpu_freqs, key=lambda x: abs(x - fixed_cpu_freq))
            logger.info(f"Using closest existing CPU frequency: {closest:.1f}MHz (requested: {fixed_cpu_freq:.1f}MHz)")
            fixed_cpu_freq = closest
    # Filter data to only include rows with the fixed CPU frequency
    cpu_df = df[df['cpu_freq'] == fixed_cpu_freq]
    if cpu_df.empty:
        logger.warning(f"No data with fixed CPU frequency {fixed_cpu_freq:.1f}MHz")
        return
    # Get unique layers
    layers = cpu_df['layer_name'].unique()
    # Create individual plots for each layer
    for layer_name in layers:
        layer_df = cpu_df[cpu_df['layer_name'] == layer_name]
        # Group by GPU frequency
        grouped = layer_df.groupby('gpu_freq')
        # Calculate average values for each GPU frequency
        gpu_freqs = []
        latencies = []
        powers = []
        for gpu, group in grouped:
            gpu_freqs.append(gpu)
            latencies.append(group['latency_ms'].mean())
            powers.append(group['avg_power_watt'].mean())
        # Sort by GPU frequency
        sorted_indices = np.argsort(gpu_freqs)
        gpu_freqs = [gpu_freqs[i] for i in sorted_indices]
        latencies = [latencies[i] for i in sorted_indices]
        powers = [powers[i] for i in sorted_indices]
        # Create plot
        plt.figure(figsize=(10, 6))
        ax1 = plt.gca()
        # Plot latency (bar chart)
        bars = ax1.bar(range(len(gpu_freqs)), latencies, color='#0972BA', alpha=0.7)
        ax1.set_title(f'Layer: {layer_name} (CPU={fixed_cpu_freq:.1f}MHz)', fontsize=14)
        ax1.set_xlabel('GPU Frequency (MHz)', fontsize=12)
        ax1.set_ylabel('Latency (ms)', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='black')
        # Format x-tick labels with correct precision
        xtick_labels = [f"{gpu:.2f}" for gpu in gpu_freqs]
        ax1.set_xticks(range(len(gpu_freqs)))
        ax1.set_xticklabels(xtick_labels, rotation=45, fontsize=10)
        # Create secondary axis for power
        ax2 = ax1.twinx()
        line = ax2.plot(range(len(gpu_freqs)), powers, 'o-', color='black', linewidth=2, markersize=4,
                        markerfacecolor='white', markeredgewidth=2, markeredgecolor='black')
        ax2.set_ylabel('Power (W)', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='black')
        # Add grid
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='#0972BA', lw=4, label='Latency (ms)'),
            Line2D([0], [0], color='black', marker='o', lw=2, label='Power (W)')
        ]
        ax1.legend(handles=legend_elements, loc='upper right', fontsize=10)
        plt.tight_layout()
        # Save and show
        filename = f"gpu_performance_{model_name}_{layer_name.replace('.', '_')}_cpu{int(fixed_cpu_freq)}.png"
        plt.savefig(filename)
        logger.info(f"GPU-only performance plot saved to: {filename}")
        plt.close()


# --- Main Program ---
if __name__ == "__main__":
    # Check root privileges
    if os.geteuid() != 0:
        logger.error("This script requires root privileges. Please run with 'sudo python3 ...'.")
        exit(1)
    # Set CPU governor
    if not set_cpu_governor():
        logger.error("Cannot set CPU governor to userspace mode, frequency control may be ineffective")
    # Detect current configuration
    freq_manager = FrequencyManager()
    current_profile = freq_manager.get_current_profile()
    logger.info(f"Current frequency configuration: {current_profile}")
    # Display current frequencies
    cpu_freq_mhz, gpu_freq_mhz, emc_freq_mhz = get_current_frequencies()
    logger.info(f"Current frequencies: CPU={cpu_freq_mhz:.1f}MHz, GPU={gpu_freq_mhz:.2f}MHz, EMC={emc_freq_mhz:.2f}MHz")
    # Configuration parameters
    MODEL_NAMES = ["ResNet-18", "ViT-Base"]
    MODEL_TYPES = ["cnn", "transformer"]
    # Load models
    logger.info("\n" + "=" * 60)
    logger.info("Starting Sparsity and DVFS Analysis")
    logger.info("=" * 60)
    models = []
    input_tensor = torch.randn(1, 3, 224, 224).cuda()
    # Load ResNet-18 (CNN model)
    try:
        resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1).cuda().eval()
        models.append(("ResNet-18", "cnn", resnet))
        logger.info("ResNet-18 (CNN) model loaded successfully")
    except Exception as e:
        logger.error(f"ResNet-18 loading failed: {str(e)}")
    # Load ViT (Transformer model) - using PyTorch's built-in model
    try:
        # Try to load from torchvision.models
        vit = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1).cuda().eval()
        models.append(("ViT-Base", "transformer", vit))
        logger.info("ViT-Base (Transformer) model loaded successfully")
    except Exception as e:
        logger.error(f"ViT loading failed: {str(e)}")
        logger.info("Attempting to load ViT from PyTorch Hub...")
        try:
            vit = torch.hub.load('pytorch/vision:v0.10.0', 'vit_b_16', pretrained=True).cuda().eval()
            models.append(("ViT-Base", "transformer", vit))
            logger.info("ViT-Base (Transformer) model loaded successfully from PyTorch Hub")
        except Exception as e:
            logger.error(f"ViT loading from PyTorch Hub failed: {str(e)}")
    if not models:
        logger.error("No models loaded successfully. Exiting.")
        exit(1)
    # Generate frequency combinations at 100MHz intervals
    if not AVAILABLE_FREQUENCIES["cpu"] or not AVAILABLE_FREQUENCIES["gpu"]:
        logger.error("Cannot determine available frequencies. Check your device paths.")
        exit(1)
    # Convert to MHz for display
    cpu_freqs_mhz = [f / 1000 for f in AVAILABLE_FREQUENCIES["cpu"]]
    gpu_freqs_mhz = [f / 1000000 for f in AVAILABLE_FREQUENCIES["gpu"]]
    logger.info(f"Device supported frequency ranges (in MHz):")
    logger.info(f"  CPU: {min(cpu_freqs_mhz):.1f}MHz - {max(cpu_freqs_mhz):.1f}MHz")
    logger.info(f"  GPU: {min(gpu_freqs_mhz):.2f}MHz - {max(gpu_freqs_mhz):.2f}MHz")
    # Use actual available frequencies instead of generating
    frequency_combinations = []
    for cpu in AVAILABLE_FREQUENCIES["cpu"]:
        for gpu in AVAILABLE_FREQUENCIES["gpu"]:
            frequency_combinations.append((cpu, gpu))
    logger.info(f"Will test {len(frequency_combinations)} frequency combinations (actual available frequencies)")
    # Profile each model
    all_results = []
    for model_name, model_type, model in models:
        # Get all testable layers
        TARGET_LAYERS = get_all_layers(model, model_type)
        if not TARGET_LAYERS:
            logger.error(f"No testable layers found for {model_name}")
            continue
        logger.info(f"\nStarting profiling for {model_name} ({model_type})...")
        profiling_results_df = profile_model(
            model, model_name, model_type, TARGET_LAYERS, frequency_combinations, input_tensor
        )
        if not profiling_results_df.empty:
            output_csv = f"profiling_results_{model_name}.csv"
            profiling_results_df.to_csv(output_csv, index=False)
            logger.info(f"Profiling data saved to: {output_csv}")
            all_results.append(profiling_results_df)
            # Generate all required visualizations
            plot_sparsity_distribution(profiling_results_df, model_name)
            plot_sparsity_performance(profiling_results_df, model_name)
            plot_frequency_combination_performance(profiling_results_df, model_name)
            # Generate CPU-only and GPU-only performance plots
            # Use most common frequencies instead of median
            if not profiling_results_df.empty:
                most_common_gpu = profiling_results_df['gpu_freq'].mode()[0]
                most_common_cpu = profiling_results_df['cpu_freq'].mode()[0]
                plot_cpu_performance(profiling_results_df, model_name, fixed_gpu_freq=most_common_gpu)
                plot_gpu_performance(profiling_results_df, model_name, fixed_cpu_freq=most_common_cpu)
        else:
            logger.warning(f"Profiling produced no valid data for {model_name}")
    logger.info("\nAll profiling and visualization completed successfully.")
    logger.info("Check output files for detailed results.")
