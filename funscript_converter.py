import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.signal import find_peaks
import argparse
import json

def get_freqs(t_vals: np.ndarray, x_vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''
    Get frequency values from time and position values

    Parameters
    ----------
    t_vals : np.ndarray
        Time values
    x_vals : np.ndarray
        Position values

    Returns
    ----------
    tuple[np.ndarray, np.ndarray]
        Tuple of frequency times and frequency values
    '''
    assert len(t_vals) == len(x_vals), "Input arrays must be of the same length"
    assert len(t_vals) > 0, "Input arrays must not be empty"
    peaks, _ = find_peaks(x_vals)
    troughs, _ = find_peaks(-x_vals)  
    extrema_indices = np.sort(np.concatenate((peaks, troughs)))
    # If there are not enough extrema, return empty arrays
    if len(extrema_indices) < 2:
        return np.array([]), np.array([])
    extrema_times = t_vals[extrema_indices]

    periods = np.diff(extrema_times)
    freq_values = 1 / (periods / 1000)
    freq_times = (extrema_times[:-1] + extrema_times[1:]) / 2

    return freq_times, freq_values


def get_amplitude(t_vals: np.ndarray, x_vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    assert len(t_vals) == len(x_vals), "Input arrays must be of the same length"
    assert len(t_vals) > 0, "Input arrays must not be empty"
    peaks, _ = find_peaks(x_vals)
    troughs, _ = find_peaks(-x_vals) 
    extrema_indices = np.sort(np.concatenate((peaks, troughs)))
    # If there are not enough extrema, return empty arrays
    if len(extrema_indices) < 2:
        return np.array([]), np.array([])
    
    extrema_times = t_vals[extrema_indices]
    extrema_values = x_vals[extrema_indices]
    magnitude_values = np.abs(np.diff(extrema_values))
    mag_t_vals = (extrema_times[:-1] + extrema_times[1:]) / 2

    return mag_t_vals, magnitude_values

def get_bias(t_vals: np.ndarray, x_vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    assert len(t_vals) == len(x_vals), "Input arrays must be of the same length"
    assert len(t_vals) > 0, "Input arrays must not be empty"
    first_order = np.gradient(x_vals, t_vals)
    second_order = np.gradient(first_order, t_vals)
    inflection_points = np.where(np.diff(np.sign(second_order)))[0]
    return t_vals[inflection_points], x_vals[inflection_points]


def compress_amplitude(t_vals: np.ndarray, x_vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    assert len(t_vals) == len(x_vals), "Input arrays must be of the same length"
    assert len(t_vals) > 0, "Input arrays must not be empty"

    unique_magnitudes = []
    unique_times = []

    n = len(x_vals)
    for i in range(n):
        # If it's the first element or different from the previous one
        if i == 0 or x_vals[i] != x_vals[i - 1]:
            unique_magnitudes.append(x_vals[i])
            unique_times.append(t_vals[i]) 
        # If it's the last element or different from the next one
        if i == n - 1 or x_vals[i] != x_vals[i + 1]:
            unique_magnitudes.append(x_vals[i])
            unique_times.append(t_vals[i])  
    return np.array(unique_times), np.array(unique_magnitudes)



def compress_interesting_points(t_vals: np.ndarray, x_vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    assert len(t_vals) == len(x_vals), "Input arrays must be of the same length"
    assert len(t_vals) > 0, "Input arrays must not be empty"
    peaks, _ = find_peaks(x_vals)
    troughs, _ = find_peaks(-x_vals)
    
    derivative = np.gradient(x_vals, t_vals) 
    second_derivative = np.gradient(derivative, t_vals)
    critical_points = np.where(np.isclose(derivative, 0, atol=1e-5))[0]
    inflection_points = np.where(np.diff(np.sign(second_derivative)))[0]
    
    key_indices = np.sort(np.concatenate((peaks, troughs, inflection_points, critical_points)))
    key_indices = np.unique(key_indices)
    key_times = t_vals[key_indices]
    key_values = x_vals[key_indices]
    
    return key_times, key_values


def normalize(x_vals: np.ndarray) -> np.ndarray:
    min = np.min(x_vals)
    max = np.max(x_vals)
    return  (x_vals - min) / (max - min) * 100
    




def generate_new_data(fname: str, option: str, fname_out: str, enable_plot: bool=False) -> None:
    try:   
        with open(fname, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("File not found")
        return
    except json.JSONDecodeError:
        print("Invalid format")
        return

    # extract points, interpolate with pchip
    actions = data["actions"]
    actions.sort(key=lambda x: x['at'])
    times = np.array([action['at'] for action in actions])
    positions = np.array([action['pos'] for action in actions])
    pchip_interpolator = PchipInterpolator(times, positions)
    new_times = np.arange(0, times[-1] + 1, 1)  # from 0 to the last time in ms
    interpolated_positions = pchip_interpolator(new_times)

    t = []
    x = []

    if option == "frequency":
        freq_times, freq_vals = get_freqs(new_times, interpolated_positions)
        freq_times, freq_vals = compress_interesting_points(freq_times, freq_vals)
        freq_vals = normalize(freq_vals)
        t = freq_times
        x = freq_vals
    
    if option == "magnitude":
        mag_times, mag_vals = get_amplitude(new_times, interpolated_positions)
        mag_times, mag_vals = compress_amplitude(mag_times, mag_vals)
        mag_vals = normalize(mag_vals)
        t = mag_times
        x = mag_vals

    if option == "trend":
        bias_times, bias_vals = get_bias(new_times, interpolated_positions)
        bias_times, bias_vals = compress_amplitude(bias_times, bias_vals)
        # bias_vals = normalize(bias_vals)
        t = bias_times
        x = bias_vals

    # recreate actions
    new_actions = []
    for i in range(len(t)):
        new_action = {"at": int(t[i]), "pos": int(x[i])} 
        new_actions.append(new_action)
    data["actions"] = new_actions
    

    with open(fname_out, "w") as f:
        json.dump(data, f, indent=2)

    if enable_plot:
        msize = 4
        plt.figure(figsize=(30, 3))
        plt.plot(t, x, label=option, color='green')
        plt.plot(new_times, interpolated_positions, label="original", color='blue')
        # opacity and smaller dots
        plt.plot(t, x, 'o', alpha=0.5, markersize=msize, color='green')
        plt.plot(times, positions, 'o',alpha=0.5, markersize=msize, color='blue')

        plt.xlabel("time (ms)")
        plt.ylabel("value (scaled down by 5 for easier viewing)")
        plt.ylim(0, 300)
        plt.legend()
        plt.show()



def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Process data files.')

    # Add arguments for the command-line version
    parser.add_argument('-i', '--input', type=str, help='Input filename (interactive mode if not provided)')
    parser.add_argument('-o', '--output', type=str, default=False, help='Output filename (interactive mode if not provided)')
    parser.add_argument('-a', '--append-name', type=str, default=False, help='Append some device name (L0, A1, vibe, etc.) to output filename')
    parser.add_argument("-t", "--trend", default=False, action='store_true', help='Use bias/trend/offset option')
    parser.add_argument('-f', '--frequency', default=False, action='store_true', help='Use frequency option')
    parser.add_argument('-m', '--magnitude', default=False, action='store_true', help='Use magnitude option')
    parser.add_argument('-p', '--plot',default=False, action='store_true', help='Enable plot')


    # Parse command-line arguments
    args = parser.parse_args()

    # check if -o or -a are mutuly exclusive
    if args.output and args.append_name:
        print("Error: -o and -a are mutually exclusive.")
        return

    # Check if either input or output files are provided, if not, switch to interactive mode
    if not args.input or not (args.output or args.append_name):
        print("Interactive mode selected. Please provide the following:")
        args.input = input("Enter the input filename: ")
        args.output = input("Enter the output filename: ")

    # Determine the option based on the flags
    if args.frequency + args.magnitude + args.trend > 1:  
        print("Error: Please specify only one of -t (trend), -f (frequency) or -m (magnitude) options.")
        return

    if args.frequency:
        option = 'frequency'
    elif args.magnitude:
        option = 'magnitude'
    elif args.trend:
        option = 'trend'
    else:
        print("Error: You must specify either -f, -m or -t for the option.")
        return
    

    if args.append_name:
        args.output = args.input[:-10] + f".{args.append_name}.funscript"




    # Call the gen_new_data function with the parsed arguments
    generate_new_data(args.input, option, args.output, args.plot)

if __name__ == "__main__":
    main()
