import toml
import subprocess
import numpy as np
import matplotlib.pyplot as plt

def main() -> None:
    # Get cumulative exits per time, for different pedestrians' initial positions
    with open("config.toml", "r") as f:
        config = toml.load(f)

    simulations = run_simulations(config)

    exit_widths = config["benchmarks"]["exitWidths"]
    pedestrians = config["benchmarks"]["pedestrians"]

    # Plot cumulative exits per time, for each exit width
    exit_rate_comp(simulations, exit_widths, pedestrians)

    # Plot flow rate vs exit width
    flow_rates = []
    errors = []

    for d in exit_widths:
        flow_rates.append(simulations[d]["flow_rate"][0])
        errors.append(simulations[d]["flow_rate"][1])

    # Plot flow rate vs exit width, with vertical error bars
    plt.errorbar(exit_widths, flow_rates, yerr=errors, fmt='o', capsize=3, markersize=4, color="black")
    plt.plot(exit_widths, flow_rates)  # Connect markers with straight lines

    plt.xlabel("Exit width (m)", fontsize=14)
    plt.ylabel("Flow rate (people/s)", fontsize=14)

    plt.tight_layout()
    plt.grid()
    plt.savefig("out/flow_rate_vs_d.png")
    plt.show()

    # Make a linear regression over the previous plot
    coefficients = np.polyfit(exit_widths, flow_rates, 1)
    slope = coefficients[0]
    intercept = coefficients[1]
    curve = slope * np.array(exit_widths) + intercept

    plt.errorbar(exit_widths, flow_rates, yerr=errors, fmt='o', capsize=3, markersize=4, color="black")
    plt.plot(exit_widths, curve, color='red', label=f"y = {slope:.2f}d + {intercept:.2f}")

    # Customize the plot
    plt.xlabel("Exit width (m)", fontsize=14)
    plt.ylabel("Flow rate (people/s)", fontsize=14)

    # Save and show the plot
    plt.tight_layout()
    plt.grid()
    plt.legend()
    plt.savefig("out/flow_rate_d_regression.png")
    plt.show()


def run_simulations(config, rounds: int = 3):
    exit_width = config["benchmarks"]["exitWidths"]
    pedestrians = config["benchmarks"]["pedestrians"]
    
    # Average flow rate for each exit width
    simulations = {}
    simulations["rounds"] = rounds

    for i, d in enumerate(exit_width):
        config["simulation"]["exitWidth"] = d
        config["simulation"]["pedestrians"] = pedestrians[i]

        print(f"Running simulation with {d=} and pedestrians={pedestrians[i]}")

        with open("config.toml", "w") as f:
            toml.dump(config, f)

        simulations[d] = {}

        for j in range(rounds):
            # Create particles
            subprocess.run(["python", "generate_pedestrians.py"])

            # Run simulation
            subprocess.run(["java", "-jar", "./target/dinamica-peatonal-1.0-SNAPSHOT-jar-with-dependencies.jar"])

            # Save exits per dt
            with open(config["files"]["benchmark"], 'r') as file:
                lines = file.readlines()
            
            simulations[d][j] = {}
            simulations[d][j]["times"] = []
            simulations[d][j]["exits"] = []

            for line in lines:
                data = line.split()
                time = float(data[0])
                cumulative_exits = int(data[1])
                simulations[d][j]["times"].append(time)
                simulations[d][j]["exits"].append(cumulative_exits)
            
            # Get the stationary period, between 10 and 45 seconds
            lower_bound = np.where(np.array(simulations[d][j]["times"]) >= 10)[0][0]
            upper_bound = np.where(np.array(simulations[d][j]["times"]) <= 45)[0][-1]
            stationary_times = np.array(simulations[d][j]["times"])
            stationary_exits = np.array(simulations[d][j]["exits"])

            # Get the flow rate over time, for the stationary period
            dt = 0.025
            time_window_steps = int(10 / dt) # 10 seconds
            time_steps = int(1 / dt)         # 1 second

            flow_rates = []
            times = []

            for k in range(0, len(stationary_times) - time_window_steps + 1, time_steps):
                exits = stationary_exits[k:k+time_window_steps]
                # Get the amount of people that exited in the time window (people / second)
                flow_rate = (exits[-1] - exits[0]) / (time_window_steps * dt)
                flow_rates.append(flow_rate)
                times.append(k * dt)
            
            # Plot a sample of the flow rate over time
            if j == 0:
                plt.plot(times, flow_rates, marker="o", markersize=3, markerfacecolor="black", markeredgecolor="black", label=f"d={d} ; N={pedestrians[i]}")
            
            # Save the flow rate for the current simulation
            simulations[d][j]["flow_rate"] = np.mean(flow_rates)
        
        # Get the average flow rate for current exit width, and the standard deviation
        flow_rates = [simulations[d][j]["flow_rate"] for j in range(rounds)]
        simulations[d]["flow_rate"] = (np.mean(flow_rates), np.std(flow_rates))
    
    plt.xlabel("Exit width (m)", fontsize=14)
    plt.ylabel("Flow rate (people/s)", fontsize=14)
    plt.tight_layout()
    plt.grid()
    plt.legend()
    plt.savefig("out/flow_rate_vs_time.png")
    plt.show()

    return simulations


def exit_rate_comp(simulations, exit_widths, pedestrians):
    # Plot cumulative exits per time
    for i, d in enumerate(exit_widths):
        max_exits = pedestrians[i]
        exits_step = max_exits // 20
        average_times = {}

        for j in range(simulations["rounds"]):
            times = np.array(simulations[d][j]["times"])
            exits = np.array(simulations[d][j]["exits"])

            for exit_count in range(0, max_exits + 1, exits_step):
                # Find the first occurrance of the given step in the exits array
                index = np.where(exits >= exit_count)[0][0] # the equality might not exist, so we take the first greater value
                try:
                    average_times[exit_count].append(times[index])
                except KeyError:
                    average_times[exit_count] = [times[index]]
        
        # Iterate over the `average_times` keys and values, and compute the average time for each exit count
        exits = []
        times = []
        errors = []

        for exit_count, _times in average_times.items():
            average_time = np.mean(_times)
            std = np.std(_times)
            exits.append(exit_count)
            times.append(average_time)
            errors.append(std)
        
        # Plot exit count vs average time, with horizontal error bars
        plt.errorbar(times, exits, xerr=errors, fmt='o', capsize=3, markersize=4, color="black")
        plt.plot(times, exits, label=f"d={d} ; N={max_exits}")

    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("Exits", fontsize=14)

    plt.tight_layout()
    plt.grid()
    plt.legend()
    plt.savefig("out/flow_rate_comp.png")
    plt.show()


if __name__ == '__main__':
    main()