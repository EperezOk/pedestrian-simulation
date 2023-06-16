import toml
import subprocess
import numpy as np
import matplotlib.pyplot as plt

def main() -> None:
    # Get cumulative exits per time, for different pedestrians' initial positions
    rounds = 10
    simulations = run_simulations(rounds)

    # Plot cumulative exits per time
    for j in range(rounds):
        times = simulations[j]["times"]
        exits = simulations[j]["exits"]

        plt.plot(times, exits)

    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("Exits", fontsize=14)

    plt.tight_layout()
    plt.grid()
    plt.savefig("out/exits_per_dt.png")
    plt.show()

    # Compute the average time for each amount of exits, given a step of 10 exits.
    # E.g. average time for 10 exits, average time for 20 exits, etc.
    exits_step = 10
    max_exits = 200
    average_times = {}

    for j in range(rounds):
        times = np.array(simulations[j]["times"])
        exits = np.array(simulations[j]["exits"])

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
    plt.plot(times, exits)  # Connect markers with straight lines

    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("Exits", fontsize=14)

    plt.tight_layout()
    plt.grid()
    plt.savefig("out/avg_exits_per_dt.png")
    plt.show()


def run_simulations(rounds: int = 5):
    with open("config.toml", "r") as f:
        config = toml.load(f)
    
    simulations = {}

    for j in range(rounds):
        simulations[j] = {}
        
        # Create particles
        subprocess.run(["python", "generate_pedestrians.py"])

        # Run simulation
        subprocess.run(["java", "-jar", "./target/dinamica-peatonal-1.0-SNAPSHOT-jar-with-dependencies.jar"])

        # Save event times
        with open(config["files"]["benchmark"], 'r') as file:
            lines = file.readlines()

        simulations[j]["times"] = []
        simulations[j]["exits"] = []

        for line in lines:
            data = line.split()
            time = float(data[0])
            cumulative_exits = int(data[1])
            simulations[j]["times"].append(time)
            simulations[j]["exits"].append(cumulative_exits)
    
    return simulations


if __name__ == '__main__':
    main()