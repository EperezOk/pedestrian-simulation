import tomllib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import EllipseCollection

def main() -> None:
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)

    with open(config["files"]["output"], 'r') as file:
        lines = file.readlines()

    EXIT_WIDTH = config["simulation"]["exitWidth"]
    BOX_LENGTH = config["simulation"]["boxLength"]
    MIN_RADIUS = config["simulation"]["minRadius"]
    events = {}
    time = None

    for line in lines:
        data = line.split()

        if len(data) == 1:
            time = float(data[0])
            events[time] = []
        else:
            particle = {
                'x': float(data[0]),
                'y': float(data[1]),
                'v': float(data[2]),
                'radius': float(data[3]),
            }
            events[time].append(particle)

    # Update animation frame
    def update(frame):
        t = list(events.keys())[frame]
        particles = events[t]

        x = [p['x'] for p in particles]
        y = [p['y'] for p in particles]
        diameters = [p['radius'] * 2 for p in particles]

        ax.clear()

        if len(particles) > 0: # if there are no particles, it will be the final frame
            # Create the EllipseCollection as a scatter plot, to be able to use the `diameters` array
            # Ref: https://stackoverflow.com/questions/33094509/correct-sizing-of-markers-in-scatter-plot-to-a-radius-r-in-matplotlib
            ax.add_collection(EllipseCollection(widths=MIN_RADIUS*2, heights=MIN_RADIUS*2, angles=0, units='xy', edgecolors='none', offsets=list(zip(x, y)), transOffset=ax.transData))
            ax.add_collection(EllipseCollection(widths=diameters, heights=diameters, angles=0, units='xy', facecolors='none', edgecolors='k', linestyle='dotted', offsets=list(zip(x, y)), transOffset=ax.transData))


        ax.plot([0, BOX_LENGTH/2 - EXIT_WIDTH/2], [0, 0], color='black')
        ax.plot([BOX_LENGTH/2 + EXIT_WIDTH/2, 20], [0, 0], color='black')

        ax.set_aspect('auto')
        ax.set_xlim([0, 20])
        ax.set_ylim([-2, 20])
        ax.set_title(f'Time: {t:.2f} s', fontsize=18)
        ax.xaxis.set_tick_params(labelsize=18)
        ax.yaxis.set_tick_params(labelsize=18)
        plt.tight_layout()

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 10))

    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=len(events))

    # Save animation as mp4
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=20, bitrate=1800)

    anim.save('out/animation.gif', writer=writer)


if __name__ == '__main__':
    main()