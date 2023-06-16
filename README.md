# Pedestrian Dynamics Model

This simulation is based on the CMP model described in the paper `docs/cmp_model.pdf`.

> An off-lattice automaton for modeling pedestrian dynamics is presented. Pedestrians are represented by disks
with variable radius that evolve following predefined rules. The key feature of our approach is that although
positions and velocities are continuous, forces do not need to be calculated. This has the advantage that it
allows using a larger time step than in force-based models. The room evacuation problem and circular racetrack
simulations quantitatively reproduce the available experimental data, both for the specific flow rate and for the
fundamental diagram of pedestrian traffic with an outstanding performance. In this last case, the variation of two
free parameters (rmin and rmax) of the model accounts for the great variety of experimental fundamental diagrams
reported in the literature. Moreover, this variety can be interpreted in terms of these model parameters.

![](https://github.com/eperezok/pedestrian-simulation/blob/main/results/animation.gif)

![](https://github.com/eperezok/pedestrian-simulation/blob/main/results/flow_rate_vs_time.png)

![](https://github.com/eperezok/pedestrian-simulation/blob/main/results/flow_rate_comp.png)

# Requirements

- python >= 3.10
  - numpy
  - matplotlib
  - toml
- java >= 11

# Configuration

Project configuration can be changed modifying the `config.toml` file:

```toml
[simulation]
boxLength = 20     # m
exitWidth = 1.2    # m
pedestrians = 200  # units
minRadius = 0.1    # m
maxRadius = 0.37   # m
beta = 0.9         # unitless
vdMax = 2.0        # m/s
outputInterval = 5 # units

[benchmarks]
exitWidths = [ 3.0, 2.4, 1.8, 1.2,]
pedestrians = [ 380, 320, 260, 200,]

[files]
staticInput = "./static.txt"
output = "./out/output.txt"
benchmark = "./out/benchmark.txt"
```

# Generate particles

To generate random particles to run the simulation, run:

```shell
python generate_pedestrians.py
```

This will generate a `static.txt` file with the (x, y) positions of each particle (pedestrian).

# Run Simulation

To generate the `.jar` file run the following command:

```shell  
mvn clean package
```

In order to run the simulation run:

```shell
java -jar ./target/dinamica-peatonal-1.0-SNAPSHOT-jar-with-dependencies.jar
```

This will generate two files:

1. `out/output.txt`, which contains the position, velocity and radius of each particle in the room for each time step:

```
time_0
x0 y0 v0 r0
x1 y1 v1 r1
...
xn yn vn rn
time_1
x0 y0 v0 r0
x1 y1 v1 r1
...
xn yn vn rn
...
```

2. `out/bencharmk.txt`, which contains the amount of pedestrians that have left the room for each time step:

```
t_0 exits_0
t_1 exits_1
...
t_n exits_n
```

# Run Animation

To run the animations based on the simulation output, execute from the root folder:

```shell
python visualization/visuals.py
```

# Visualize benchmarks

- Visualize **exits per time step**:

```bash
python visualization/exits_per_dt.py
```

- Visualize **flow rate** for different exit widths and pedestrian amounts:

```bash
python visualization/flow_rate.py
```
