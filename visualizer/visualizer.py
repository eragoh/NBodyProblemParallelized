import matplotlib.pyplot as plt
import matplotlib.animation as animation
import csv
from itertools import islice

# Open the CSV file and create a CSV reader object
csvfile = open('visualizer/body_positions.csv', 'r')
csvreader = csv.reader(csvfile)

# Function to update the data for animation
def update(num, scatter):
    try:
        # Skip 9 rows
        next(islice(csvreader, 999, 999), None)

        row = next(csvreader)
        coords = [float(x) for x in row]
        x = coords[::3]
        y = coords[1::3]
        z = coords[2::3]
        # print(x)
        scatter._offsets3d = (x, y, z)
    except StopIteration:
        print("Reached end of CSV.")
        ani.event_source.stop()  # Stop the animation
    return scatter,


# Initialize plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Initialize scatter plot (initial dummy data)
scatter = ax.scatter([], [], [])

# Set axis limits (you may want to set these based on your specific data)
ax.set_xlim([-3.279e11, 3.279e11])
ax.set_ylim([-3.279e11, 3.279e11])
ax.set_zlim([-3.279e11, 3.279e11])

# Count lines in file
frames = sum(1 for line in csvfile)
csvfile.seek(0)

# Animation
ani = animation.FuncAnimation(fig, update, frames // 1000, fargs=(scatter,), blit=False, interval=1)

plt.show()
