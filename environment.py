# Task 9 - Grid World Environment


class GridWorld:

    def __init__(self):

        # Create the grid
        self.grid = [
            ["S", ".", ".", "#", "."],
            [".", "#", ".", "#", "."],
            [".", ".", "X", ".", "."],
            [".", "#", ".", "X", "."],
            [".", ".", ".", ".", "G"]
        ]

        # Starting position of the agent
        self.start_position = (0, 0)

        # Current position of the agent
        self.agent_position = self.start_position

        # Actions
        self.UP = 0
        self.DOWN = 1
        self.LEFT = 2
        self.RIGHT = 3


    def move_agent(self, position, action):

        row, column = position

        # Calculate the next position
        if action == self.UP:
            row -= 1

        elif action == self.DOWN:
            row += 1

        elif action == self.LEFT:
            column -= 1

        elif action == self.RIGHT:
            column += 1

        # Check whether the new position is outside the grid
        if (
            row < 0
            or row >= len(self.grid)
            or column < 0
            or column >= len(self.grid[0])
        ):
            return position

        # Check whether the new position is a wall
        if self.grid[row][column] == "#":
            return position

        return row, column


    def step(self, action):

        # Find the new position
        new_position = self.move_agent(
            self.agent_position,
            action
        )

        # Check if the move was blocked
        if new_position == self.agent_position:

            return self.agent_position, -10, False

        # Update agent position
        self.agent_position = new_position

        row, column = new_position

        # Find what is present in the new cell
        cell = self.grid[row][column]

        # Goal
        if cell == "G":
            return new_position, 100, True

        # Trap
        if cell == "X":
            return new_position, -100, True

        # Normal cell
        return new_position, -1, False


    def reset(self):

        # Return agent to starting position
        self.agent_position = self.start_position

        return self.agent_position


# --------------------------------------------------
# Testing the Grid World
# --------------------------------------------------
if __name__ == "__main__":

    environment = GridWorld()

    print("\nTesting RIGHT:")
    position = environment.reset()
    position, reward, done = environment.step(environment.RIGHT)
    print("Position:", position)
    print("Reward:", reward)
    print("Done:", done)

    print("\nTesting trap:")
    environment.reset()
    environment.agent_position = (2, 1)
    position, reward, done = environment.step(environment.RIGHT)
    print("Position:", position)
    print("Reward:", reward)
    print("Done:", done)

    print("\nTesting goal:")
    environment.reset()
    environment.agent_position = (4, 3)
    position, reward, done = environment.step(environment.RIGHT)
    print("Position:", position)
    print("Reward:", reward)
    print("Done:", done)

    print("\nTesting wall:")
    environment.reset()
    environment.agent_position = (0, 2)
    position, reward, done = environment.step(environment.RIGHT)
    print("Position:", position)
    print("Reward:", reward)
    print("Done:", done)

    print("\nTesting boundary:")
    environment.reset()
    position, reward, done = environment.step(environment.UP)
    print("Position:", position)
    print("Reward:", reward)
    print("Done:", done)

    print("\nTesting reset:")
    environment.agent_position = (2, 1)
    position = environment.reset()
    print("New episode starts at:", position)