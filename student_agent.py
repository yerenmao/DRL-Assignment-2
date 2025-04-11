# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
import random
import math

# my imports
from collections import defaultdict


class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()

        self.size = 4  # 4x4 2048 board
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0

        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]

        self.last_move_valid = True  # Record if the last move was valid

        self.reset()

    def reset(self):
        """Reset the environment"""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board

    def add_random_tile(self):
        """Add a random tile (2 or 4) to an empty cell"""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        """Compress the row: move non-zero values to the left"""
        new_row = row[row != 0]  # Remove zeros
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')  # Pad with zeros on the right
        return new_row

    def merge(self, row):
        """Merge adjacent equal numbers in the row"""
        for i in range(len(row) - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                row[i + 1] = 0
                self.score += row[i]
        return row

    def move_left(self):
        """Move the board left"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            new_row = self.compress(self.board[i])
            new_row = self.merge(new_row)
            new_row = self.compress(new_row)
            self.board[i] = new_row
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_right(self):
        """Move the board right"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            # Reverse the row, compress, merge, compress, then reverse back
            reversed_row = self.board[i][::-1]
            reversed_row = self.compress(reversed_row)
            reversed_row = self.merge(reversed_row)
            reversed_row = self.compress(reversed_row)
            self.board[i] = reversed_row[::-1]
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_up(self):
        """Move the board up"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            col = self.compress(self.board[:, j])
            col = self.merge(col)
            col = self.compress(col)
            self.board[:, j] = col
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def move_down(self):
        """Move the board down"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            # Reverse the column, compress, merge, compress, then reverse back
            reversed_col = self.board[:, j][::-1]
            reversed_col = self.compress(reversed_col)
            reversed_col = self.merge(reversed_col)
            reversed_col = self.compress(reversed_col)
            self.board[:, j] = reversed_col[::-1]
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def is_game_over(self):
        """Check if there are no legal moves left"""
        # If there is any empty cell, the game is not over
        if np.any(self.board == 0):
            return False

        # Check horizontally
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j+1]:
                    return False

        # Check vertically
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.board[i, j] == self.board[i+1, j]:
                    return False

        return True

    def step(self, action):
        """Execute one action"""
        assert self.action_space.contains(action), "Invalid action"

        if action == 0:
            moved = self.move_up()
        elif action == 1:
            moved = self.move_down()
        elif action == 2:
            moved = self.move_left()
        elif action == 3:
            moved = self.move_right()
        else:
            moved = False

        self.last_move_valid = moved  # Record if the move was valid

        if moved:
            self.add_random_tile()

        done = self.is_game_over()

        return self.board, self.score, done, {}

    def render(self, mode="human", action=None):
        """
        Render the current board using Matplotlib.
        This function does not check if the action is valid and only displays the current board state.
        """
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)

        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i, j]
                color = COLOR_MAP.get(value, "#3c3a32")  # Default dark color
                text_color = TEXT_COLOR.get(value, "white")
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="black")
                ax.add_patch(rect)

                if value != 0:
                    ax.text(j, i, str(value), ha='center', va='center',
                            fontsize=16, fontweight='bold', color=text_color)
        title = f"score: {self.score}"
        if action is not None:
            title += f" | action: {self.actions[action]}"
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.show()

    def simulate_row_move(self, row):
        """Simulate a left move for a single row"""
        # Compress: move non-zero numbers to the left
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        # Merge: merge adjacent equal numbers (do not update score)
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i + 1] and new_row[i] != 0:
                new_row[i] *= 2
                new_row[i + 1] = 0
        # Compress again
        new_row = new_row[new_row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def is_move_legal(self, action):
        """Check if the specified move is legal (i.e., changes the board)"""
        # Create a copy of the current board state
        temp_board = self.board.copy()

        if action == 0:  # Move up
            for j in range(self.size):
                col = temp_board[:, j]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col
        elif action == 1:  # Move down
            for j in range(self.size):
                # Reverse the column, simulate, then reverse back
                col = temp_board[:, j][::-1]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col[::-1]
        elif action == 2:  # Move left
            for i in range(self.size):
                row = temp_board[i]
                temp_board[i] = self.simulate_row_move(row)
        elif action == 3:  # Move right
            for i in range(self.size):
                row = temp_board[i][::-1]
                new_row = self.simulate_row_move(row)
                temp_board[i] = new_row[::-1]
        else:
            raise ValueError("Invalid action")

        # If the simulated board is different from the current board, the move is legal
        return not np.array_equal(self.board, temp_board)

class MineGame2048Env:
    def __init__(self, board=None, score=None):
        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action = ["up", "down", "left", "right"]
        if board is None:
            self.reset()
        elif score is None:
            self.score = 0
            self.board = board
        else:
            self.score = score
            self.board = board

    def reset(self):
        self.score = 0
        self.board = [0] * 16
        self.add_random_tile()
        self.add_random_tile()
        return self.board

    def add_random_tile(self):
        empty_cells = [i for i, v in enumerate(self.board) if v == 0]
        if not empty_cells:
            return -1
        ni = random.choice(empty_cells)
        nv = 2 if random.random() < 0.9 else 4
        self.board = [nv if i == ni else v for i, v in enumerate(self.board)]
        return 0

    def _compress(self, row):
        return [v for v in row if v != 0]

    def _merge(self, row):
        reward = 0
        row = self._compress(row)
        result, tmp = [], None
        while row:
            v = row.pop(0)
            if tmp is None:  # store
                tmp = v
                continue
            if tmp == v:  # merge
                reward += tmp * 2
                result.append(tmp * 2)
                tmp = None
            else:  # next
                result.append(tmp)
                tmp = v
        if tmp is not None:
            result.append(tmp)
        while len(result) < 4:
            result.append(0)
        self.score += reward
        return reward, result

    def _merge_to_left(self):
        reward = 0
        result = []
        for i in range(4):
            i *= 4
            s, r = self._merge(self.board[i : i + 4])
            reward += s
            result.extend(r)
        self.board = result
        return reward

    def _rotate_90(self):
        result = []
        for i in range(4):
            result.extend(self.board[i::4][::-1])
        self.board = result
        return self

    def act(self, action):
        before = self.board
        if self.action[action] == "up":
            reward = self._rotate_90()._rotate_90()._rotate_90()._merge_to_left()
            self._rotate_90()
        elif self.action[action] == "down":
            reward = self._rotate_90()._merge_to_left()
            self._rotate_90()._rotate_90()._rotate_90()
        elif self.action[action] == "left":
            reward = self._merge_to_left()
        elif self.action[action] == "right":
            reward = self._rotate_90()._rotate_90()._merge_to_left()
            self._rotate_90()._rotate_90()
        moved = tuple(before) != tuple(self.board)
        return reward, moved

    def step(self, action):
        reward, moved = self.act(action)
        afterstate = copy.copy(self.board)
        r = self.add_random_tile() if moved else 0
        done = r < 0
        return (
            self.board,
            self.score,
            done,
            {"afterstate": afterstate, "reward": reward, "moved": moved},
        )

    def render(self):
        for i in range(4):
            i *= 4
            row = self.board[i : i + 4]
            print(" ".join(f"{v:3d}" for v in row))

class NTupleApproximator:
    def __init__(self, board_size, patterns):
        """
        Initializes the N-Tuple approximator.
        Hint: you can adjust these if you want
        """
        self.board_size = board_size
        self.patterns = patterns
        # Create a weight dictionary for each pattern (shared within a pattern group)
        self.weights = [defaultdict(float) for _ in patterns]
        # Generate symmetrical transformations for each pattern
        self.patterns_group = self.generate_symmetries(patterns)

    def generate_symmetries(self, patterns):
        result = []
        for pattern in patterns:
            group = [tuple(r(v) for v in pattern) for r in Transform.rots()]
            flipped = tuple(Transform.flip(v) for v in pattern)
            if pattern != tuple(sorted(flipped)):
                group.extend([tuple(r(v) for v in flipped) for r in Transform.rots()])
            result.append(group)
        return result

    def tile_to_index(self, tile):
        """
        Converts tile values to an index for the lookup table.
        """
        if tile == 0:
            return 0
        else:
            return int(math.log(tile, 2))

    def get_feature(self, board, pattern):
        # TODO: Extract tile values from the board based on the given coordinates and convert them into a feature tuple.
        return tuple(self.tile_to_index(board[i]) for i in pattern)

    def value(self, board):
        # TODO: Estimate the board value: sum the evaluations from all patterns.
        values = []
        for patterns, weight in zip(self.patterns_group, self.weights):
            for pattern in patterns:
                feature = self.get_feature(board, pattern)
                values.append(weight[feature])
        return np.mean(values)

    def update(self, board, delta, alpha):
        # TODO: Update weights based on the TD error.
        for patterns, weight in zip(self.patterns_group, self.weights):
            for pattern in patterns:
                feature = self.get_feature(board, pattern)
                weight[feature] += delta * alpha

    def evaluate(self, state, action):
        env = Game2048Env(state)
        _, _, _, info = env.step(action)
        afterstate, reward, moved = info["afterstate"], info["reward"], info["moved"]
        if moved:
            return reward + self.value(afterstate), info
        return 0, info

    def best_action(self, state):
        action, reward, info = None, float("-inf"), None
        for a in range(4):
            r, i = self.evaluate(state, a)
            if r > reward:
                reward = r
                action = a
                info = i
        return action, info

    def learn(self, state, action, reward, afterstate, next_state, alpha):
        action, info = self.best_action(next_state)
        if info["moved"]:
            next_reward = info["reward"]
            next_afterstate_value = self.value(info["afterstate"])
        else:
            next_reward = 0
            next_afterstate_value = 0

        afterstate_value = self.value(afterstate)
        delta = next_reward + next_afterstate_value - afterstate_value
        self.update(afterstate, delta, alpha)

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.weights, f)

class Max_Node:  # truestate
    def __init__(self, state, score, parent=None, action=None, prob=0):
        """
        state: current board state (numpy array)
        score: cumulative score at this node
        parent: parent node (None for root)
        action: action taken from parent to reach this node
        """
        self.state = state
        self.score = score
        self.parent = parent
        self.action = action
        self.prob = prob
        self.reward = 0
        self.children = {}
        self.value = None
        # for UCT formula
        self.visits = 0
        self.total_reward = 0.0

class Chance_Node:  # afterstate
    def __init__(self, afterstate, score, parent=None, action=None, reward=0):
        self.afterstate = afterstate
        self.score = score
        self.parent = parent
        self.action = action
        self.reward = reward
        self.children = {}
        self.value = None
        # for UCT formula
        self.visits = 0
        self.total_reward = 0.0

class TD_MCTS:
    def __init__(
        self, approximator, iterations=500, exploration_constant=0.06, v_norm=4e5
    ):
        self.approximator = approximator
        self.iterations = iterations
        self.c = exploration_constant
        self.v_norm = v_norm

    def approximate(self, afterstate):
        return self.approximator.value(afterstate)

    def select_child_from_max_node(self, node):
        # Use the UCT formula: Q + c * sqrt(log(parent.visits)/child.visits) to select the best child.
        best_child, best_score = None, float("-inf")
        for child in node.children.values():  # each child is Chance_Node
            if child.visits == 0:
                return child
            uct_score = (child.total_reward / child.visits) + self.c * math.sqrt(
                math.log(node.visits) / child.visits
            )
            if uct_score > best_score:
                best_score = uct_score
                best_child = child
        return best_child

    def select_child_from_chance_node(self, node):
        # select randomly according to the probabilities
        actions, probs = [], []
        for a, n in node.children.items():
            actions.append(a)
            probs.append(n.prob)
        action = random.choices(actions, weights=probs)[0]
        return node.children[action]

    def expand_max_node(self, node):
        max_value = 0
        for action in range(4):
            act_env = MineGame2048Env(node.state, node.score)
            reward, moved = act_env.act(action)
            if moved:
                chance_node = Chance_Node(
                    act_env.board, act_env.score, node, action, reward
                )
                chance_node.value = self.approximate(act_env.board)
                max_value = max(max_value, reward + chance_node.value)
                node.children[action] = chance_node
        node.value = max_value

    def expand_chance_node(self, node):
        empty_cells = [i for i, v in enumerate(node.afterstate) if v == 0]
        num_empty = len(empty_cells)
        for i in empty_cells:
            for v, p in ((2, 0.9), (4, 0.1)):
                board = node.afterstate.copy()
                board[i] = v
                max_node = Max_Node(board, node.score, node, i, p / num_empty)
                node.children[(i, v)] = max_node

    def run_simulation(self, root):
        node = root

        # TODO: Selection: Traverse the tree until reaching an unexpanded node.
        path_reward = 0
        while True:
            if not node.children:
                break
            node = (
                self.select_child_from_max_node(node)
                if isinstance(node, Max_Node)
                else self.select_child_from_chance_node(node)
            )
            path_reward += node.reward

        # TODO: Expansion: If the node is not terminal, expand "all" untried action.
        if isinstance(node, Max_Node):
            self.expand_max_node(node)
        else:
            self.expand_chance_node(node)

        # Rollout
        rollout_reward = (path_reward + node.value) / self.v_norm
        # Backpropagate
        self.backpropagate(node, rollout_reward)

    def backpropagate(self, node, reward):
        # TODO: Propagate the obtained reward back up the tree.
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def best_action(self, root):
        # TODO: get the best action
        best_action, best_reward = None, float("-inf")
        for action, chance_node in root.children.items():
            if chance_node.visits == 0:
                continue
            avg_reward = chance_node.total_reward / chance_node.visits
            if avg_reward > best_reward:
                best_reward = avg_reward
                best_action = action
        return best_action

patterns_6tuple = [
    # type 1:
    (4, 5, 8, 9, 12, 13),
    # type 2:
    (5, 6, 9, 10, 13, 14),
    # type 3:
    (0, 4, 8, 9, 12, 13),
    # type 4:
    (1, 5, 9, 10, 13, 14),
]

approximator = NTupleApproximator(board_size=4, patterns=patterns_6tuple)

path = f"final_stage1.pkl"
with open(path, "rb") as f:
    approx = pickle.load(f)
    for j, loaded_dict in enumerate(approx):
        for key in loaded_dict:
            approximator.weights[j][key] = loaded_dict[key]
    
mcts = TD_MCTS(approximator)

def get_action(state, score):
    board = state.flatten().tolist()
    root = Max_Node(board, score)
    for _ in range(mcts.iterations):
        mcts.run_simulation(root)
    best_act = mcts.best_action(root)
    return best_act