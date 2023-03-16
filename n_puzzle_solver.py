"""
Implementation of 8-puzzle problem using best-first alogrithm and
A* algorithm with various heuristics functions such as 
   1. Manhattan distance
   2. Euclidean distance
   3. Number of misplaced tiles
"""

from enum import Enum
import math

class SearchAlgorithm(Enum):
    ASTAR = "A star"
    BFS = "Best first search"
    
def get_bfs_manhattan_distance_average_steps_position():
    return 0, 0

def get_bfs_euclidean_distance_average_steps_position():
    return 0, 1

def get_bfs_misplaced_tiles_average_steps_position():
    return 0, 2

def get_astar_manhattan_distance_average_steps_position():
    return 1, 0

def get_astar_euclidean_distance_average_steps_position():
    return 1, 1

def get_astar_misplaced_tiles_average_steps_position():
    return 1, 2

class PuzzleBoardProblem:

    def __init__(self, grid, puzzle_goal_state, max_move_limit):
        self.intial_node = None
        self.duplicate_matrix = []
        self.heuristic_value = 0
        self.depth = 0
        self.grid = grid
        self.puzzle_goal_state = puzzle_goal_state
        self.max_move_limit = max_move_limit
        self.set_average_steps()
        self.set_dup_matrix()

    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return False
        else:
            return self.duplicate_matrix == other.duplicate_matrix

    def __str__(self):
        result = ''
        for row in range(self.grid):
            result += ' '.join(map(str, self.duplicate_matrix[row]))
            result += '\r\n'
        return result

    def set_average_steps(self):
        self.average_steps = [[0, 0, 0],
                              [0, 0, 0]]

    def set_dup_matrix(self):
        for row in range(self.grid):
            self.duplicate_matrix.append(self.puzzle_goal_state[row][:])

    def set_matrix(self, values):
        position = 0
        for row in range(self.grid):
            for col in range(self.grid):
                self.duplicate_matrix[row][col] = int(values[position])
                position = position + 1

    def get_value(self, row, col):
        return self.duplicate_matrix[row][col]

    def set_value(self, row, col, value):
        self.duplicate_matrix[row][col] = value

    def swap_value(self, position_a, position_b):
        temp = self.get_value(*position_a)
        self.set_value(position_a[0], position_a[1],
         self.get_value(*position_b))
        self.set_value(position_b[0], position_b[1], temp)

    def find_row_col_of_value(self, value):
        if value < 0 or value > 8:
            raise Exception("Given value is out of range. Should be between 0 and 8")
        for row in range(self.grid):
            for col in range(self.grid):
                if self.duplicate_matrix[row][col] == value:
                    return row, col
        
    def deep_copy_matrix(self):
        puzzle_board = PuzzleBoardProblem(self.grid, self.puzzle_goal_state, self.max_move_limit)
        for row in range(self.grid):
            puzzle_board.duplicate_matrix[row] = self.duplicate_matrix[row][:]
        return puzzle_board

    def possible_move_actions(self):
        row, col = self.find_row_col_of_value(0)
        possible_move_matrix = []

        if row > 0:
            possible_move_matrix.append((row - 1, col))
        if col > 0:
            possible_move_matrix.append((row, col - 1))
        if row < self.grid -1:
            possible_move_matrix.append((row + 1, col))
        if col < self.grid -1:
            possible_move_matrix.append((row, col + 1))
        return possible_move_matrix

    def create_position(self):
        possible_move_matrix = self.possible_move_actions()
        zero = self.find_row_col_of_value(0)

        def swap_and_copy(a, b):
            pb = self.deep_copy_matrix()
            pb.swap_value(a, b)
            pb.depth = self.depth + 1
            pb.intial_node = self
            return pb

        return map(lambda pair: swap_and_copy(zero, pair), possible_move_matrix)

    def get_solution_path(self, path):
        if self.intial_node is None:
            return path
        else:
            path.append(self)
            return self.intial_node.get_solution_path(path)
    
    def get_index(self, i, inp):
        if i in inp:
            return inp.index(i)
        else:
            return -1
        
    def heuristic_function(self, item_total_cost, total_cost):
        """
        Estimate the cost of reaching the goal state from the given state
        """
        total = 0
        for row in range(self.grid):
            for col in range(self.grid):
                value = self.get_value(row, col) - 1
                goal_row = value / self.grid
                goal_col = value % self.grid
                if goal_row < 0:
                    goal_row = self.grid - 1
                total += item_total_cost(row, goal_row, col, goal_col)
        return total_cost(total)

    def euclidean_distance(self):
        """
        Measures the straight-line distance between two points in a Euclidean space
        """
        return self.heuristic_function(
                        lambda row, goal_row, col, goal_col: math.sqrt(
                            (goal_row - row) ** 2 + (goal_col - col) ** 2),
                        lambda target: target)
    
    def manhattan_distance(self):
        """
        Measures the absolute difference between the coordinates
        of two points in a grid-based system
        """
        return self.heuristic_function(
                        lambda row, goal_row, col, goal_col: abs(goal_row - row) + abs(goal_col - col),
                        lambda target: target)

    def misplaced_tiles(self):
        """
        Measures the number of tiles that are in the wrong position
        in the current state compared to the goal state
        """
        return self.heuristic_function(
                        lambda row, goal_row, col, goal_col: self.get_no_of_misplaced_tiles_count(),
                        lambda target: target)

    def get_no_of_misplaced_tiles_count(self):
        count = 0
        for row in range(3):
            for col in range(3):
                if self.get_value(row, col) != self.puzzle_goal_state[row][col]:
                    count = count + 1
        return count
    
    def solve_puzzle(self, algorithm, heuristic_function_type, get_average_steps_position):
        print(f"\nSolving puzzle using {algorithm.value} {heuristic_function_type.__name__} heuristics:")
        result_path, count = self.search_algorithm(algorithm, heuristic_function_type)
        row, col = get_average_steps_position()
        if result_path:
            result_path.reverse()
            for i in result_path:
                print(i.duplicate_matrix)
            self.average_steps[row][col] = self.average_steps[row][col] + len(result_path)
            print(f"\nNo.of steps {heuristic_function_type.__name__} took for {algorithm.value} is {len(result_path)}")
        else:
            print(f"Searching stopped after traversing {count -1} solution paths")

    def search_algorithm(self, algorithm, given_heuristic_function):
        def check_is_solved(puzzle):
            return puzzle.duplicate_matrix == self.puzzle_goal_state

        given_input_matrix = [self]
        intermediate_matrix = []
        no_of_moves = 0
        while len(given_input_matrix) > 0:
            matrix = given_input_matrix.pop(0)
            no_of_moves += 1

            if no_of_moves > self.max_move_limit:
                print("No solution for the given puzzle is found. Maximum move limit reached")
                return [], no_of_moves
            if check_is_solved(matrix):
                if len(intermediate_matrix) > 0:
                    return matrix.get_solution_path([]), no_of_moves
                else:
                    return [matrix]
            next_possible_position = matrix.create_position()
            matrix_index_open = matrix_index_closed = -1
            for move in next_possible_position:
                matrix_index_open = self.get_index(move, given_input_matrix)
                matrix_index_closed = self.get_index(move, intermediate_matrix)
                heuristic_value = given_heuristic_function(move)
                function_value = heuristic_value + move.depth if algorithm == SearchAlgorithm.ASTAR \
                                    else heuristic_value
                if matrix_index_closed == -1 and matrix_index_open == -1:
                    move.heuristic_value = heuristic_value
                    given_input_matrix.append(move)
                elif matrix_index_open > -1:
                    copy = given_input_matrix[matrix_index_open]
                    copy_function_value = copy.heuristic_value + copy.depth if algorithm == SearchAlgorithm.ASTAR \
                                            else copy.heuristic_value
                    if function_value < copy_function_value:
                        copy.heuristic_value = heuristic_value
                        copy.intial_node = move.intial_node
                        if algorithm == SearchAlgorithm.ASTAR:
                            copy.depth = move.depth
                elif matrix_index_closed > -1:
                    copy = intermediate_matrix[matrix_index_closed]
                    copy_function_value = copy.heuristic_value + copy.depth if algorithm == SearchAlgorithm.ASTAR \
                                            else copy.heuristic_value
                    if function_value < copy_function_value:
                        move.heuristic_value = heuristic_value
                        intermediate_matrix.remove(copy)
                        given_input_matrix.append(move)
            intermediate_matrix.append(matrix)
            given_input_matrix = sorted(given_input_matrix, key=lambda p: p.heuristic_value + p.depth)  \
                                if algorithm == SearchAlgorithm.ASTAR else sorted(given_input_matrix, key=lambda p: p.heuristic_value)
        return [], no_of_moves

class HeuristicFunction(Enum):
    EUCLIDEAN_DISTANCE = PuzzleBoardProblem.euclidean_distance
    MANHATTAN_DISTANCE = PuzzleBoardProblem.manhattan_distance
    MISPLACED_TILES = PuzzleBoardProblem.misplaced_tiles