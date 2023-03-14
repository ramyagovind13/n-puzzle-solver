# Implementation of 8-puzzle problem using best-first alogrithm and
# A* algorithm with various heuristics functions such as 
#   1. Manhattan distance
#   2. Euclidean distance
#   3. Number of misplaced tiles


from enum import Enum
import math

# Set maximum nuber of move limit
max_move_limit = 1500
    
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

    def __init__(self, n, puzzle_goal_state):
        self.intial_node = None
        self.dup_matrix = []
        self.heuristic_value = 0
        self.depth = 0
        self.n = n
        self.puzzle_goal_state = puzzle_goal_state
        self.set_average_steps()
        self.set_dup_matrix()

    def __eq__(self, other):
        # checks the equality of the class obj and copied matrices
        if self.__class__ != other.__class__:
            return False
        else:
            return self.dup_matrix == other.dup_matrix

    def __str__(self):
        # Covert the coned matrix into string representation
        result = ''
        for r in range(self.n):
            result += ' '.join(map(str, self.dup_matrix[r]))
            result += '\r\n'
        return result

    def set_average_steps(self):
        self.average_steps = [[0, 0, 0],
                              [0, 0, 0]]

    def set_dup_matrix(self):
        for i in range(self.n):
            self.dup_matrix.append(self.puzzle_goal_state[i][:])

    def set_matrix(self, values):
        # initialize matrix with the given matrix
        i = 0
        for row in range(self.n):
            for col in range(self.n):
                self.dup_matrix[row][col] = int(values[i])
                i = i + 1

    def get_value(self, row, col):
        # get matrix value
        return self.dup_matrix[row][col]

    def set_value(self, row, col, value):
        # set matrix value
        self.dup_matrix[row][col] = value

    def switch_value(self, position_a, position_b):
        # Swap matrix value
        temp = self.get_value(*position_a)
        self.set_value(position_a[0], position_a[1],
         self.get_value(*position_b))
        self.set_value(position_b[0], position_b[1], temp)

    def find_row_col_of_value(self, val):
        # Return the row and col of the given value
        if val < 0 or val > 8:
            raise Exception("Given value is out of range. Should be between 0 and 8")
        for r in range(self.n):
            for c in range(self.n):
                if self.dup_matrix[r][c] == val:
                    return r, c
        
    def deep_copy_matrix(self):
        # Deep copy matrix
        pb = PuzzleBoardProblem(self.n, self.puzzle_goal_state)
        for r in range(self.n):
            pb.dup_matrix[r] = self.dup_matrix[r][:]
        return pb

    def possible_move_actions(self):
        # choose possible moves such as up,down,left or right
        r, c = self.find_row_col_of_value(0)
        possible_move_matrix = []

        if r > 0:
            possible_move_matrix.append((r - 1, c))
        if c > 0:
            possible_move_matrix.append((r, c - 1))
        if r < self.n -1:
            possible_move_matrix.append((r + 1, c))
        if c < self.n -1:
            possible_move_matrix.append((r, c + 1))
        return possible_move_matrix

    def create_position(self):
        # Generate possible possible 
        possible_move_matrix = self.possible_move_actions()
        zero = self.find_row_col_of_value(0)

        def switch_and_copy(a, b):
            # swapping tiles value with the blank value (zero)
            pb = self.deep_copy_matrix()
            pb.switch_value(a, b)
            pb.depth = self.depth + 1
            pb.intial_node = self
            return pb

        return map(lambda pair: switch_and_copy(zero, pair), possible_move_matrix)

    def get_resultant_path(self, path):
        # Get the solution path
        if self.intial_node is None:
            return path
        else:
            path.append(self)
            return self.intial_node.get_resultant_path(path)
    
    def get_index(self, i, inp):
        if i in inp:
            return inp.index(i)
        else:
            return -1
    
    def best_first_search_algorithm(self, heuristic_function_type, get_average_steps_position):
        # Solving puzzle using Best first search
        print(f"\nSolving puzzle using Best first search {heuristic_function_type.__name__} heuristics:")
        res_path, count = self.best_first_search(heuristic_function_type)
        row, col = get_average_steps_position()
        if res_path:
            res_path.reverse()
            for i in res_path:
                print(i.dup_matrix)
            self.average_steps[row][col] = self.average_steps[row][col] + len(res_path)
            print(f"\nNo.of steps {heuristic_function_type.__name__} took for best first algorithm is {len(res_path)}")
        else:
            print(f"Searching stopped after traversing {count -1} solution paths")

    def astar_algorithm(self, heuristic_function_type, get_average_steps_position):
        # Solving puzzle using A star algorithm
        print(f"\nSolving puzzle using astar {heuristic_function_type.__name__} heuristics:")
        res_path, count = self.astar_search(heuristic_function_type)
        row, col = get_average_steps_position()
        if res_path:
            res_path.reverse()
            for i in res_path:
                print(i.dup_matrix)
            self.average_steps[row][col] = self.average_steps[row][col] + len(res_path)
            print(f"\nNo.of steps {heuristic_function_type.__name__} took for best first algorithm is {len(res_path)}")
        else:
            print(f"Searching stopped after traversing {count -1} solution paths")

    def best_first_search(self, given_heuristic_function):
        # Best First Search implementation
        def check_is_solved(puzzle):
            return puzzle.dup_matrix == self.puzzle_goal_state

        given_input_matrix = [self]
        intermediate_matrix = []
        no_of_moves = 0
        while len(given_input_matrix) > 0:
            v = given_input_matrix.pop(0)
            no_of_moves += 1

            if no_of_moves > max_move_limit:
                print("No solution for the given puzzle is found. Maximum move limit reached")
                return [], no_of_moves
            if check_is_solved(v):
                if len(intermediate_matrix) > 0:
                    return v.get_resultant_path([]), no_of_moves
                else:
                    return [v]
            next_possible_position = v.create_position()
            matrix_index_open = matrix_index_closed = -1
            for move in next_possible_position:
                matrix_index_open = self.get_index(move, given_input_matrix)
                matrix_index_closed = self.get_index(move, intermediate_matrix)
                heuristic_value = given_heuristic_function(move)
                function_value = heuristic_value

                if matrix_index_closed == -1 and matrix_index_open == -1:
                    move.heuristic_value = heuristic_value
                    given_input_matrix.append(move)
                elif matrix_index_open > -1:
                    copy = given_input_matrix[matrix_index_open]
                    if function_value < copy.heuristic_value:
                        copy.heuristic_value = heuristic_value
                        copy.intial_node = move.intial_node
                elif matrix_index_closed > -1:
                    copy = intermediate_matrix[matrix_index_closed]
                    if function_value < copy.heuristic_value:
                        move.heuristic_value = heuristic_value
                        intermediate_matrix.remove(copy)
                        given_input_matrix.append(move)
            intermediate_matrix.append(v)
            given_input_matrix = sorted(given_input_matrix, key=lambda p: p.heuristic_value)
        return [], no_of_moves

    def astar_search(self, given_heuristic_function):
        # A* algorithm implementation
        def check_is_solved(puzzle):
            return puzzle.dup_matrix == self.puzzle_goal_state

        given_input_matrix = [self]
        intermediate_matrix = []
        no_of_moves = 0
        while len(given_input_matrix) > 0:
            v = given_input_matrix.pop(0)
            no_of_moves += 1

            if no_of_moves > max_move_limit:
                print("No solution for the given puzzle is found. Maximum move limit reached")
                return [], no_of_moves

            if check_is_solved(v):
                if len(intermediate_matrix) > 0:
                    return v.get_resultant_path([]), no_of_moves
                else:
                    return [v]
            next_possible_position = v.create_position()
            matrix_index_open = matrix_index_closed = -1
            for move in next_possible_position:
                matrix_index_open = self.get_index(move, given_input_matrix)
                matrix_index_closed = self.get_index(move, intermediate_matrix)
                heuristic_value = given_heuristic_function(move)
                function_value = heuristic_value + move.depth

                if matrix_index_closed == -1 and matrix_index_open == -1:
                    move.heuristic_value = heuristic_value
                    given_input_matrix.append(move)
                
                elif matrix_index_open > -1:
                    copy = given_input_matrix[matrix_index_open]
                    if function_value < copy.heuristic_value + copy.depth:
                        copy.heuristic_value = heuristic_value
                        copy.intial_node = move.intial_node
                        copy.depth = move.depth
                
                elif matrix_index_closed > -1:
                    copy = intermediate_matrix[matrix_index_closed]
                    if function_value < copy.heuristic_value + copy.depth:
                        move.heuristic_value = heuristic_value
                        intermediate_matrix.remove(copy)
                        given_input_matrix.append(move)
            intermediate_matrix.append(v)
            given_input_matrix = sorted(given_input_matrix, key=lambda p: p.heuristic_value + p.depth)
        return [], no_of_moves

    def heuristic_function(self, item_tot_cost, total_cost):
        # estimate the cost of reaching a goal state from a given state
        tot = 0
        for row in range(self.n):
            for col in range(self.n):
                val = self.get_value(row, col) - 1
                goal_row = val / self.n
                goal_col = val % self.n
                if goal_row < 0:
                    goal_row = self.n - 1
                tot += item_tot_cost(row, goal_row, col, goal_col)
        return total_cost(tot)

    def euclidean_distance(self):
        # measures the straight-line distance between two points in a Euclidean space
        return self.heuristic_function(
                        lambda row, goal_row, col, goal_col: math.sqrt(
                            (goal_row - row) ** 2 + (goal_col - col) ** 2),
                        lambda target: target)
        
    def manhattan_distance(self):
        # measures the absolute difference between the coordinates
        # of two points in a grid-based system
        return self.heuristic_function(
                        lambda row, goal_row, col, goal_col: abs(goal_row - row) + abs(goal_col - col),
                        lambda target: target)

    def misplaced_tiles(self):
        # measures the number of tiles that are in the wrong position 
        # in the current state compared to the goal state.
        return self.heuristic_function(
                        lambda row, goal_row, col, goal_col: self.get_no_of_misplaced_tiles_count(),
                        lambda target: target)

    def get_no_of_misplaced_tiles_count(self):
        # Get number of misplaced tiles count
        count = 0
        for row in range(3):
            for col in range(3):
                if self.get_value(row, col) != self.puzzle_goal_state[row][col]:
                    count = count + 1
        return count

class HeuristicFunction(Enum):
    EUCLIDEAN_DISTANCE = PuzzleBoardProblem.euclidean_distance
    MANHATTAN_DISTANCE = PuzzleBoardProblem.manhattan_distance
    MISPLACED_TILES = PuzzleBoardProblem.misplaced_tiles