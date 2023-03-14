from n_puzzle_solver import (PuzzleBoardProblem, 
                             HeuristicFunction,
                             get_bfs_euclidean_distance_average_steps_position,
                             get_bfs_manhattan_distance_average_steps_position,
                             get_bfs_misplaced_tiles_average_steps_position,
                             get_astar_euclidean_distance_average_steps_position,
                             get_astar_manhattan_distance_average_steps_position,
                             get_astar_misplaced_tiles_average_steps_position)



class EightPuzzleProblem(PuzzleBoardProblem):
    
    def __init__(self):
        self.n = 3
        self.puzzle_goal_state = [[1, 2, 3],
                                  [4, 5, 6],
                                  [7, 8, 0]]
        
        super().__init__(self.n, self.puzzle_goal_state)
    

if __name__ == "__main__":

    inital_puzzle_conf = list()

    print("Enter initial puzzle configuration : ")
    inital_puzzle_conf = input().split()
    inital_puzzle_conf = [int(i) for i in inital_puzzle_conf]
    print(inital_puzzle_conf)

    puzzle_board = EightPuzzleProblem()
    puzzle_board.set_matrix(inital_puzzle_conf)
    puzzle_board.best_first_search_algorithm(HeuristicFunction.EUCLIDEAN_DISTANCE,
     get_bfs_euclidean_distance_average_steps_position)
    print("Best First Search Euclidean distance average count: ", puzzle_board.average_steps[0][1] / 5)

    puzzle_board = EightPuzzleProblem()
    puzzle_board.set_matrix(inital_puzzle_conf)
    puzzle_board.best_first_search_algorithm(HeuristicFunction.MANHATTAN_DISTANCE,
     get_bfs_manhattan_distance_average_steps_position)
    print("Best First Search Manhattan distance average count: ", puzzle_board.average_steps[0][0] / 5)
   
    puzzle_board = EightPuzzleProblem()
    puzzle_board.set_matrix(inital_puzzle_conf)
    puzzle_board.best_first_search_algorithm(HeuristicFunction.MISPLACED_TILES,
     get_bfs_misplaced_tiles_average_steps_position)
    print("Best First number of Misplaced tiles average count: ", puzzle_board.average_steps[0][2] / 5)

    puzzle_board = EightPuzzleProblem()
    puzzle_board.set_matrix(inital_puzzle_conf)
    puzzle_board.astar_algorithm(HeuristicFunction.EUCLIDEAN_DISTANCE,
    get_astar_euclidean_distance_average_steps_position)
    print("A* search Euclidean distance average count: ", puzzle_board.average_steps[1][1] / 5)

    puzzle_board = EightPuzzleProblem()
    puzzle_board.set_matrix(inital_puzzle_conf)
    puzzle_board.astar_algorithm(HeuristicFunction.MANHATTAN_DISTANCE,
     get_astar_manhattan_distance_average_steps_position)
    print("A* search Manhattan distance averaege count: ", puzzle_board.average_steps[1][0] / 5)

    puzzle_board = EightPuzzleProblem()
    puzzle_board.set_matrix(inital_puzzle_conf)
    puzzle_board.astar_algorithm(HeuristicFunction.MISPLACED_TILES,
     get_astar_misplaced_tiles_average_steps_position)
    print("A* search number of Misplaced tiles average count: ", puzzle_board.average_steps[1][2] / 5)