from copy import deepcopy
from heapq import heappush, heappop
import time
import argparse
import sys

# ====================================================================================

char_goal = '1'
char_single = '2'


class Piece:
    """
    This represents a piece on the Hua Rong Dao puzzle.
    """

    def __init__(self, is_goal, is_single, coord_x, coord_y, orientation):
        """
        :param is_goal: True if the piece is the goal piece and False otherwise.
        :type is_goal: bool
        :param is_single: True if this piece is a 1x1 piece and False otherwise.
        :type is_single: bool
        :param coord_x: The x coordinate of the top left corner of the piece.
        :type coord_x: int
        :param coord_y: The y coordinate of the top left corner of the piece.
        :type coord_y: int
        :param orientation: The orientation of the piece (one of 'h'qdor 'v')
            if the piece is a 1x2 piece. Otherwise, this is None
        :type orientation: str
        """

        self.is_goal = is_goal
        self.is_single = is_single
        self.coord_x = coord_x
        self.coord_y = coord_y
        self.orientation = orientation

    def __repr__(self):
        return '{} {} {} {} {}'.format(self.is_goal, self.is_single, self.coord_x, self.coord_y,
                                       self.orientation)


class Board:
    """
    Board class for setting up the playing board.
    """

    def __init__(self, pieces):
        """
        :param pieces: The list of Pieces
        :type pieces: List[Piece]
        """

        self.width = 4
        self.height = 5

        self.pieces = pieces

        # self.grid is a 2-d (size * size) array automatically generated
        # using the information on the pieces when a board is being created.
        # A grid contains the symbol for representing the pieces on the board.
        self.grid = []
        self.__construct_grid()

    def __construct_grid(self):
        """
        Called in __init__ to set up a 2-d grid based on the piece location information.

        """

        for _ in range(self.height):
            line = []
            for _ in range(self.width):
                line.append('.')
            self.grid.append(line)

        for piece in self.pieces:
            if piece.is_goal:
                self.grid[piece.coord_y][piece.coord_x] = char_goal
                self.grid[piece.coord_y][piece.coord_x + 1] = char_goal
                self.grid[piece.coord_y + 1][piece.coord_x] = char_goal
                self.grid[piece.coord_y + 1][piece.coord_x + 1] = char_goal
            elif piece.is_single:
                self.grid[piece.coord_y][piece.coord_x] = char_single
            else:
                if piece.orientation == 'h':
                    self.grid[piece.coord_y][piece.coord_x] = '<'
                    self.grid[piece.coord_y][piece.coord_x + 1] = '>'
                elif piece.orientation == 'v':
                    self.grid[piece.coord_y][piece.coord_x] = '^'
                    self.grid[piece.coord_y + 1][piece.coord_x] = 'v'

    def display(self):
        """
        Print out the current board.

        """
        for _, line in enumerate(self.grid):
            for ch in line:
                print(ch, end='')
            print()

    def empty(self):
        x = []
        y = []

        for i in range(self.height):
            for j in range(self.width):
                if len(x) == 2:
                    break
                if self.grid[i][j] == '.':
                    x.append(j)
                    y.append(i)

            if len(x) == 2:
                break

        return x, y

    def valid_move_up(self, p):
        x = p.coord_x
        y = p.coord_y

        y += -1

        if y < 0:
            return False

        if p.is_goal or (p.orientation == 'h'):
            if (self.grid[y][x] == '.') and (self.grid[y][x + 1] == '.'):
                return True
            else:
                return False
        elif p.is_single or (p.orientation == 'v'):
            if self.grid[y][x] == '.':
                return True
            else:
                return False

    def valid_move_bottom(self, p):
        x = p.coord_x
        y = p.coord_y

        y += 1

        if y > 4:
            return False

        if p.is_goal:
            if y > 3:
                return False
            if (self.grid[y + 1][x] == '.') and (self.grid[y + 1][x + 1] == '.'):
                return True
            else:
                return False
        elif p.orientation == 'h':
            if (self.grid[y][x] == '.') and (self.grid[y][x + 1] == '.'):
                return True
            else:
                return False
        elif p.is_single:
            if self.grid[y][x] == '.':
                return True
            else:
                return False
        elif p.orientation == 'v':
            if y > 3:
                return False
            if self.grid[y + 1][x] == '.':
                return True
            else:
                return False

    def valid_move_right(self, p):
        x = p.coord_x
        y = p.coord_y

        x += 1

        if x > 3:
            return False

        if p.is_goal:
            if x > 2:
                return False

            if (self.grid[y][x + 1] == '.') and (self.grid[y + 1][x + 1] == '.'):
                return True
            else:
                return False
        elif p.orientation == 'v':
            if (self.grid[y][x] == '.') and (self.grid[y + 1][x] == '.'):
                return True
            else:
                return False
        elif p.is_single:
            if self.grid[y][x] == '.':
                return True
            else:
                return False
        elif p.orientation == 'h':
            if x > 2:
                return False

            if self.grid[y][x + 1] == '.':
                return True
            else:
                return False

    def valid_move_left(self, p):
        x = p.coord_x
        y = p.coord_y

        x += -1

        if x < 0:
            return False

        if p.is_goal or (p.orientation == 'v'):
            if (self.grid[y][x] == '.') and (self.grid[y + 1][x] == '.'):
                return True
            else:
                return False

        elif p.is_single or (p.orientation == 'h'):
            if self.grid[y][x] == '.':
                return True
            else:
                return False

    def move_up(self, p):
        x = p.coord_x
        y = p.coord_y

        if p.is_goal:
            self.grid[y - 1][x] = self.grid[y][x]
            self.grid[y - 1][x + 1] = self.grid[y][x + 1]
            self.grid[y][x] = self.grid[y + 1][x]
            self.grid[y][x + 1] = self.grid[y + 1][x + 1]
            self.grid[y + 1][x] = '.'
            self.grid[y + 1][x + 1] = '.'
        elif p.is_single:
            self.grid[y - 1][x] = self.grid[y][x]
            self.grid[y][x] = '.'
        elif p.orientation == 'v':
            self.grid[y - 1][x] = self.grid[y][x]
            self.grid[y][x] = self.grid[y + 1][x]
            self.grid[y + 1][x] = '.'
        elif p.orientation == 'h':
            self.grid[y - 1][x] = self.grid[y][x]
            self.grid[y - 1][x + 1] = self.grid[y][x + 1]
            self.grid[y][x] = '.'
            self.grid[y][x + 1] = '.'

        for piece in self.pieces:
            if (piece.coord_x == x) and (piece.coord_y == y):
                piece.coord_y += -1
                break

    def move_bottom(self, p):
        x = p.coord_x
        y = p.coord_y

        if p.is_goal:
            self.grid[y + 2][x] = self.grid[y][x]
            self.grid[y + 2][x + 1] = self.grid[y][x + 1]
            self.grid[y + 1][x] = self.grid[y][x]
            self.grid[y + 1][x + 1] = self.grid[y][x + 1]
            self.grid[y][x] = '.'
            self.grid[y][x + 1] = '.'
        elif p.is_single:
            self.grid[y + 1][x] = self.grid[y][x]
            self.grid[y][x] = '.'
        elif p.orientation == 'v':
            self.grid[y + 2][x] = self.grid[y + 1][x]
            self.grid[y + 1][x] = self.grid[y][x]
            self.grid[y][x] = '.'
        elif p.orientation == 'h':
            self.grid[y + 1][x] = self.grid[y][x]
            self.grid[y + 1][x + 1] = self.grid[y][x + 1]
            self.grid[y][x] = '.'
            self.grid[y][x + 1] = '.'

        for piece in self.pieces:
            if (piece.coord_x == x) and (piece.coord_y == y):
                piece.coord_y += 1
                break

    def move_left(self, p):
        x = p.coord_x
        y = p.coord_y

        if p.is_goal:
            self.grid[y][x - 1] = self.grid[y][x]
            self.grid[y + 1][x - 1] = self.grid[y + 1][x]
            self.grid[y][x] = self.grid[y][x + 1]
            self.grid[y + 1][x] = self.grid[y + 1][x + 1]
            self.grid[y][x + 1] = '.'
            self.grid[y + 1][x + 1] = '.'
        elif p.is_single:
            self.grid[y][x - 1] = self.grid[y][x]
            self.grid[y][x] = '.'
        elif p.orientation == 'v':
            self.grid[y][x - 1] = self.grid[y][x]
            self.grid[y + 1][x - 1] = self.grid[y + 1][x]
            self.grid[y][x] = '.'
            self.grid[y + 1][x] = '.'
        elif p.orientation == 'h':
            self.grid[y][x - 1] = self.grid[y][x]
            self.grid[y][x] = self.grid[y][x + 1]
            self.grid[y][x + 1] = '.'

        for piece in self.pieces:
            if (piece.coord_x == x) and (piece.coord_y == y):
                piece.coord_x += -1
                break

    def move_right(self, p):
        x = p.coord_x
        y = p.coord_y

        if p.is_goal:
            self.grid[y][x + 2] = self.grid[y][x + 1]
            self.grid[y + 1][x + 2] = self.grid[y + 1][x + 1]
            self.grid[y][x + 1] = self.grid[y][x]
            self.grid[y + 1][x + 1] = self.grid[y + 1][x]
            self.grid[y][x] = '.'
            self.grid[y + 1][x] = '.'
        elif p.is_single:
            self.grid[y][x + 1] = self.grid[y][x]
            self.grid[y][x] = '.'
        elif p.orientation == 'v':
            self.grid[y][x + 1] = self.grid[y][x]
            self.grid[y + 1][x + 1] = self.grid[y + 1][x]
            self.grid[y][x] = '.'
            self.grid[y + 1][x] = '.'
        elif p.orientation == 'h':
            self.grid[y][x + 2] = self.grid[y][x + 1]
            self.grid[y][x + 1] = self.grid[y][x]
            self.grid[y][x] = '.'

        for piece in self.pieces:
            if (piece.coord_x == x) and (piece.coord_y == y):
                piece.coord_x += 1
                break


class State:
    """
    State class wrapping a Board with some extra current state information.
    Note that State and Board are different. Board has the locations of the pieces.
    State has a Board and some extra information that is relevant to the search:
    heuristic function, f value, current depth and parent.
    """

    def __init__(self, board, f, depth, parent=None):
        """
        :param board: The board of the state.
        :type board: Board
        :param f: The f value of current state.
        :type f: int
        :param depth: The depth of current state in the search tree.
        :type depth: int
        :param parent: The parent of current state.
        :type parent: Optional[State]
        """
        self.board = board
        self.f = f
        self.depth = depth
        self.parent = parent
        self.id = hash(board)  # The id for breaking ties.


def goal_test(s):
    state = s.board
    return (state.grid[4][1] == "1") and (state.grid[4][2] == "1")


def heuristic(s):
    global x, y
    state = s.board

    for piece in state.pieces:
        if piece.is_goal:
            x = piece.coord_x
            y = piece.coord_y
            break

    man_x = abs(1 - x)
    man_y = abs(3 - y)

    s.f = man_y + man_x + s.depth

    return man_y + man_x


def successor(s):
    states = []

    d = s.depth
    b = s.board

    for piece in b.pieces:
        if b.valid_move_left(piece):
            new_board = deepcopy(b)
            new_board.move_left(piece)
            new_state = State(new_board, 0, d + 1, s)
            new_state.f = heuristic(new_state) + new_state.depth
            states.append(new_state)
        if b.valid_move_right(piece):
            new_board = deepcopy(b)
            new_board.move_right(piece)
            new_state = State(new_board, 0, d + 1, s)
            new_state.f = heuristic(new_state) + new_state.depth
            states.append(new_state)
        if b.valid_move_up(piece):
            new_board = deepcopy(b)
            new_board.move_up(piece)
            new_state = State(new_board, 0, d + 1, s)
            new_state.f = heuristic(new_state) + new_state.depth
            states.append(new_state)
        if b.valid_move_bottom(piece):
            new_board = deepcopy(b)
            new_board.move_bottom(piece)
            new_state = State(new_board, 0, d + 1, s)
            new_state.f = heuristic(new_state) + new_state.depth
            states.append(new_state)

    return states


def dfs(init_state):
    frontier = [init_state]
    explored = set()

    while len(frontier) != 0:
        curr = frontier.pop()
        str_board = " ".join(sum(curr.board.grid, []))
        if str_board not in explored:
            explored.add(str_board)
            if goal_test(curr):
                return curr
            frontier = frontier + successor(curr)

    return None


def a_star(init_state):
    frontier = []
    explored = set()
    heappush(frontier, (init_state.f, init_state.id, init_state))

    while len(frontier) != 0:
        _, _, curr = heappop(frontier)

        str_board = " ".join(sum(curr.board.grid, []))
        if str_board not in explored:
            explored.add(str_board)
            if goal_test(curr):
                return curr
            next_states = successor(curr)
            for s in next_states:
                heappush(frontier, (s.f, s.id, s))

    return None


def get_solution(goal):
    curr = goal
    seq = [curr]
    while curr.parent is not None:
        seq.append(curr.parent)
        curr = curr.parent

    seq.reverse()
    return seq


def read_from_file(filename):
    """
    Load initial board from a given file.

    :param filename: The name of the given file.
    :type filename: str
    :return: A loaded board
    :rtype: Board
    """

    puzzle_file = open(filename, "r")

    line_index = 0
    pieces = []
    g_found = False

    for line in puzzle_file:

        for x, ch in enumerate(line):

            if ch == '^':  # found vertical piece
                pieces.append(Piece(False, False, x, line_index, 'v'))
            elif ch == '<':  # found horizontal piece
                pieces.append(Piece(False, False, x, line_index, 'h'))
            elif ch == char_single:
                pieces.append(Piece(False, True, x, line_index, None))
            elif ch == char_goal:
                if g_found is False:
                    pieces.append(Piece(True, False, x, line_index, None))
                    g_found = True
        line_index += 1

    puzzle_file.close()

    board = Board(pieces)

    return board


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputfile",
        type=str,
        required=True,
        help="The input file that contains the puzzle."
    )
    parser.add_argument(
        "--outputfile",
        type=str,
        required=True,
        help="The output file that contains the solution."
    )
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=['astar', 'dfs'],
        help="The searching algorithm."
    )
    args = parser.parse_args()

    # read the board from the file
    board = read_from_file(args.inputfile)

    s = State(board, 0, 0)
    f = heuristic(s)
    s.f = f

    output_file = open(args.outputfile, "w")

    if args.algo == 'astar':
        astar_solution = a_star(s)

        if astar_solution is None:
            output_file.write("No solution found \n")
        else:
            l2 = get_solution(astar_solution)
            for state in l2:
                for _, line in enumerate(state.board.grid):
                    for ch in line:
                        output_file.write(ch)
                        output_file.write('')
                    output_file.write('\n')
                output_file.write('\n\n')

    elif args.algo == 'dfs':
        dfs_solution = dfs(s)

        if dfs_solution is None:
            output_file.write("No solution found \n")
        else:
            l1 = get_solution(dfs_solution)
            for state in l1:
                for _, line in enumerate(state.board.grid):
                    for ch in line:
                        output_file.write(ch)
                        output_file.write('')
                    output_file.write('\n')
                output_file.write('\n\n')
