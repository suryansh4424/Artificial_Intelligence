import math
EMPTY = 0
PLAYER_X = 1
PLAYER_O = -1
def game_over(board):
    for i in range(3):
        if sum(board[i]) == 3 or sum(board[i]) == -3:
            return True
        if sum(board[j][i] for j in range(3)) == 3 or sum(board[j][i] for j in range(3)) == -3:
            return True
    if board[0][0] + board[1][1] + board[2][2] == 3 or board[0][2] + board[1][1] + board[2][0] == 3:
        return True
    if board[0][0] + board[1][1] + board[2][2] == -3 or board[0][2] + board[1][1] + board[2][0] == -3:
        return True
    if all(board[i][j] != EMPTY for i in range(3) for j in range(3)):
        return True
    return False

def available_moves(board):
    moves = []
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                moves.append((i, j))
    return moves

def evaluate(board):
    for i in range(3):
        if sum(board[i]) == 3 or sum(board[i]) == -3:
            return sum(board[i]) / 3
        if sum(board[j][i] for j in range(3)) == 3 or sum(board[j][i] for j in range(3)) == -3:
            return sum(board[j][i] for j in range(3)) / 3
    if board[0][0] + board[1][1] + board[2][2] == 3 or board[0][2] + board[1][1] + board[2][0] == 3:
        return 1
    if board[0][0] + board[1][1] + board[2][2] == -3 or board[0][2] + board[1][1] + board[2][0] == -3:
        return -1
    return 0

def minimax(board, depth, alpha, beta, maximizing_player):
    if game_over(board) or depth == 0:
        return evaluate(board)

    if maximizing_player:
        max_eval = -math.inf
        for move in available_moves(board):
            board[move[0]][move[1]] = PLAYER_X
            eval = minimax(board, depth - 1, alpha, beta, False)
            board[move[0]][move[1]] = EMPTY
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = math.inf
        for move in available_moves(board):
            board[move[0]][move[1]] = PLAYER_O
            eval = minimax(board, depth - 1, alpha, beta, True)
            board[move[0]][move[1]] = EMPTY
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

def get_best_move(board):
    best_move = None
    best_eval = -math.inf
    for move in available_moves(board):
        board[move[0]][move[1]] = PLAYER_X
        eval = minimax(board, 5, -math.inf, math.inf, False)
        board[move[0]][move[1]] = EMPTY
        if eval > best_eval:
            best_eval = eval
            best_move = move
    return best_move

def print_board(board):
    for row in board:
        print(" | ".join(["X" if cell == PLAYER_X else "O" if cell == PLAYER_O else " " for cell in row]))
        print("-" * 5)

def play_game():
    board = [[EMPTY] * 3 for _ in range(3)]
    while not game_over(board):
        print_board(board)
        x, y = map(int, input("Enter your move (row column): ").split())
        board[x][y] = PLAYER_O
        if game_over(board):
            break
        print_board(board)
        print("Computer's turn...")
        move = get_best_move(board)
        board[move[0]][move[1]] = PLAYER_X
    print_board(board)
    if evaluate(board) == 1:
        print("You lose!")
    elif evaluate(board) == -1:
        print("You win!")
    else:
        print("It's a draw!")
play_game()
