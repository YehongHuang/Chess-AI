import sys
import pygame as p
import chess
import torch
from helpers.conversion import board_to_matrix, az_index_to_move, move_to_az_index
from models.ccgan import load_models

# Initialize constants
BOARD_WIDTH = BOARD_HEIGHT = 512
DIMENSION = 8
SQ_SIZE = BOARD_HEIGHT // DIMENSION
MAX_FPS = 15
IMAGES = {}

# AI vs Human settings
SET_WHITE_AS_BOT = False  # Set True if AI plays as white
SET_BLACK_AS_BOT = False  # Set True if AI plays as black

# Colors for the chessboard
LIGHT_SQUARE_COLOR = (240, 217, 181)
DARK_SQUARE_COLOR = (181, 136, 99)

# Load chess piece images
def loadImages():
    pieces = ['bR', 'bN', 'bB', 'bQ', 'bK', 'bP','wR', 'wN', 'wB', 'wQ', 'wK', 'wP']
    for piece in pieces:
        image_path = "images1/" + piece + ".png"
        original_image = p.image.load(image_path)
        IMAGES[piece] = p.transform.smoothscale(original_image, (SQ_SIZE, SQ_SIZE))

# Simplified board drawing function
def drawBoard(screen, board):
    colors = [p.Color(LIGHT_SQUARE_COLOR), p.Color(DARK_SQUARE_COLOR)]
    for row in range(DIMENSION):
        for col in range(DIMENSION):
            color = colors[(row + col) % 2]
            # Draw squares with reversed row order
            p.draw.rect(screen, color, p.Rect(col * SQ_SIZE, (DIMENSION - row - 1) * SQ_SIZE, SQ_SIZE, SQ_SIZE))

            piece = board.piece_at(chess.square(col, row))
            if piece is not None:
                piece_symbol = piece.symbol()
                piece_image_key = ('w' if piece.color == chess.WHITE else 'b') + piece_symbol.upper()
                # Draw pieces with reversed row order
                screen.blit(IMAGES[piece_image_key], p.Rect(col * SQ_SIZE, (DIMENSION - row - 1) * SQ_SIZE, SQ_SIZE, SQ_SIZE))

# Function to handle pawn promotion
def handle_pawn_promotion(gs, move, screen):
    piece = gs.piece_at(move.from_square)
    if piece and piece.piece_type == chess.PAWN:
        if chess.square_rank(move.to_square) in [0, 7]:
            print(f"Pawn at {move.to_square} is being promoted")
            promotion_piece = show_promotion_popup(screen, gs)
            return chess.Move(move.from_square, move.to_square, promotion=promotion_piece)
    return move

# Display a promotion popup for selecting the piece type
def show_promotion_popup(screen, gs):
    print("Showing promotion popup...")
    # Create a semi-transparent overlay
    overlay = p.Surface((BOARD_WIDTH, BOARD_HEIGHT), p.SRCALPHA)
    overlay.fill((0, 0, 0, 128))  # Semi-transparent black overlay
    screen.blit(overlay, (0, 0))

    # Determine color and position
    color = 'w' if gs.turn == chess.WHITE else 'b'

    options = ['Q', 'R', 'B', 'N']
    promotion_pieces = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
    option_rects = []

    # Draw options
    option_size = SQ_SIZE
    total_width = option_size * len(options)
    start_x = (BOARD_WIDTH - total_width) // 2
    y = (BOARD_HEIGHT - option_size) // 2

    for i, piece_symbol in enumerate(options):
        x = start_x + i * option_size
        piece_image_key = color + piece_symbol
        screen.blit(IMAGES[piece_image_key], p.Rect(x, y, option_size, option_size))
        option_rects.append(p.Rect(x, y, option_size, option_size))

    p.display.flip()

    # Wait for user to click
    while True:
        for e in p.event.get():
            if e.type == p.MOUSEBUTTONDOWN:
                pos = p.mouse.get_pos()
                for i, rect in enumerate(option_rects):
                    if rect.collidepoint(pos):
                        print(f"Player selected promotion to {options[i]}")
                        return promotion_pieces[i]
            if e.type == p.QUIT:
                p.quit()
                sys.exit()

# Function to handle human player's input
def get_human_move(gs, validMoves, playerClicks, screen):
    moveMade = False
    squareSelected = ()

    if len(playerClicks) == 1:  # First click: Selecting the piece
        row, col = playerClicks[0]
        # Adjust row to match board orientation
        row = DIMENSION - row - 1
        piece = gs.piece_at(chess.square(col, row))

        if piece is None or (piece.color == chess.WHITE and not gs.turn) or (piece.color == chess.BLACK and gs.turn):
            print("Invalid piece selection. Please select a valid piece.")
            playerClicks = []
        else:
            squareSelected = (row, col)
            print(f"Selected piece at {(row, col)}.")

    elif len(playerClicks) == 2:  # Second click: Selecting the destination
        if playerClicks[0] == playerClicks[1]:
            print("Clicked the same square twice. Please select a different destination.")
            playerClicks = []
        else:
            start_row, start_col = playerClicks[0]
            end_row, end_col = playerClicks[1]
            start_row = DIMENSION - start_row - 1
            end_row = DIMENSION - end_row - 1
            uci_move = convert_clicks_to_uci([(start_row, start_col), (end_row, end_col)])

            print(f"Attempting move from {(start_row, start_col)} to {(end_row, end_col)}")
            if uci_move:
                move = chess.Move.from_uci(uci_move)
                # Check for pawn promotion
                if gs.piece_at(move.from_square) and gs.piece_at(move.from_square).piece_type == chess.PAWN:
                    if chess.square_rank(move.to_square) in [0, 7]:
                        print(f"Promotion required for move {uci_move}.")
                        move = handle_pawn_promotion(gs, move, screen)
                if move in validMoves:
                    gs.push(move)
                    moveMade = True
                    playerClicks = []
                else:
                    print("Invalid move. Please select a valid move.")
                    squareSelected = ()
                    playerClicks = []

    # Handle changing selection
    if len(playerClicks) == 1 and not moveMade:
        row, col = playerClicks[0]
        row = DIMENSION - row - 1
        piece = gs.piece_at(chess.square(col, row))

        if piece and ((piece.color == chess.WHITE and gs.turn) or (piece.color == chess.BLACK and not gs.turn)):
            print(f"Changing selection to new piece at {(row, col)}.")
            squareSelected = (row, col)
        else:
            print("Invalid piece selection. Please select a valid piece.")
            playerClicks = []

    return moveMade, squareSelected, playerClicks

# Convert clicks to UCI move format
def convert_clicks_to_uci(playerClicks):
    if len(playerClicks) != 2:
        return None

    start_pos = playerClicks[0]
    end_pos = playerClicks[1]

    start_square = chess.square_name(chess.square(start_pos[1], start_pos[0]))
    end_square = chess.square_name(chess.square(end_pos[1], end_pos[0]))

    return start_square + end_square

# Function to let AI make a move using the GAN model
def get_ai_move(generator, board):
    in_board = torch.tensor([board_to_matrix(board)], dtype=torch.float32)
    in_board = in_board.permute(0, 3, 1, 2)
    in_noise = torch.randn(1, 1)
    probabilities = generator(in_board, in_noise)[0].detach().numpy()

    ai_move_index = highest_probability_valid_move(board, probabilities)
    ai_move = az_index_to_move(ai_move_index, board)  # 传递 board 对象

    # 调试信息
    print("AI move index:", ai_move_index)
    print("AI generated move:", ai_move)
    print("Legal moves:", list(board.legal_moves))

    if ai_move not in board.legal_moves:
        raise ValueError("AI generated an illegal move.")
    return ai_move


# Function to select the highest probability valid move
def highest_probability_valid_move(board, probabilities):
    legal_az_move_indices = [move_to_az_index(move) for move in board.legal_moves]
    invalid_indices = [i for i in range(len(probabilities)) if i not in legal_az_move_indices]
    probabilities[invalid_indices] = 0  # Zero out illegal move probabilities
    return probabilities.argmax()

# Main game loop
def main():
    p.init()
    screen = p.display.set_mode((BOARD_WIDTH, BOARD_HEIGHT))
    clock = p.time.Clock()
    screen.fill(p.Color(LIGHT_SQUARE_COLOR))
    loadImages()

    # Load GAN model (only needed if AI is enabled)
    generate_dir = "saved_models__new model/generator_25.pth"
    discriminator_dir = "saved_models__new model/discriminator_25.pth"
    generator, _ = load_models(generate_dir, discriminator_dir)


    # Initialize game state
    gs = chess.Board()

    validMoves = list(gs.legal_moves)

    moveMade = False
    squareSelected = ()
    playerClicks = []
    gameOver = False
    running = True
    moveHistory = []

    while running:
        humanTurn = (not gs.turn and not SET_BLACK_AS_BOT) or (gs.turn and not SET_WHITE_AS_BOT)
        
        # Event handling
        for e in p.event.get():
            if e.type == p.QUIT:
                running = False

            # Mouse input handling (human player)
            elif e.type == p.MOUSEBUTTONDOWN and humanTurn and not gameOver:
                location = p.mouse.get_pos()
                col = location[0] // SQ_SIZE
                row = location[1] // SQ_SIZE
                playerClicks.append((row, col))  # Record clicks

                # Handle human move
                moveMade, squareSelected, playerClicks = get_human_move(gs, validMoves, playerClicks, screen)
                if moveMade:
                    moveHistory.append(gs.copy())

            # Reset game
            elif e.type == p.KEYDOWN:
                if e.key == p.K_r:  # Reset the board
                    gs.reset()
                    validMoves = list(gs.legal_moves)
                    squareSelected = ()
                    playerClicks = []
                    moveMade = False
                    gameOver = False
                    moveHistory = []
                elif e.key == p.K_u:  # Undo the last move
                    if len(moveHistory) > 1:
                        moveHistory.pop()
                        gs = moveHistory.pop()
                        moveMade = True

        # AI move handling
        if not humanTurn and not gameOver:
            if (gs.turn and SET_WHITE_AS_BOT) or (not gs.turn and SET_BLACK_AS_BOT):
                move = get_ai_move(generator, gs)
                gs.push(move)
                moveMade = True
                moveHistory.append(gs.copy())

        # Update board state if move was made
        if moveMade:
            validMoves = list(gs.legal_moves)
            moveMade = False

            # Check for game over conditions
            if gs.is_checkmate():
                gameOver = True
                draw_text(screen, "Checkmate! Black wins" if gs.turn else "Checkmate! White wins")
                print("Checkmate! Black wins" if gs.turn else "Checkmate! White wins")
            elif gs.is_stalemate():
                gameOver = True
                draw_text(screen, "Stalemate!")
                print("Stalemate!")

        # Draw the board
        drawBoard(screen, gs)
        clock.tick(MAX_FPS)
        p.display.flip()

# Function to draw text on the screen
def draw_text(screen, text):
    font = p.font.SysFont("Arial", 32, True, False)
    text_object = font.render(text, 0, p.Color('Red'))
    text_location = p.Rect(0, 0, BOARD_WIDTH, BOARD_HEIGHT).move(
        BOARD_WIDTH / 2 - text_object.get_width() / 2, BOARD_HEIGHT / 2 - text_object.get_height() / 2)
    screen.blit(text_object, text_location)
    p.display.flip()
    # Keep displaying the text until a key is pressed
    while True:
        for e in p.event.get():
            if e.type == p.KEYDOWN or e.type == p.QUIT:
                return

if __name__ == "__main__":
    main()