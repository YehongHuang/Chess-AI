import chess
import chess.pgn
import torch
import numpy as np
from datetime import date
from helpers.conversion import AZ_MOVE_COUNT, board_to_matrix, az_index_to_move, move_to_az_index
from helpers.iteration import iter_until_except
from models.ccgan import load_models  # 假设你已经把模型转换成 PyTorch 并保存好

# 生成棋局的 PGN 头信息
def generate_headers_for(game):
    return {
        "Event": "?",
        "Site": "?",
        "Date": date.today().strftime("%Y.%m.%d"),
        "Round": "-",
        "White": "ChessMoveGAN",
        "Black": "ChessMoveGAN",
        "Result": game.end().board().result(),
        "WhiteIsComp": "Yes",
        "BlackIsComp": "Yes",
        "WhiteType": "program",
        "BlackType": "program",
        "PlyCount": "%s" % (game.end().board().fullmove_number + 1 if game.end().board().turn == False else 0)
    }

# 创建一个棋局
def generate_game(board):
    game = chess.pgn.Game.from_board(board)
    for header, value in generate_headers_for(game).items():
        game.headers[header] = value
    return game

# 生成棋盘状态
def generate_boards(generator):
    board = chess.Board()

    while not board.is_game_over(claim_draw=True):
        # 将棋盘矩阵的维度从 [8, 8, 12] 调整为 [12, 8, 8]
        in_board = torch.tensor([board_to_matrix(board)], dtype=torch.float32)
        in_board = in_board.permute(0, 3, 1, 2)  # 调整维度： [batch_size, 8, 8, 12] -> [batch_size, 12, 8, 8]
        
        # 打印输入棋盘的形状
        print(f"Input board shape: {in_board.shape}")
        
        in_noise = torch.randn(1, 1)  # 生成随机噪声
        print(f"Noise input shape: {in_noise.shape}")  # 打印噪声的形状

        # 获取生成器输出，并查看其形状
        probabilities = generator(in_board, in_noise)[0].detach().numpy()  # 通过生成器预测，并转换为 NumPy 格式
        
        # 打印生成器的输出形状和前10个值
        print(f"Generator output shape: {probabilities.shape}")
        print(f"First 10 generator output probabilities: {probabilities[:10]}")
        print
        board.push(az_index_to_move(highest_probability_valid_move(board, probabilities)))


    return board

# 选择概率最高的合法棋步
def highest_probability_valid_move(board, probabilities):
    legal_az_move_indices = [move_to_az_index(move) for move in board.legal_moves]
    invalid_indices = [i for i in range(AZ_MOVE_COUNT) if i not in legal_az_move_indices]
    for invalid in invalid_indices:
        probabilities[invalid] = 0    
    return np.argmax(probabilities)

# 主函数
def main():
    games_to_generate = 3
    pgn_out_path = "test.pgn"
    
    # 加载 PyTorch 模型
    generate_dir = "saved_models__new model\generator_25.pth"
    discriminator_dir = "saved_models__new model\discriminator_25.pth"

    generator, _, = load_models(generate_dir,discriminator_dir)  # 假设 load_models 函数已修改为加载 PyTorch 模型

    # 生成游戏并写入 PGN 文件
    with open(pgn_out_path, 'w+') as pgn_out_file:
        for game_count in range(games_to_generate):
            print("Generating game %s of %s" % (game_count + 1, games_to_generate))
            game = generate_game(generate_boards(generator))
            print(game, file=pgn_out_file, end="\n\n")

if __name__ == "__main__":
    main()

