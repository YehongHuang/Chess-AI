import msgpack
import numpy as np
import random

# 定义读取 .mpk 文件的函数，返回所有数据
def read_all_data_from_mpk(file_path):
    with open(file_path, "rb") as f:
        unpacker = msgpack.Unpacker(f, raw=False)  # 使用 msgpack.Unpacker 逐步读取数据
        data_list = []
        for item in unpacker:
            data_list.append(item)
        return data_list  # 返回所有数据

# 随机选择一个数据
def get_random_board(data_list):
    return random.choice(data_list)

# 读取数据
boards_path = "preprocessed_games/100__boards__standard_2018_human_morethan10.mpk"
all_board_data = read_all_data_from_mpk(boards_path)

# 获取一个随机的棋盘数据
random_board_data = get_random_board(all_board_data)

# 可视化随机选择的棋盘数据
import matplotlib.pyplot as plt

def visualize_board(board_data):
    # 将 board_data 从 (8, 8, 12) 转换为 (12, 8, 8)
    board_data = np.transpose(board_data, (2, 0, 1))
    num_channels = board_data.shape[0]  # 获取通道数
    
    # 为每个通道设置具体的棋子标签
    piece_labels = [
        'Black Pawn', 'Black Knight', 'Black Bishop', 'Black Rook', 'Black King', 'Black Queen',
        'White Pawn', 'White Knight', 'White Bishop', 'White Rook', 'White King', 'White Queen'
    ]

    fig, axs = plt.subplots(1, num_channels, figsize=(20, 10))  # 创建子图
    for i in range(num_channels):  # 遍历通道
        axs[i].imshow(board_data[i], cmap='gray')  # 显示每个通道的矩阵
        axs[i].set_title(piece_labels[i])  # 使用具体的棋子标签
        axs[i].axis('off')  # 关闭坐标轴
    plt.show()

def board_to_chess_notation(board_data):
    # 定义棋子符号与通道的对应关系
    piece_symbols = [
        'bP', 'bN', 'bB', 'bR', 'bK', 'bQ',  # 黑棋符号
        'wP', 'wN', 'wB', 'wR', 'wK', 'wQ'   # 白棋符号
    ]
    
    # 转置 board_data 使其变为 (12, 8, 8)
    board_data = np.transpose(board_data, (2, 0, 1))
    
    # 初始化棋盘，每个位置都为空
    chess_board = [['--' for _ in range(8)] for _ in range(8)]
    
    # 遍历每个通道，并填充棋盘
    for i in range(12):
        for row in range(8):
            for col in range(8):
                if board_data[i, row, col] == 1:  # 如果该位置有棋子
                    chess_board[row][col] = piece_symbols[i]  # 使用对应的棋子符号填充
    
    return chess_board

def chess_notation_to_board(chess_notation):
    # 定义棋子符号与通道的对应关系
    piece_symbols = [
        'bP', 'bN', 'bB', 'bR', 'bK', 'bQ',  # 黑棋符号
        'wP', 'wN', 'wB', 'wR', 'wK', 'wQ'   # 白棋符号
    ]
    
    # 创建一个全零的 (8, 8, 12) 数组
    board_data = np.zeros((8, 8, 12), dtype=int)
    
    # 遍历棋盘，将每个位置上的棋子映射到通道
    for row in range(8):
        for col in range(8):
            piece = chess_notation[row][col]  # 获取该位置的棋子
            if piece != '--':  # 如果不是空格
                channel = piece_symbols.index(piece)  # 找到对应的通道
                board_data[row, col, channel] = 1  # 设置通道中的该位置为 1
    
    # 将 (8, 8, 12) 转换为 (8, 8, 12) 格式不变
    return board_data

new_board = board_to_chess_notation(random_board_data)
#board (8,8,12) to (8,8)
print(np.array(new_board))
np_array=np.array(new_board)
print(np_array.shape)
visualize_board(np.array(random_board_data))

def chess_notation_to_board(chess_notation):
    # 定义棋子符号与通道的对应关系
    piece_symbols = [
        'bP', 'bN', 'bB', 'bR', 'bK', 'bQ',  # 黑棋符号
        'wP', 'wN', 'wB', 'wR', 'wK', 'wQ'   # 白棋符号
    ]
    
    # 创建一个全零的 (8, 8, 12) 数组
    board_data = np.zeros((8, 8, 12), dtype=int)
    
    # 遍历棋盘，将每个位置上的棋子映射到通道
    for row in range(8):
        for col in range(8):
            piece = chess_notation[row][col]  # 获取该位置的棋子
            if piece != '--':  # 如果不是空格
                channel = piece_symbols.index(piece)  # 找到对应的通道
                board_data[row, col, channel] = 1  # 设置通道中的该位置为 1
    
    # 将 (8, 8, 12) 转换为 (8, 8, 12) 格式不变
    return board_data

#board (8,8) to (8,8,12)
converted_board_data = chess_notation_to_board(new_board)

print(np.array(converted_board_data))
np_array=np.array(converted_board_data)
print(converted_board_data.shape)

if np.array_equal(converted_board_data,random_board_data):
    print("两个数组是一样的")
else:
    print("两个数组不一样")
#compare if not correct 