import msgpack
import random

# 定义读取 moves 数据的函数，返回所有 move 数据
def read_moves_data(moves_path):
    moves = []  # 用于存储所有的 move 数据
    with open(moves_path, "rb") as f:
        unpacker = msgpack.Unpacker(f, raw=False)
        for obj in unpacker:
            moves.append(obj)  # 将每个 move 加入到列表
    return moves

# 随机选择一个 move
def get_random_move(moves_list):
    return random.choice(moves_list)

# 示例：读取 moves 数据
moves_path = "preprocessed_games/100__moves__standard_2018_human_morethan10.mpk"
all_moves_data = read_moves_data(moves_path)

# 随机选择一个 move 数据
random_move = get_random_move(all_moves_data)

# 输出随机选择的 move 数据
print("Random move data:", random_move)

# 创建棋盘上的所有格子的标签
def create_square_labels():
    files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']  # 棋盘文件
    ranks = ['1', '2', '3', '4', '5', '6', '7', '8']  # 棋盘行
    squares = [file + rank for rank in ranks for file in files]  # 所有64个格子的标签
    return squares

# 创建棋步映射表
def create_move_mappings():
    squares = create_square_labels()
    all_moves = []
    for start_square in squares:
        for end_square in squares:
            if start_square != end_square:  # 不能移动到同一个位置
                all_moves.append(start_square + end_square)  # 如 "e2e4"
    return all_moves

# 获取具体的棋步表示
def get_move_from_index(move_index):
    moves = create_move_mappings()
    if 0 <= move_index < len(moves):
        return moves[move_index]
    else:
        raise ValueError(f"Invalid move index: {move_index}")

# 示例：假设随机选择的 move 索引是 4007
random_move_index = 4007
predicted_move = get_move_from_index(random_move_index)
print(f"Random Move: {predicted_move}")
