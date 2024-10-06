import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import msgpack
import numpy as np
import itertools
from models.ccgan import Generator, Discriminator, create_ccgan, train_ccgan, get_saver
from helpers.conversion import AZ_MOVE_COUNT
# File name prefix for logging / saving
run_name = "new model"

# Get networks
generator, discriminator, generator_optimizer, discriminator_optimizer = create_ccgan()
print(generator)
print(discriminator)

# Training parameters
epochs = 25
batch_size = 100
train_samples = 750
shuffle_buffer = 150
n_th_epochs = 10
save_generated = True  # Should the nth epoch's final batch's generated moves be saved?
step_per_epochs = 25

# Custom Dataset class
class ChessDataset(Dataset):
    def __init__(self, boards_path, moves_path):
        self.boards = []
        self.moves = []

        # Use msgpack.Unpacker to load the boards data incrementally
        with open(boards_path, "rb") as f:
            unpacker = msgpack.Unpacker(f, raw=False)
            for obj in unpacker:
                self.boards.append(obj)

        # Use msgpack.Unpacker to load the moves data incrementally
        with open(moves_path, "rb") as f:
            unpacker = msgpack.Unpacker(f, raw=False)
            for obj in unpacker:
                self.moves.append(obj)

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx):
        board = np.array(self.boards[idx], dtype=np.float32).reshape(12, 8, 8)  # (12, 8, 8) shape for the board
        move = np.array(self.moves[idx], dtype=np.float32)

        # Reshape move to AZ_MOVE_COUNT dimensions
        if move.size != AZ_MOVE_COUNT:
            move = np.zeros(AZ_MOVE_COUNT, dtype=np.float32)
            move[self.moves[idx]] = 1.0  # One-hot encoding the move

        return torch.tensor(board), torch.tensor(move)

# Data file paths
boards_path = "preprocessed_games/2500__boards__standard_2023_human_morethan10.mpk"
moves_path = "preprocessed_games/2500__moves__standard_2023_human_morethan10.mpk"

# Get training dataset generator
dataset = ChessDataset(boards_path, moves_path)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print ()
# Train the GAN
print("Beginning training")

history = train_ccgan(dataloader, generator, discriminator, generator_optimizer, discriminator_optimizer, epochs,step_per_epochs,
                     n_th_epochs, save_models_fn=get_saver("saved_models__%s/" % run_name) if save_generated else None)

print("Training ended")

# Write captured generated samples
if save_generated:
    generated_path = "generated__%s.msgpack" % run_name
    with open(generated_path, 'ab+') as output:
        for epoch in range(epochs):
            d_loss, d_acc, g_loss, generated = history[epoch]
            print("\nEpoch %d:\n\tDiscriminator Loss: %f (Accuracy: %.2f%%)\n\tGenerator Loss: %f" % (epoch + 1, d_loss, d_acc * 100, g_loss))
            if generated is not None:
                msgpack.pack((epoch, epochs, [np.argmax(move).item() for move in generated]), output)

# Log run metrics
with open("RunHistoryData__%s.csv" % run_name, 'a+') as history_csv:
    history_csv.write("Discriminator Loss,Discriminator Accuracy,Generator Loss\n")
    for d_loss, d_acc, g_loss, _ in history:
        history_csv.write("%s,%s,%s\n" % (d_loss, d_acc, g_loss))

