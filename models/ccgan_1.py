import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import os
import torch.nn.functional as F
from helpers.conversion import AZ_MOVE_COUNT

from helpers.io_helpers import ensure_dir_exists



class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv = nn.Conv2d(12, 2, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2 * 8 * 8 + 1, 256)
        self.fc2 = nn.Linear(256, AZ_MOVE_COUNT)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, board, noise):
        x = torch.relu(self.conv(board))
        x = self.flatten(x)
        x = torch.cat((x, noise), dim=1)
        x = torch.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Conv2d(12, 2, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2 * 8 * 8 + AZ_MOVE_COUNT, 256)
        self.fc2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, board, move):
        x = F.leaky_relu(self.conv(board), negative_slope=0.2)
        x = self.flatten(x)
        x = torch.cat((x, move), dim=1)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = self.sigmoid(self.fc2(x))
        return x


def create_ccgan():
    generator = Generator()
    discriminator = Discriminator()

    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    generator_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    return generator, discriminator, generator_optimizer, discriminator_optimizer


def train_ccgan(dataloader, generator, discriminator, generator_optimizer, discriminator_optimizer, epochs, steps_per_epoch, nth_epoch, save_models_fn=None):
    criterion = nn.BCELoss()
    history = []  # 用于存储每个 epoch 的训练历史

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for step, (boards, moves) in enumerate(dataloader):
            # 创建真实标签和假的标签
            real_labels = torch.ones(boards.size(0), 1)
            fake_labels = torch.zeros(boards.size(0), 1)

            # 生成噪声
            noise = torch.randn(boards.size(0), 1)

            # 生成假动作
            fake_moves = generator(boards, noise)

            # 判别器训练 - 真实数据
            discriminator_optimizer.zero_grad()
            real_outputs = discriminator(boards, moves)
            d_loss_real = criterion(real_outputs, real_labels)
            d_loss_real.backward()

            # 判别器训练 - 假数据
            fake_outputs = discriminator(boards, fake_moves.detach())
            d_loss_fake = criterion(fake_outputs, fake_labels)
            d_loss_fake.backward()
            discriminator_optimizer.step()

            d_loss = (d_loss_real + d_loss_fake) / 2

            # 生成器训练
            generator_optimizer.zero_grad()
            outputs = discriminator(boards, fake_moves)
            g_loss = criterion(outputs, real_labels)
            g_loss.backward()
            generator_optimizer.step()

            print(f"Step {step + 1}/{steps_per_epoch}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

            # 记录每个批次的损失
            history.append((d_loss.item(), real_outputs.mean().item(), g_loss.item(), fake_moves.detach().cpu().numpy() if nth_epoch != 0 and (epoch % nth_epoch == 0 or epoch == epochs - 1) else None))

        # 计算每个 epoch 的平均损失
        current_epoch_history = history[epoch * steps_per_epoch:]
        d_loss_avg = sum(map(lambda x: x[0], current_epoch_history)) / steps_per_epoch
        d_acc_avg = sum(map(lambda x: x[1], current_epoch_history)) / steps_per_epoch
        g_loss_avg = sum(map(lambda x: x[2], current_epoch_history)) / steps_per_epoch
        print(f"\t[mean D loss: {d_loss_avg:.6f}, mean acc.: {100 * d_acc_avg:.2f}%%] [mean G loss: {g_loss_avg:.6f}]")

        # 检查点 - 每 nth_epoch 保存模型
        if save_models_fn is not None and ((epoch + 1) % nth_epoch == 0 or epoch == epochs - 1 or epoch == 0):
            save_models_fn(generator, discriminator, epoch + 1)

    print("End of final epoch")
    return history


def get_saver(out_dir):
    ensure_dir_exists(out_dir)

    def save_models(generator, discriminator, epoch):
        torch.save(generator.state_dict(), f"{out_dir}/generator_{epoch}.pth")
        torch.save(discriminator.state_dict(), f"{out_dir}/discriminator_{epoch}.pth")

    return save_models


def load_models(generator_dir,discriminator_dir):
    generator = Generator()
    discriminator = Discriminator()
    generator.load_state_dict(torch.load(f"{generator_dir}"))
    discriminator.load_state_dict(torch.load(f"{discriminator_dir}"))
    return generator, discriminator


if __name__ == "__main__":
    board_shape = (1, 12, 8, 8)  # (batch_size, channels, height, width)
    noise_shape = (1, 1)  # (batch_size, noise_dimension)
    move_shape = (1, AZ_MOVE_COUNT)  # (batch_size, number_of_possible_moves)

    # Create random input data
    board_input = torch.randn(board_shape)  # Random board input
    noise_input = torch.randn(noise_shape)  # Random noise input
    move_input = torch.randn(move_shape)    # Random move input

    # Initialize the models
    generator = Generator()
    discriminator = Discriminator()

    # Generate a move using the generator
    generated_move = generator(board_input, noise_input)
    print("Generated Move: ")
    print(generated_move)

    # Check if the discriminator can evaluate the generated move
    validity = discriminator(board_input, generated_move)
    print("Discriminator's Validity for Generated Move: ")
    print(validity)

    # Check if the discriminator can evaluate a real move
    validity_real = discriminator(board_input, move_input)
    print("Discriminator's Validity for Real Move: ")
    print(validity_real)