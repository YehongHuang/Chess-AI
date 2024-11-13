import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import os
import torch.nn.functional as F
from helpers.conversion import AZ_MOVE_COUNT

from helpers.io_helpers import ensure_dir_exists


class ResBlock(nn.Module):

    def __init__(self, num_filters=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv1_bn = nn.BatchNorm2d(num_filters, )
        self.conv1_act = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv2_bn = nn.BatchNorm2d(num_filters, )
        self.conv2_act = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv1_bn(y)
        y = self.conv1_act(y)
        y = self.conv2(y)
        y = self.conv2_bn(y)
        y = x + y
        return self.conv2_act(y)
    
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=12, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.act1 = nn.ReLU()

        self.res_blocks = nn.ModuleList([ResBlock(num_filters=256) for _ in range(2)])

        self.conv2 = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=(1, 1), stride=(1, 1))
        self.bn2 = nn.BatchNorm2d(16)
        self.act2 = nn.ReLU()

        # Fully connected layer, concatenates noise before the fully connected layer
        self.fc1 = nn.Linear(16 * 8 * 8 + 1, 256)  # Concatenate noise
        self.fc2 = nn.Linear(256, AZ_MOVE_COUNT)  # Output action space
        self.softmax = nn.Softmax(dim=1)

    def forward(self, board, noise):
        # Convolution and batch normalization
        x = self.conv1(board)
        x = self.bn1(x)
        x = self.act1(x)

        # Residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)

        # Second convolution and batch normalization
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        # Flatten the output of the convolution layer
        x = x.reshape(x.size(0), -1)

        # Concatenate the flattened features with noise
        x = torch.cat((x, noise), dim=1)

        # Fully connected layer and Softmax
        x = F.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))

        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # Convolution layer to process board input
        self.conv1 = nn.Conv2d(in_channels=12, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.act1 = nn.LeakyReLU(negative_slope=0.2)

        # Residual blocks (you can refer to the generator's definition if needed)
        self.res_blocks = nn.ModuleList([ResBlock(num_filters=256) for _ in range(7)])

        # Second convolution layer to reduce the number of output channels
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=(1, 1), stride=(1, 1))
        self.bn2 = nn.BatchNorm2d(16)
        self.act2 = nn.LeakyReLU(negative_slope=0.2)

        # Fully connected layer
        self.fc1 = nn.Linear(16 * 8 * 8 + AZ_MOVE_COUNT, 256)  # Concatenate features after action
        self.fc2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, board, move):
        # First convolution and activation
        x = self.conv1(board)
        x = self.bn1(x)
        x = self.act1(x)

        # Feature extraction through residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)

        # Second convolution and activation
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        # Flatten the output of the convolution layer
        x = x.view(x.size(0), -1)

        # Concatenate the flattened features with the action
        x = torch.cat((x, move), dim=1)

        # Fully connected layer and Sigmoid output
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
    history = []  # Used to store training history of each epoch
    best_g_loss = float('inf')  # trace best loss

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        step = 0
        dataloader_iter = iter(dataloader)  # Generate an iterable object through iter
        
        while step < steps_per_epoch:
            try:
                # get data
                boards, moves = next(dataloader_iter)
            except StopIteration:
                # If the dataloader ends, reset the iterator and continue
                print('load new chess borad')
                dataloader_iter = iter(dataloader)
                boards, moves = next(dataloader_iter)

            device = next(generator.parameters()).device
            boards = boards.to(device)
            moves = moves.to(device)

            # Create real and fake labels
            real_labels = torch.ones(boards.size(0), 1).to(device)
            fake_labels = torch.zeros(boards.size(0), 1).to(device)

            # Generate noise
            noise = torch.randn(boards.size(0), 1).to(device)

            # Generate fake moves
            fake_moves = generator(boards, noise)

            # Discriminator training - real data
            discriminator_optimizer.zero_grad()
            real_outputs = discriminator(boards, moves)
            d_loss_real = criterion(real_outputs, real_labels)
            d_loss_real.backward()

            # Discriminator training - fake data
            fake_outputs = discriminator(boards, fake_moves.detach())
            d_loss_fake = criterion(fake_outputs, fake_labels)
            d_loss_fake.backward()
            discriminator_optimizer.step()

            d_loss = (d_loss_real + d_loss_fake) / 2

            # Generator training
            generator_optimizer.zero_grad()
            outputs = discriminator(boards, fake_moves)
            g_loss = criterion(outputs, real_labels)
            g_loss.backward()
            generator_optimizer.step()

            print(f"Step {step + 1}/{steps_per_epoch}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

            # Record loss for each batch
            history.append((d_loss.item(), real_outputs.mean().item(), g_loss.item(), fake_moves.detach().cpu().numpy() if nth_epoch != 0 and (epoch % nth_epoch == 0 or epoch == epochs - 1) else None))

            step += 1  

        current_epoch_history = history[epoch * steps_per_epoch:]
        d_loss_avg = sum(map(lambda x: x[0], current_epoch_history)) / steps_per_epoch
        d_acc_avg = sum(map(lambda x: x[1], current_epoch_history)) / steps_per_epoch
        g_loss_avg = sum(map(lambda x: x[2], current_epoch_history)) / steps_per_epoch
        print(f"\t[mean D loss: {d_loss_avg:.6f}, mean acc.: {100 * d_acc_avg:.2f}%%] [mean G loss: {g_loss_avg:.6f}]")

        # save the best model
        if g_loss_avg < best_g_loss:
            best_g_loss = g_loss_avg
            print(f"Saving best model at epoch {epoch + 1} with G loss: {best_g_loss:.6f}")
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