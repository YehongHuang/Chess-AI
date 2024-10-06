import msgpack
from random import sample

# Number of games after filtering by preprocessing
games_count = 557049
# In/out paths
all_boards_path = "preprocessed_games/boards__standard_2023_human_morethan10.mpk"
all_moves_path = "preprocessed_games/moves__standard_2023_human_morethan10.mpk"
boards_output_path_like = "preprocessed_games/%s__boards__standard_2023_human_morethan10.mpk"
moves_output_path_like = "preprocessed_games/%s__moves__standard_2023_human_morethan10.mpk"

# Numbers of games to sample, with one pair of files output for each size
sample_sizes = [10, 2500]

def main():
    # Select games by index for each sample, using sets for efficient removal
    indices_by_sample = {sample_size: set(sample(range(games_count), sample_size)) for sample_size in sample_sizes}
    print("Writing %s samples:" % len(indices_by_sample), *sample_sizes, sep="\n  ")

    from contextlib import ExitStack
    with open(all_boards_path, 'rb') as game_input, open(all_moves_path, 'rb') as label_input, ExitStack() as stack:
        game_unpacker = msgpack.Unpacker(game_input)
        label_unpacker = msgpack.Unpacker(label_input)

        # A pair of output files for each sample
        outfiles = {sample_size: {
                    "moves_file": stack.enter_context(open(moves_output_path_like % sample_size, 'ab+')),
                    "boards_file": stack.enter_context(open(boards_output_path_like % sample_size, 'ab+')),
                } for sample_size in sample_sizes }

        print("Have %s out file pairs" % len(outfiles))

        # For counting total moves in each sample
        counts = {sample_size: 0 for sample_size in sample_sizes}

        # Iterate all games, appending that game to all sample sizes it has been selected for
        for index in range(games_count):
            these_boards = None
            these_moves = None
            try:
                for sample_size in sample_sizes:
                    if index in indices_by_sample[sample_size]:
                        # Get boards & moves in current game (only once per game)
                        these_moves = these_moves or label_unpacker.unpack()
                        these_boards = these_boards or game_unpacker.unpack()

                        moves_in_game = len(these_boards)
                        print(f"Packing game {index} for sample {sample_size}. Contains {moves_in_game} moves.")

                        # Append all boards & moves in the game to appropriate out files for the sample
                        for i in range(len(these_boards)):
                            msgpack.pack(these_boards[i], outfiles[sample_size]['boards_file'])
                            msgpack.pack(these_moves[i], outfiles[sample_size]['moves_file'])

                        counts[sample_size] += moves_in_game

                        # Ensure each game is included only once (use set discard for efficiency)
                        indices_by_sample[sample_size].discard(index)

            except msgpack.exceptions.OutOfData:
                print(f"Ran out of data at game {index}. Ending early.")
                break

            # Skip game if it appears in no samples
            if these_boards is None:
                try:
                    label_unpacker.skip()
                    game_unpacker.skip()
                except msgpack.exceptions.OutOfData:
                    print("No more data available to skip. Ending.")
                    break

            # Count number of samples with no remaining games to sample to allow early exit
            empties = sum(1 for sample_size in sample_sizes if len(indices_by_sample[sample_size]) == 0)
            if empties == len(sample_sizes):
                break

    for sample_size in sample_sizes:
        print(f"Split {sample_size} contains {counts[sample_size]} moves")

if __name__ == "__main__":
    main()

