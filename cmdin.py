import argparse

args = None

def parse_arguments():

    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument('-B', '--batch_size', default=100, type=int, help="Batch size")
    parser.add_argument('-E', '--epochs', default=300, type=int, help="Number of Epochs")
    parser.add_argument('--lr', '--learning_rate', default=0.001, type=float, help="Learning rate")
    parser.add_argument('--sess', '--session_name', default="MNIST", type=str, help="Session name")
    parser.add_argument('--mode', '--mode', default="one", type=str, help="One or two layers")
    parser.add_argument('-m', '--m', default=4, type=int, help="Loss parameter")
    parser.add_argument('-n', '--n', default=1, type=int, help="Activation function parameter")

    args = parser.parse_args()
    # args, unknown = parser.parse_known_args()
    return args

# Parse arguments
def run_args():
    global args
    if args is None:
        args = parse_arguments()

run_args()
