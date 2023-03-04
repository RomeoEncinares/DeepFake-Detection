import argparse
import sys

def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=str, help='features', required=True)
    parser.add_argument('--labels', type=str, help='labels', required=True)
    parser.add_argument('--outputdirectory', type=str, help='output directory to store the models', required=True)

    return parser.parse_args(argv)
    
def main(argv):
    args = parse_args(argv)

    features = args.features
    labels = args.labels
    output_directory = args.outputdirectory

if __name__ == '__main__':
    main(sys.argv[1:])