import argparse
import sys

def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataframe', type=str, help='csv dataframe', required=True)
    parser.add_argument('--opticalflow', type=str, help='csv opticalflow dataframe', required=False)
    parser.add_argument('--architecture', choices=['resnet50'], help='cnn network architecture', required=True)
    parser.add_argument('--features', type=int, help='number of features', required=True, default=1024)
    parser.add_argument('--outputdirectory', type=str, help='output directory to store the features', required=True)

# flow_df.head()
def main(argv):
    args = parse_args(argv)

    df = args.dataframe
    df_flow = args.opticalflow
    architecture = args.architecture
    num_features = args.features
    output_directory = args.outputdirectory


if __name__ == '__main__':
    main(sys.argv[1:])