import argparse
import weights

if __name__=="__main__":
    
    parser=argparse.ArgumentParser()
    parser.add_argument("input_name",help="The input MSA preprocessed using the file execute_preprocessing.")
    parser.add_argument("output_name", help="Name for the output file.")
    args=parser.parse_args()
    
    weights.weights(args.input_name, args.output_name)