import argparse
import preprocessing

if __name__=="__main__":
    
    parser=argparse.ArgumentParser()
    parser.add_argument("input_name",help="The input MSA in fasta format.")
    parser.add_argument("output_name", help="Name for the output file.")
    args=parser.parse_args()
    
    preprocessing.preprocessing(args.input_name, args.output_name)