import argparse
import couplings

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    #parser.add_argument("triplets_name",help="File containing the triplets of the NN obtained using model.py.")
    parser.add_argument("model_name", help="Name of the file where the trained model was saved")
    parser.add_argument("model_type", help="type of the model, can be 'linear', 'non-linear' or 'mix'")
    parser.add_argument("L", help="length of the sequences in the MSA")
    parser.add_argument("K", help="number of categories in the MSA")
    parser.add_argument("output_name", help="Name for the output file that will containg the couplings.")
    args=parser.parse_args()
    
    couplings.couplings(args.model_name, args.model_type, args.L, args.K, args.output_name)