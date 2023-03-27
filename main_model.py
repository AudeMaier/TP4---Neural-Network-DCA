import argparse
import model
from distutils.util import strtobool

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("MSA_name",help="File containing the preprocessed MSA obtained using preprocessing.py.")
    parser.add_argument("weights_name", help="File containing the weights of the MSA obtained using weights.py.")
    parser.add_argument("model_type", help="Type of model to be built and trained, can be 'linear', 'non-linear' or 'mix'")
    parser.add_argument("activation", help="activation function of hidden layer if model_type is 'non-linear' or 'mix', can be 'square' or 'tanh', if model_type is 'linear' the parameter will be ignored")
    parser.add_argument("output_name", help="name for the output files")
    args=parser.parse_args()
    
    model.execute(args.MSA_name, args.weights_name, args.model_type, args.activation, args.output_name)

