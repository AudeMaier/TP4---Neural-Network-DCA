import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import torch


def prediction(input, model, model_type) :
    if model_type == "non-linear" : return model.non_linear(input)
    elif model_type == "mix" : return model.linear(input) + model.non_linear(input)
    else : print("error with model type")    



def extract_couplings(model, model_type, original_shape) :
    """
    extract coupling coefficients from the model

    :param model: pytorch model from which to extract the coupling coefficients
    :type model: nn.Module
    :param model_type: type of the model, can be "linear", "non-linear" or "mix"
    :type model_type: string
    :param original_shape: original shape of the MSA (L,K)
    :type original_shape: tuple of int
    :return: coupling coeff of the model
    :rtype: numpy array
    """

    (L,K) = original_shape
    
    #for a linear model, the couplings are directly the weights learned
    if model_type == "linear" :
        return np.array(model.masked_linear.linear.weight.detach())

    elif model_type == "non-linear" or model_type == "mix":
        model.eval()
        with torch.no_grad() :
            #bias
            zeros = torch.zeros((L*K,L*K))
            pred_zero = prediction(zeros, model, model_type)
            pred_zero = pred_zero.detach()

            #prediction of delta_{i'=i} = bias + couplings[i]
            input_i = torch.eye(L*K)
            pred_i = prediction(input_i, model, model_type)
            pred_i = pred_i.detach()
            
            return np.array(pred_i - pred_zero)

    else : print("error with model type")




def ising_gauge(couplings, model, model_type, original_shape) :
    """
    apply Ising gauge on the coupling coefficients

    if model_type is "linear", only the second order Ising gauge is applies
    if model_type is "non-linear" or "mix", both second and third order Ising gauge are applied

    :param couplings: couplings coefficients on which ising gauge is needed
    :type couplings: numpy array
    :param model: pytorch model from which the couplings have been extracted
    :type model: nn.Module
    :param original_shape: original shape of the MSA (L,K)
    :type original_shape: tuple of int
    :return: coupling coefficients after Ising gauge
    :rtype: numpy array
    """

    (L,K) = original_shape
    
    #third order Ising gauge
    if model_type == "non-linear" or model_type == "mix" :
        model.eval()
        with torch.no_grad() :
            #bias
            zeros = torch.zeros((L*K,L*K))
            pred_zero = prediction(zeros, model, model_type)
            pred_zero = pred_zero.detach()

            #extract third order interaction coefficients from the model and use them to apply third order ising gauge
            for l in range(L) :
                print("gauge process on triplets : ", l, "/", L)

                nb_rows = (L-1) * K
                input_j = torch.cat((torch.eye(nb_rows)[:,:l*K], torch.zeros(nb_rows, K), torch.eye(nb_rows)[:,l*K:]), dim=1)
                pred_j = prediction(input_j, model, model_type)
                pred_j = pred_j.detach()

                for k in range(K) :

                    input_i_j = torch.clone(input_j)
                    input_i_j[:,l*K+k] = 1
                    pred_i_j = prediction(input_i_j, model, model_type)
                    pred_i_j = pred_i_j.detach()

                    input_i = torch.zeros(nb_rows, L*K)
                    input_i[:,l*K+k] = 1
                    pred_i = prediction(input_i, model, model_type)
                    pred_i = pred_i.detach()

                    #third order interaction coefficients between residue k and all other residues for all categories
                    triplets_l = np.array(pred_i_j - pred_i - pred_j + pred_zero[:len(pred_i_j)])

                    #third order ising gauge
                    couplings[l*K+k,:] += 2 * (np.sum(triplets_l,axis=0) / K)

    #second order Ising gauge
    for i in range(L) :
        couplings[i*K:(i+1)*K,:] -= np.sum(couplings[i*K:(i+1)*K,:], axis=0) / K

    return couplings




@jit(nopython=True, parallel=True) #parallelise using numba
def average_product_correction(f) :
    """
    apply the average product correction on f, a numpy array containing the couplings,
    and return the corrected f

    :param f: array on which we want to apply the average product correction
    :type f: numpy array
    :return: f after average product correction
    :rtype: numpy array
    """
    shape = f.shape
    f_i_s = np.sum(f, 1)/(shape[1]-1)
    f_j_s = np.sum(f, 0)/(shape[0]-1)
    f_ = np.sum(f)/(shape[0]*(shape[1]-1))
    for i in range(shape[0]) :
        for j in range(shape[1]) :
            if j!= i : f[i,j] -= f_i_s[i]*f_j_s[j]/f_
    return f




def couplings(model_name, model_type, L, K, output_name) :
    model = torch.load(model_name)
    L = int(L)
    K = int(K)

    #weight extraction and Ising gauge
    couplings = extract_couplings(model, model_type, (L,K))

    #plot symmetry of coupling coeff before Ising gauge
    plt.plot(np.triu(couplings).flatten(), np.tril(couplings).T.flatten(), '.')
    plt.plot(np.linspace(-2, 3), np.linspace(-2, 3), '--')
    plt.xlabel("$C_{\lambda \kappa lk}$", fontsize=18)
    plt.ylabel("$C_{lk \lambda \kappa}$", fontsize=18)
    plt.grid()
    plt.show()

    #Ising gauge
    couplings = ising_gauge(couplings, model, model_type, (L,K))
    

    #plot symmetry of coupling coeff after Ising gauge
    plt.plot(np.triu(couplings).flatten(), np.tril(couplings).T.flatten(), '.')
    plt.plot(np.linspace(-2, 3), np.linspace(-2, 3), '--')
    plt.xlabel("$C_{\lambda \kappa lk}$", fontsize=18)
    plt.ylabel("$C_{lk \lambda \kappa}$", fontsize=18)
    plt.grid()
    plt.show()
    
    #average non-symetrical couplings
    couplings = 0.5*(couplings + couplings.T)

    #reshape couplings in a L x L array where each element contains the K x K categorical couplings to apply frobenius norm on each element
    matrix = []
    for i in range(L) :
        rows = []
        for j in range(L) :
            rows.append(couplings[i*K:(i+1)*K, j*K:(j+1)*K])
        matrix.append(rows)
    couplings = np.array(matrix)

    #frobenius norm
    couplings = np.linalg.norm(couplings, 'fro', (2, 3))

    #average product correction
    couplings = average_product_correction(couplings)

    #reshape in form (0,1) (0,2) ... (1,2) (1,3) ...
    couplings = np.triu(couplings)
    tmp = []
    for i in range(L) :
        for j in range(i+1, L) :
            tmp.append(couplings[i,j])
    couplings = np.array(tmp)


    np.savetxt(output_name, couplings)
