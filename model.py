import numpy as np
import torch
from torch import nn
from itertools import chain
import matplotlib.pyplot as plt



class SquareActivation(nn.Module) :
  'custom square activation'
  def __init__(self) :
    super().__init__()
  def forward(self, input) :
    return input.square()



class MaskedLinear(nn.Module):
  """
  linear model of one non fully connected layer

  build a non fully connected layer by putting a mask on the weights that must be null
  """
  def __init__(self, in_dim, out_dim, indices_mask):
    """
    :param in_features: number of input features
    :type in_features: int
    :param out_features: number of output features
    :type out_features: int
    :param indices_mask: list of couples of input and output that must be disconnected
    :type indices_maks: list of tuples of int
    """
    super(MaskedLinear, self).__init__()
        
    self.linear = nn.Linear(in_dim, out_dim) #MaskedLinear is made of a linear layer
    #Force the weights indicated by indices_mask to be zero by use of a mask
    self.mask = torch.zeros([out_dim, in_dim]).bool()
    for a, b in indices_mask : self.mask[(a, b)] = 1
    self.linear.weight.data[self.mask] = 0 # zero out bad weights

    #modify backward_hood to prevent changes to the masked weights
    def backward_hook(grad):
      # Clone due to not being allowed to modify in-place gradients
      out = grad.clone()
      out[self.mask] = 0
      return out
    
    self.linear.weight.register_hook(backward_hook)
 
  def forward(self, input):
    #CUDA
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    self.linear.weight = self.linear.weight.to(device)
    self.linear.bias = self.linear.bias.to(device)
    input = input.to(device)

    return self.linear(input)




class LinearNetwork(nn.Module):
  'linear model with softmax activation on the output layer applied on residue positions'

  def __init__(self, indices_mask, in_dim, original_shape):
    """
    :param indices_mask: list of input and output that must be disconnected
    :type indices_mask: list of tuples of int
    :param in_dim: dimension of input layer
    :type in_dim: int
    :param original_shape: original shape of the MSA (N,L,K)
    :type original_shape: tuple of int
    """
    super(LinearNetwork, self).__init__()

    self.masked_linear = MaskedLinear(in_dim, in_dim, indices_mask) #build masked linear layer
    self.softmax = nn.Softmax(dim=2)

    (N,L,K) = original_shape
    self.L = L
    self.K = K

  def forward(self, x):
    x = self.masked_linear(x)
    #apply softmax on residues
    x = torch.reshape(x, (len(x), self.L, self.K))
    x = self.softmax(x)
    x = torch.reshape(x, (len(x), self.L*self.K))

    return x




class NonLinearNetwork(nn.Module):
  'model with hidden layer and square / tanh activation on the hidden layer'
  def __init__(self, indices_mask1, indices_mask2, in_dim, hidden_dim, original_shape, activation="square"):
    """
    :param indices_mask1: list of input and hidden neurons that must be disconnected
    :type indices_mask1: list of tuples of int
    :param indices_mask2: list of hidden and output neurons that must be disconnected
    :type indices_mask2: list of tuples of int
    :param in_dim: dimension of input layer
    :type in_dim: int
    :param hidden_dim: dimension of hidden layer
    :type hidden_dim: int
    :param original_shape: original shape of the MSA (N,L,K)
    :type original_shape: tuple of int
    :param activation: activation for the hidden layer, must be "square" or "tanh" otherwise square is taken by default
    :type activation: string
    """
    super(NonLinearNetwork, self).__init__()

    #define activation function
    if activation == "square" : activation_function = SquareActivation()
    elif activation == "tanh" : activation_function = nn.Tanh()
    else :
      print("invalid activation function, square taken instead")
      activation_function = SquareActivation()

    #elements of the network
    self.non_linear = nn.Sequential(MaskedLinear(in_dim, hidden_dim, indices_mask1), activation_function, MaskedLinear(hidden_dim, in_dim, indices_mask2))
    self.softmax = nn.Softmax(dim=2)
    
    (N,L,K) = original_shape
    self.L = L
    self.K = K

  def forward(self, x):
    x = self.non_linear(x)
    #apply softmax on residues
    x = torch.reshape(x, (len(x), self.L, self.K))
    x = self.softmax(x)
    x = torch.reshape(x, (len(x), self.L*self.K))
    return x
  



class MixNetwork(nn.Module) :
  'network mixing linear model and model with hidden layer and square/tanh activation'
  def __init__(self, indices_mask1, indices_mask2, indices_mask_linear, in_dim, hidden_dim, original_shape, activation="square"):
    """
    :param indices_mask1: list of input and hidden neurons that must be disconnected for the non-linear model
    :type indices_mask1: list of tuples of int
    :param indices_mask2: list of hidden and output neurons that must be disconnected for the non-linear model
    :type indices_mask2: list of tuples of int
    :param indices_mask_linear: list of input and output neurons that must be disconnected for the linear model
    :type indices_mask_linear: list of tuples of int
    :param in_dim: dimension of input layer
    :type in_dim: int
    :param hidden_dim: dimension of hidden layer
    :type hidden_dim: int
    :param original_shape: original shape of the MSA (N,L,K)
    :type original_shape: tuple of int
    :param activation: activation for the hidden layer, must be "square" or "tanh" otherwise square is taken by default
    :type activation: string
    """
    super(MixNetwork, self).__init__()

    #define activation function
    if activation == "square" : activation_function = SquareActivation()
    elif activation == "tanh" : activation_function = nn.Tanh()
    else :
      print("invalid activation function, square taken instead")
      activation_function = SquareActivation()

    #elements of the network
    self.linear = MaskedLinear(in_dim, in_dim, indices_mask_linear)
    self.non_linear = nn.Sequential(MaskedLinear(in_dim, hidden_dim, indices_mask1), activation_function, MaskedLinear(hidden_dim, in_dim, indices_mask2))
    self.softmax = nn.Softmax(dim=2)
    
    (N,L,K) = original_shape
    self.L = L
    self.K = K

  def forward(self, x):
    #combine linear and non-linear models
    x = self.linear(x) + self.non_linear(x)
    #apply softmax on residues
    x = torch.reshape(x, (len(x), self.L, self.K))
    x = self.softmax(x)
    x = torch.reshape(x, (len(x), self.L*self.K))
    
    return x




class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, indices, data, labels):
    """
    :param indices: indices of the data that must be taken to create the dataset
    :type indices: list of int
    :param data: data from which to take the elements to create the dataset
    :type data: numpy array
    :param labels: labels correspondings to the data
    :type labels: numpy array
    """
    self.indices = indices
    self.data = torch.Tensor(data[indices])
    self.labels = torch.Tensor(labels[indices])

  def __len__(self):
    #Denotes the total number of samples
    return len(self.indices)

  def __getitem__(self, index):
    #Generates one sample of data
    return self.data[index], self.labels[index]




def loss_function(output, labels) :
  """
  loss_function that will be used to train the model

  it corresponds cross entropy loss with one hot encoded inputs and labels
  """
  loss = -torch.dot(labels, torch.log(output))
  return loss




def train(train_points, train_labels, model, loss_function, optimizer):
  'train function of the neural networks'

  model.train()
  # Compute prediction error
  pred = model(train_points)
  loss = loss_function(torch.flatten(pred), torch.flatten(train_labels))
  # Backpropagation
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()



def error(data, labels, model, original_shape) :
  """
  classification error achieved by the model on the given data

  :param data: input data (e.g. data of the training or validation set)
  :type data: torch tensor
  :param labels: labels corresponding to points
  :type labels: torch tensor
  :param model: model to be evaluated
  :type model: nn.Module
  :param original_shape: original shape of the MSA (N,L,K)
  :type original_shape: tuple
  :return: prediction error (rate) on the data
  :rtype: list of float
  """

  #CUDA
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda:0" if use_cuda else "cpu")
  labels = labels.to(device)
  data = data.to(device)

  N = len(data)
  (_,L,K) = original_shape

  model.eval()
  with torch.no_grad() :
    #prediction of points given by model, reshaped to remove one-hot encoding
    _, pred = torch.max(torch.reshape(model(data), (N, L, K)), 2)
    #labels reshaped to remove one-hot encoding
    _, labels = torch.max(torch.reshape(labels, (N, L, K)), 2)
    #total number of predicted amino acids (#sequences * length of sequences)
    total = torch.numel(pred)
    #number of correctly predicted amino acids
    correct = (pred == labels).sum().item()
    #return the fraction of uncorrect predictions
    return(1 - correct / total)




def error_positions(data, labels, model, original_shape) :
  """
  classification error by residue achieved by the model on the given data

  :param data: input data (e.g. data of the training or validation set)
  :type data: torch tensor
  :param labels: labels corresponding to points
  :type labels: torch tensor
  :param model: model to be evaluated
  :type model: nn.Module
  :param original_shape: original shape of the MSA (N,L,K)
  :type original_shape: tuple
  :return: prediction error (rate) per position
  :rtype: list of float
  """

  #CUDA
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda:0" if use_cuda else "cpu")
  labels = labels.to(device)
  data = data.to(device)

  N = len(data)
  (_,L,K) = original_shape


  model.eval()
  with torch.no_grad() :
    #prediction of points given by model, reshaped to remove one-hot encoding
    _, pred = torch.max(torch.reshape(model(data), (N, L, K)), 2)
    #labels reshaped to remove one-hot encoding
    _, labels = torch.max(torch.reshape(labels, (N, L, K)), 2)
    #error rate per position
    errors_positions = []
    for position in range(L) :
      #number of correctly predicted amino acids for this position
      correct = (pred[:, position] == labels[:, position]).sum().item()
      errors_positions.append(1 - correct / N)
    
    return errors_positions




def get_data_labels(MSA_file, weights_file, max_size = None) :
  """
  load the data and labels, returns one-hot-encoded data and labels and original shape of the MSA (N,L,K)

  :param MSA_file: name of the file containing the preprocessed MSA
  :type MSA_file: string
  :param weights_file: name of the file containing the weights of the sequences w.r.t the MSA
  :type weights_file: string
  :param max_size : if not none, the function will load only the first max_size sequences of the dataset
  :type max:size: int
  :return: one-hot-encoded data and labels, original shape of the MSA (N,L,K)
  :rtype: (numpy array, numpy array, tuple of int)
  """
  #load data and weights
  data = np.genfromtxt(MSA_file, delimiter=',').astype(int)
  weights = np.loadtxt(weights_file)
  weights = weights.reshape((len(data), 1, 1))

  if max_size is not None and max_size < len(data) :
    data = data[:max_size]
    weights = weights[:max_size]

  #put the data in one hot encoding form
  data = np.array(nn.functional.one_hot(torch.Tensor(data).to(torch.int64)))
  (N,L,K) = data.shape

  #the labels are the weighted data
  labels = weights * data

  #reshape such that each sequence has only one dimension
  data = np.reshape(data, (N, L * K))
  labels = np.reshape(labels, (N, L * K))

  print("Data and labels have been successfully obtained")
  return data, labels, (N,L,K)




def create_datasets(data, labels, separations) :
  """
  create 3 instances of class Dataset : train, validation and test set

  :param data: data from which create the Datasets
  :type data: numpy array
  :param labels: labels corresponding to data
  :type labels: numpy array
  :param separations: fractions of the dataset separating train / validation set and validation / test set (ex:(0.6,0.8))
  :type separations: tuple of float
  :return: training, validation and test dataset
  :rtype: (torch dataset, torch dataset, torch dataset)
  """
  #compute the indices of the 3 datasets
  indices = np.array(range(len(data)))
  np.random.shuffle(indices)
  train_indices, validation_indices, test_indices = np.split(indices, [int(separations[0]*len(data)), int(separations[1]*len(data))])

  #create training, validation and test dataset
  training_set = Dataset(train_indices, data, labels)
  validation_set = Dataset(validation_indices, data, labels)
  test_set = Dataset(test_indices, data, labels)

  print("the datasets have been successfully created")
  return training_set, validation_set, test_set




def build_and_train_model(data, labels, original_shape, model_type, activation=None, max_epochs=10, learning_rate=1e-3, weight_decay=1e-2, nb_hidden_neurons=32, validation = False, test = False) :
  """
  build a model according to model_type and activation, train the model and returns the model and the errors

  return a list containing the train error after each epoch, the weights and bias of the model after training
  and a list containing the train error after training for each amino acid position
  :param data, labels: data and labels obtained using get_data_labels
  :type data, labels: numpy arrays
  :param original_shape: shape of the one-hot encoded MSA (#sequences, length of the sequences, #possible values at one position)
  :type original_shape: tuple of int
  :param model_type: type of model that has to be built, if "linear" then an instance of LinearNetwork will be created, if "non-linear"
      then it will be an instance of NonLinearNetwork and if "mix" then it will be a MixNetwork
  :type model_type: string
  :param activation: if model_type is "non-linear" or "mix", it is the activation function of the hidden layer, can be "square" or "tanh"
  :type activation: string
  :param max_epochs: duration of training
  :type max_epochs: int
  :param learning_rate: learning rate of Adam optimizer used to train the model
  :type learning_rate: float
  :param weight_decay: regularization strength
  :type weight_decay: float
  :param nb_hidden_neurons: for non-linear or mix, number of neurons per bloc in the hidden layer, for linear the parameter will be ignored
  :tyoe nb_hidden_neurons: int
  :param validation: if true, a validation set will be created and the return values "errors"  and "errors_positions" will contain
      an additional dimension containing the validation errors
  :type validation: bool
  :param test: if true, a test set will be created and the return values "errors"  and "errors_positions" will contain
      an additional dimension containing the test errors
  :type test: bool
  :return: trained model, errors after each epoch, final errors for each position
  :rtype: (nn.Module, list of list of float, list of list of float)
  """
  
  print("Training duration : ", max_epochs, "epochs")
  print("Model type : ", model_type)
  if model_type == "non-linear" or model_type == "mix" : print("Activation function : ", activation)

  #generate training / validation / test datasets from the data and the corresponding labels
  if validation and test : separations = (0.6, 0.8)
  elif validation : separations = (0.8, 1)
  elif test : separations = (0.8, 0.8)
  else : separations = (1, 1)
  training_set, validation_set, test_set = create_datasets(data, labels, separations)
  
  (N,L,K) = original_shape

  #parameters
  params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 2}

  #create data loader for the training_set
  training_generator = torch.utils.data.DataLoader(training_set, **params)

  #define the model according to model_type (linear, non-linear or mix) and, if not linear, activation (square or tanh)
  if model_type == "linear" :

    #list of tuples (input, output) that we want to be disconnected in the model
    indices_mask = []
    for j in range(0, L*K, K) :
      for a in range(j, j+K) :
        indices_mask += [(a, b) for b in range(j, j+K)]

    model = LinearNetwork(indices_mask, training_set.labels.shape[1], original_shape)

  elif model_type == "non-linear" :
    
    #list of tuples (input, hidden layer) that must be disconnected in the model
    indices_mask1 = []
    #list of tuples (hidden layer, output) that must be disconnected in the model
    indices_mask2 = []

    for j in range(0, L) :
      for a in range(j*nb_hidden_neurons, j*nb_hidden_neurons+nb_hidden_neurons) :
        indices_mask1 += [(a, b) for b in range(j*K, (j+1)*K)]
        indices_mask2 += [(b,a) for b in chain(range(0, j*K), range((j+1)*K,L*K))]
    
    model = NonLinearNetwork(indices_mask1, indices_mask2, L*K, L*nb_hidden_neurons, (N,L,K), activation)

  elif model_type == "mix" :

    #list of tuples (input, output) that we want to be disconnected in the model
    indices_mask_linear = []
    for j in range(0, L*K, K) :
      for a in range(j, j+K) :
        indices_mask_linear += [(a, b) for b in range(j, j+K)] #list of tuples (input, output) that must be disconnected in the model

    #list of tuples (input, hidden layer) that must be disconnected in the model
    indices_mask1 = []
    #list of tuples (hidden layer, output) that must be disconnected in the model
    indices_mask2 = []

    for j in range(0, L) :
      for a in range(j*nb_hidden_neurons, j*nb_hidden_neurons+nb_hidden_neurons) :
        indices_mask1 += [(a, b) for b in range(j*K, (j+1)*K)] #list of tuples (input, hidden layer) that must be disconnected in the model
        indices_mask2 += [(b,a) for b in chain(range(0, j*K), range((j+1)*K,L*K))] #list of tuples (hidden layer, output) that must be disconnected in the model
    
    model = MixNetwork(indices_mask1, indices_mask2, indices_mask_linear, L*K, L*nb_hidden_neurons, (N,L,K), activation)

  else :
    print("unknown model type, linear taken instead")

    #list of tuples (input, output) that we want to be disconnected in the model
    indices_mask = []
    for j in range(0, L*K, K) :
      for a in range(j, j+K) :
        indices_mask += [(a, b) for b in range(j, j+K)] #list of tuples (input, output) that we want to be disconnected in the model

    model = LinearNetwork(indices_mask, training_set.labels.shape[1], original_shape)
  
  
  #optimizer with the learning rate and weight decay given in parameters
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = weight_decay)


  # CUDA for PyTorch
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda:0" if use_cuda else "cpu")
  torch.backends.cudnn.benchmark = True
  model = model.to(device)


  #lists that will contain the errors after each epoch
  train_errors = []
  if validation == True : validation_errors = []
  if test ==True : test_errors = []


  print("training...")
  # Loop over epochs
  for epoch in range(max_epochs):
      # Training
      for local_batch, local_labels in training_generator:
          # Transfer to GPU
          local_batch, local_labels = local_batch.to(device), local_labels.to(device)

          #update weights
          train(local_batch, local_labels, model, loss_function, optimizer)
      #display the current epoch
      print("epoch = ", epoch)
      #add the errors to the lists after each epoch using the current weights of the model and display them
      train_errors.append(error(training_set.data, training_set.labels, model, original_shape))
      print("train error : ", error(training_set.data, training_set.labels, model, original_shape))
      if validation == True :
        validation_errors.append(error(validation_set.data, validation_set.labels, model, original_shape))
        print("validation error : ", error(validation_set.data, validation_set.labels, model, original_shape))
      if test == True :
        test_errors.append(error(test_set.data, test_set.labels, model, original_shape))
        print("test error : ", error(test_set.data, test_set.labels, model, original_shape))

  print("trained")

  #put all the errors in a list
  errors = [train_errors]
  if validation == True : errors.append(validation_errors)
  if test == True : errors.append(test_errors)

  #compute the final error per position
  errors_positions = [error_positions(training_set.data, training_set.labels, model, original_shape)]
  if validation == True : errors_positions.append(error_positions(validation_set.data, validation_set.labels, model, original_shape))
  if test == True : errors_positions.append(error_positions(test_set.data, test_set.labels, model, original_shape))
  

  return model, errors, errors_positions





def execute(MSA_file, weights_file, model_type, activation, output_name) :
  """
  train a model according to model_type and activation on the data contained in the given files

  :param MSA_file: name of the file containing the preprocessed MSA
  :type MSA_file: string
  :param weights_file: name of the file containing the weights of the sequences
  :type weights_file: string
  :param model_type: described the model that has to be built and trained, can be "linear" (will produce an instance of LinearNetwork),
      "non-linear" (instance of NonLinearNetwork) or "mix" (instance of MixNetwork)
  :type model_type: string
  :param activation: if model_type is "non-linear" or "mix", activation will be the activation function of the hidden layer, can be "square"
      or "tanh", if model_type is "linear" this parameter will be ignored
  :type activation: string
  :param output_name: the function will save the trained models, the errors after each epoch and the final errors for each position, output_name
      is the name of the output files
  :type output_name: string
  """

  #######################################################
  #learning parameters :
  learning_rate = 1e-3
  weight_decay = 1e-1
  if model_type == "non-linear" :
    max_epochs = 20
    nb_hidden_neurons = 32
  elif model_type == "mix" :
    max_epochs = 10
    nb_hidden_neurons = 20
  else :
    max_epochs = 10
    nb_hidden_neurons = 0
  #######################################################


  data, labels, original_shape = get_data_labels(MSA_file, weights_file)
  print("MSA shape : ", original_shape)

  model, errors, errors_positions = build_and_train_model(data, labels, original_shape, model_type, activation, max_epochs=max_epochs, learning_rate=learning_rate, weight_decay=weight_decay, nb_hidden_neurons=nb_hidden_neurons, test=True)

  #plot learning curve
  plt.plot(range(len(errors[0])), errors[0], label="train error")
  plt.plot(range(len(errors[1])), errors[1], label="test error")
  plt.ylabel("categorical error")
  plt.xlabel("epoch")
  plt.legend()
  plt.grid()
  plt.show()


  #save model and errors
  model_name = "model_" + output_name
  errors_name = "errors_" + output_name + ".txt"
  errors_positions_name = "errors_positions_" + output_name + ".txt"
  
  torch.save(model, model_name)
  np.savetxt(errors_name, errors)
  np.savetxt(errors_positions_name, errors_positions)
