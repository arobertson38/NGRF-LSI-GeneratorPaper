"""
Hello! Thanks for your interest in playing around with my generator. I am providing
this code in hopes that it acts as a simple tutorial for using the generator. 

The generators are stored in their corresponding classes in StochasticGeneration. 
EigenGenerator is the class corresponding to the generator described in the paper. 

In general, the following workflow needs to be followed:

(1) compute/identify autocorrelation and necessary crosscorrelations
(2) Initialize the generator with the two-point statistics
(3) Initialize your desired filters (unless you don't want any)
(4) .generate() to generate microstructures (2 at a time)


Created by: Andreas E. Robertson
Contact: arobertson38@gatech.edu
"""
from StochasticGeneration import *
import numpy as np
import matplotlib.pyplot as plt
from HelperFunctions_StochasticGeneration import *

struct_names = ['./Example_Data/TwoPhaseSimple.npy',
                './Example_Data/TwoPhaseElongated.npy',
                './Example_Data/ThreePhaseElongated.npy',
                './Example_Data/ThreePhaseCircles.npy',
                './Example_Data/ThreePhaseInvertedCircles.npy',
                './Example_Data/ThreePhaseExperimental.npy',
                './Example_Data/ThreePhaseComplex.npy',
                './Example_Data/TwoPhaseUniformCircles.npy',
                './Example_Data/TwoPhaseDenseUniformCircles.npy',
                './Example_Data/TwoPhaseUniformInvertedCircles.npy',
                './Example_Data/TwoPhaseDenseUniformCircles.npy',
                './Example_Data/ThreeD_Example.npy']

# Microstructure and Generator choices:
struct_index = -2
generator = EigenGenerator

# Load the microstructure and compute statistics:
struct = np.load(struct_names[struct_index])
stats = twopointstats(struct)
print(stats.sum())

# Checking if the structure you picked is 3D: 
twoD = True if len(struct.shape)==3 else False

# generation:
# The generator code is written to accept two-point statistics that haven't been centered. argmax(f_s) -> 0,0,0 etc
gen = generator(stats, 'complete') # complete is the default parameter indicating that a complete row of 
# 2PS have been given (a complete row is returned by twopointstats). ('incomplete' is the other option)

gen.filter('flood', alpha=0.3, beta=0.35) # default parameters from the paper. (if no filter is desired either 
# input 'none' or don't call this method. 

sampled_micro_1, sampled_micro_2 = gen.generate()


# Plotting
if twoD:
    f = plt.figure(figsize=[8, 4.5])
    ax1 = f.add_subplot(121)
    ax2 = f.add_subplot(122)
    if struct.shape[-1] == 2:
        # There are two local states (i.e., phases) 
        ax1.imshow(struct[..., 0])
        ax2.imshow(sampled_micro_1[..., 0])
    else:
        ax1.imshow(struct)
        ax2.imshow(sampled_micro_1)

    ax1.set_title('Original Microstructure')
    ax2.set_title('Generated Sample')

else:
    f = plt.figure(figsize=[8, 4.5])
    ax1 = f.add_subplot(121)
    ax2 = f.add_subplot(122)
    if struct.shape[-1] == 2:
        # There are two local states (i.e., phases) 
        ax1.imshow(struct[..., int((struct.shape[2]-1)/2), 0])
        ax2.imshow(sampled_micro_1[..., int((struct.shape[2]-1)/2), 0])
    else:
        ax1.imshow(struct[..., int((struct.shape[2]-1)/2), :])
        ax2.imshow(sampled_micro_1[..., int((struct.shape[2]-1)/2), :])

    ax1.set_title('Original Microstructure')
    ax2.set_title('Generated Sample')

plt.show()

