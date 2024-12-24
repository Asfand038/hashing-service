from flask import Flask, request, jsonify
import numpy as np
from scipy.signal import convolve2d
import math
from functools import reduce
import operator

app = Flask(__name__)

#################

# converts string to an array 
# containing the ascii equivalent of each character in the string
def string_to_ascii(input_string):
    return [ord(char) for char in input_string]

# calculates the element-wise xor between lists in the provided list
# returns a single list containing the result
def elementwise_xor(lists):
    return np.array([(reduce(operator.xor, pair)) for pair in zip(*lists)])

# converts an array of 16-bit values to hexadecimal
# and then converts the hexadecimal to string
def convert_to_hex(values):
    return ' '.join(format(value, '04x') for value in values)

# repeats the supplied array until it is a multiple of 64
def repeat_until_multiple_of(arr, multiple_of):
    length = len(arr)
    required_length = multiple_of
    while (length > required_length):
        required_length += multiple_of
    
    pointer = 0
    new_arr = np.copy(arr)
    while (len(new_arr) != required_length):
        new_arr = np.append(new_arr,arr[pointer])
        pointer = (pointer+1) % length
    return new_arr

# takes in 4 4-length arrays and computes a 2x2 kernel from their values
def generate_kernel_from_arrays(v1, v2, v3, v4):
    a = (v1[0] + v2[0] + v3[0] + v4[0]) / 4
    b = (v1[1] + v2[1] + v3[1] + v4[1]) / 4
    c = (v1[2] + v2[2] + v3[2] + v4[2]) / 4
    d = (v1[3] + v2[3] + v3[3] + v4[3]) / 4
    k1 = a+b*c*d
    k2 = b+a*c*d
    k3 = c+a*b*d
    k4 = d+a*b*c
    return np.array([k1,k2,k3,k4]).reshape(2,2)

# divides a 64-length array into 16 4-length arrays
# A different kernel is created for every iteration
# kernel 1: generated from values in the 1st 5th  9th and 13th array
# kernel 2: generated from values in the 2nd 6th 10th and 14th array
# kernel 3: generated from values in the 3rd 7th 11th and 15th array
# kernel 4: generated from values in the 4th 8th 12th and 16th array 
def generate_kernel_list(block):
    array_list = block.reshape(16, 4)
    kernel1 = generate_kernel_from_arrays(array_list[0],array_list[1],array_list[8],array_list[9])
    kernel2 = generate_kernel_from_arrays(array_list[2],array_list[3],array_list[10],array_list[11])
    kernel3 = generate_kernel_from_arrays(array_list[4],array_list[5],array_list[12],array_list[13])
    kernel4 = generate_kernel_from_arrays(array_list[6],array_list[7],array_list[14],array_list[15])
    return [kernel1,kernel2,kernel3,kernel4]

# applies a convolution on the input array,
# using the supplied 2x2 kernel with a stride of 1
def apply_custom_convolution(input_array, kernel):
    one_side_length = int(math.sqrt(len(input_array)))
    matrix = input_array.reshape((one_side_length, one_side_length))
    convolved = convolve2d(matrix, kernel, mode='valid')
    result = convolved.flatten()
    result = result + np.arange(len(result))
    return result

# our main hashing algorithm converts a list of variable length 
# containing 8-bit values into 64-length blocks 
# by adding metadata and repeating the values
# then generates kernels based on the blocks
# and repeatedly applies convolutions until each block
# is of length 16, then XORs the contents of each blocks with each other block
# The end result is a 16-length array with 16-bit values i.e. 256 bit hash value
# that is represented as hexadecimal in string format and then returned 
def custom_hash(input_array):

    # add metadata
    new_list = [0] * (len(input_array)+1)
    new_list[0] = len(input_array)
    for i in range(1,len(input_array)+1): new_list[i] = input_array[i-1]
    new_list.append(sum(new_list))

    # make input length equal to a multiple of 64
    input_array = np.array(new_list).astype(float)
    input_array = repeat_until_multiple_of(input_array, 64)

    # split into blocks
    blocks = np.split(input_array, len(input_array) // 64)

    # make 4 kernels for every block
    kernel_lists = []
    for i in range(len(blocks)):
        kernel_lists.append(generate_kernel_list(blocks[i]))

    # apply convolutions to each blocks until the block only has 16 elements
    iteration = 0
    while len(blocks[0]) > 16:
        for i in range(len(blocks)):
            kernel = kernel_lists[i][iteration]
            blocks[i] = apply_custom_convolution(blocks[i], kernel)
            blocks[i] = blocks[i]%(2**16)

        iteration += 1

    for i in range(len(blocks)): blocks[i] = blocks[i].astype(int)

    result = elementwise_xor(blocks)
    return convert_to_hex(result)

#################

@app.route('/hash', methods=['POST'])
def hash_endpoint():
    data = request.get_json()
    if not data or 'input' not in data:
        return jsonify({'error': 'Input string is required'}), 400

    input_string = data['input']
    input_array = string_to_ascii(input_string)
    hashed_value = custom_hash(input_array)
    return jsonify({'hash': hashed_value})

if __name__ == '__main__':
    app.run()