"""pytorchfi.error_models provides different error models out-of-the-box for use."""

import random
from typing import List
import torch

import pytorchfi.core as core
from pytorchfi.util import random_value

from struct import *


class weight_bit_flip_func(core.FaultInjection):
    def __init__(self, model, batch_size: int, input_shape: List[int] = None, layer_types=None, **kwargs):
        super().__init__(model, batch_size, input_shape, layer_types, **kwargs)

    
    # getting weight location randomly
    def random_weight_location(self, layer: int = -1):
        if layer == -1:
            layer = random.randint(0, self.get_total_layers() - 1)

        dim = self.get_weights_dim(layer)
        shape = self.get_weights_size(layer)

        dim0_shape = shape[0]
        k = random.randint(0, dim0_shape - 1)
        if dim > 1:
            dim1_shape = shape[1]
            dim1_rand = random.randint(0, dim1_shape - 1)
        if dim > 2:
            dim2_shape = shape[2]
            dim2_rand = random.randint(0, dim2_shape - 1)
        else:
            dim2_rand = None
        if dim > 3:
            dim3_shape = shape[3]
            dim3_rand = random.randint(0, dim3_shape - 1)
        else:
            dim3_rand = None

        return ([layer], [k], [dim1_rand], [dim2_rand], [dim3_rand])
    
    
    # define bit_flip function
    def _random_flip_bit(self, data,location):
        # print('--------------------')
        # ---------------------------
        bit_position = random.randint(0, 31)
        # ---------------------------
        # bit_positions = [22, 23, 24, 25, 26, 31]
        # loc = random.randint(0, 5)
        # bit_position = bit_positions[loc]
        # -----------------------------
        # bit_position = random.randint(0, 31)
        # if bit_position == 30 :
        #     bit_position = 0
        # ------------------------------
        # bit_position = 22
        # print(f'bit_position:{bit_position}')
        origin_data_type = data[location].dtype
        origin_value = data[location]
        # print(f'origin_value:{origin_value}')
        fs = pack('f', origin_value)
        float_binary = ''.join(f'{byte:08b}' for byte in fs)
        # print(f'binary_value:{float_binary}')
        binary_value = list(unpack('BBBB',fs))
        [q, r] = divmod(bit_position, 8)
        # bit-flip
        # binary_value[q] ^= 1 << r

        # flip to 1
        binary_value[q] |= 1 << r

        # flip to 0
        # if binary_value[q] & (1<<r) != 0:
        #     binary_value[q] ^= 1 << r

        fs = pack('BBBB', *binary_value)
        float_binary = ''.join(f'{byte:08b}' for byte in fs)
        # print(f'binary_value:{float_binary}')
        fnew = unpack('f', fs)
        new_value = fnew[0]
        # print(f'new_value:{new_value}')
        # print(new_value-origin_value)
        return torch.tensor(new_value,dtype=origin_data_type)




    
    # injecting zero
    def _zero_rand_weight(data, location):
        new_data = data[location] * 0
        return new_data    


# Weight Perturbation Models
def random_weight_inj(pfi, corrupt_layer: int = -1, min_val: int = -1, max_val: int = 1):
    layer, k, c_in, kH, kW = pfi.random_weight_location(corrupt_layer)
    faulty_val = [random_value(min_val=min_val, max_val=max_val)]

    return pfi.declare_weight_fault_injection(
        layer_num=layer, k=k, dim1=c_in, dim2=kH, dim3=kW, value=faulty_val
    )

def zero_func_rand_weight(pfi):
    layer, k, c_in, kH, kW = pfi.random_weight_location()
    return pfi.declare_weight_fault_injection(
        function=pfi._zero_rand_weight, layer_num=layer, k=k, dim1=c_in, dim2=kH, dim3=kW
    )


# decide to inject fault according to SDC_rate
def multi_weight_inj_sdc_rate(pfi, sdc_p=1e-5):
    corrupt_idx = [[], [], [], [], []]
    for layer_idx in range(pfi.get_total_layers()):
        shape = list(pfi.get_weights_size(layer_idx))
        dim_len = len(shape)
        shape.extend([1 for i in range(4 - len(shape))])
        for k in range(shape[0]):
            for dim1 in range(shape[1]):
                for dim2 in range(shape[2]):
                    for dim3 in range(shape[3]):
                        if random.random() < sdc_p:
                            idx = [layer_idx, k, dim1, dim2, dim3]
                            for i in range(dim_len + 1):
                                corrupt_idx[i].append(idx[i])
                            for i in range(dim_len + 1, 5):
                                corrupt_idx[i].append(None)
    return pfi.declare_weight_fault_injection(
        layer_num=corrupt_idx[0],
        k=corrupt_idx[1],
        dim1=corrupt_idx[2],
        dim2=corrupt_idx[3],
        dim3=corrupt_idx[4],
        function=pfi._random_flip_bit,
    )

# calculate the number of faults accoring to the fault rate
def multi_weight_inj_fault_rate(pfi, fault_rate=1e-5):
    corrupt_idx = [[], [], [], [], []]
    number_of_weights = 0

    # 统计参数数量
    for layer_idx in range(pfi.get_total_layers()):
        shape = list(pfi.get_weights_size(layer_idx))
        number_of_weights = number_of_weights + shape[0] * shape[1] * shape[2] * shape[3] 
    print(f'number_of_weights:{number_of_weights}')
    # 计算注入错误的数量
    number_of_faluts = number_of_weights*fault_rate
    print(f'number_of_faluts:{number_of_faluts}')
    # 随机生成一个故障位置
    i = 0
    while i < number_of_faluts:
        layer, k, c_in, kH, kW= pfi.random_weight_location()
        corrupt_idx[0].append(layer[0])
        corrupt_idx[1].append(k[0])
        corrupt_idx[2].append(c_in[0])
        corrupt_idx[3].append(kH[0])
        corrupt_idx[4].append(kW[0])
        i = i + 1

    return pfi.declare_weight_fault_injection(
        layer_num=corrupt_idx[0],
        k=corrupt_idx[1],
        dim1=corrupt_idx[2],
        dim2=corrupt_idx[3],
        dim3=corrupt_idx[4],
        function=pfi._random_flip_bit,
    )