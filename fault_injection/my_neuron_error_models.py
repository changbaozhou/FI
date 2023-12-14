import logging
import torch
from pytorchfi import core
from pytorchfi.util import *
from struct import *

# logging.basicConfig(level=logging.DEBUG)

class single_bit_flip_func(core.fault_injection):
    def __init__(self, model, batch_size, input_shape=None, **kwargs):
        if input_shape is None:
            input_shape = [3, 224, 224]
        super().__init__(model, batch_size, input_shape=input_shape, **kwargs)
        logging.basicConfig(format="%(asctime)-15s %(clientip)s %(user)-8s %(message)s")

        # self.bits = kwargs.get("bits", 32)
        self.inj_bit = 0
        self.neg_ratio = 0
        self.zero_ratio = 0
        self.data_precision = "fp32"
        self.LayerRanges = []
        self.value_delta = 0

    def set_conv_max(self, data):
        self.LayerRanges = data

    def reset_conv_max(self, data):
        self.LayerRanges = []

    def get_conv_max(self, layer):
        return self.LayerRanges[layer]

    @staticmethod
    def _twos_comp(val, bits):
        if (val & (1 << (bits - 1))) != 0:
            val = val - (1 << bits)
        return val

    def _twos_comp_shifted(self, val, nbits):
        return (1 << nbits) + val if val < 0 else self._twos_comp(val, nbits)


    def _flip_bit_signed(self, orig_value, bit_pos):
        save_type = orig_value.dtype
        #pack用于将数据转换成字节流
        fs = pack('f', orig_value)
        #unpack用于将字节流转换成Python数据类型
        bval = list(unpack('BBBB', fs)) #得到输入数据的二进制表示
        [q, r] = divmod(bit_pos, 8) #divmod(a,b)计算a除以b的商和余数
        #异或（不同结果为１,相同结果为０），通过与１异或实现比特翻转
        bval[q] ^= 1 << r #00000001按位左移r个单位
        fs = pack('BBBB', *bval)
        fnew = unpack('f', fs)
        new_value = fnew[0]
        return torch.tensor(new_value, dtype=save_type)

    def _flip_bit_signed_fp16(self, orig_value, bit_pos):
        save_type = orig_value.dtype
        #pack用于将数据转换成字节流
        fs = pack('f', orig_value)
        #unpack用于将字节流转换成Python数据类型
        bval = list(unpack('BBBB', fs)) #得到输入数据的二进制表示
        [q, r] = divmod(bit_pos, 8) #divmod(a,b)计算a除以b的商和余数
        #异或（不同结果为１,相同结果为０），通过与１异或实现比特翻转
        bval[q] ^= 1 << r #00000001按位左移r个单位
        fs = pack('BBBB', *bval)
        fnew = unpack('f', fs)
        new_value = fnew[0]
        return torch.tensor(new_value, dtype=save_type)

    def transform_fp32tofp16(self):
        if self.inj_bit >= 0 and self.inj_bit <= 13:
            transformed_bit = self.inj_bit + 13
        if self.inj_bit >= 16 and self.inj_bit <= 29:
            transformed_bit = self.inj_bit - 3
        if self.inj_bit == 30 or self.inj_bit == 14:
            transformed_bit = 30
        if self.inj_bit == 31 or self.inj_bit == 15:
            transformed_bit = 31
        return transformed_bit   

    
    def single_bit_flip_signed_across_batch(self, module, input_val, output):
        corrupt_conv_set = self.get_corrupt_layer()
        logging.info("Current layer: %s", self.get_current_layer())
        if type(corrupt_conv_set) is list:
            inj_list = list(
                filter(
                    lambda x: corrupt_conv_set[x] == self.get_current_layer(),
                    range(len(corrupt_conv_set)),
                )
            )
            inj_num = len(self.corrupt_dim1)
            

            for i in inj_list:

                # --------------count the zero_value in input featuremap ------------
                total_num = 0
                zero_num = 0
                input_back = input_val[0]
                input_back = input_back.reshape(-1)
                total_num = len(input_back)
                for element in input_back:
                    if element.item() == 0:
                        zero_num = zero_num +1 
                self.zero_ratio = zero_num/total_num

                # -------------count the neg_value in output featuremap --------------
                # total_num = 0
                # neg_num = 0

                # output_back = output
                # output_back = output_back.reshape(-1)
                # total_num = len(output_back)
                # for element in output_back:
                #     if element.item() < 0:
                #         neg_num = neg_num +1 
                # self.neg_ratio = neg_num/total_num

                total_delta = 0

                rand_bit = self.inj_bit
                # print(f"rand_bit:{rand_bit}")
                for i in range(inj_num):
                    # self.assert_inj_bounds(index=i)
                    prev_value = output[self.corrupt_batch[i]][self.corrupt_dim1[i]][self.corrupt_dim2[i]][self.corrupt_dim3[i]]
                    if self.data_precision == "fp16":
                        rand_bit = self.transform_fp32tofp16()
                        prev_value = prev_value.half()
                    # print(f"prev_value:{prev_value}")
                    logging.info(f"prev_value:{prev_value}")
                    logging.info("Random Bit: %d", rand_bit)
                    new_value = self._flip_bit_signed(prev_value, rand_bit)
                    # print(f"new_value:{new_value}")
                    # total_delta = total_delta + abs(new_value-prev_value)

                    logging.info(f"new_value:{new_value}")
                    output[self.corrupt_batch[i]][self.corrupt_dim1[i]][self.corrupt_dim2[i]][self.corrupt_dim3[i]] = new_value
                # if inj_num != 0:
                #     self.value_delta = total_delta/inj_num
                # else:
                #     self.value_delta = 0
                    
        else:
            if self.get_current_layer() == corrupt_conv_set:
                prev_value = output[self.corrupt_batch][self.corrupt_dim1][self.corrupt_dim2][self.corrupt_dim3]
                rand_bit = random.randint(0, self.bits - 1)
                logging.info("Random Bit: %d", rand_bit)
                new_value = self._flip_bit_signed(prev_value, rand_bit)

                output[self.corrupt_batch][self.corrupt_dim1][self.corrupt_dim2][self.corrupt_dim3] = new_value

        self.updateLayer()
        if self.get_current_layer() >= self.get_total_layers():
            self.reset_current_layer()


'''
-获取batch_element
'''
def random_batch_element(pfi):
    return random.randint(0, pfi.get_total_batches() - 1)

'''
-根据layer, mode, fault_num 获取一层中故障注入的所有位置 
'''
# def random_neuron_location(pfi, layer, mode, fault_num):

    
#     shape = pfi.get_layer_shape(layer)
#     output_size = pfi.get_output_size()
#     # print(f"layer:{layer}")
#     print(f"shape:{shape}")

#     # 故障注入的位置通过list返回
#     batch = [] #batch
#     C = [] #out_channel
#     H = [] #height
#     W = [] #width

#     c_range, h_range, w_range = shape[1], shape[2], shape[3]
    
#     if mode == "one_bit":
#         c = random.randint(0, c_range-1)
#         h = random.randint(0, h_range-1)
#         w = random.randint(0, w_range-1)
#         batch.append(0)
#         C.append(c)
#         H.append(h) 
#         W.append(w) 
#     if mode == "col":
#         h = random.randint(0, h_range-1)
#         w = random.randint(0, w_range-1)
#         if fault_num == c_range or fault_num > c_range:
#             c_start = 0
#             fault_num = c_range
#         else:
#             c_start = (random.randint(0, c_range-fault_num)//fault_num)*fault_num
#         for i in range(fault_num):
#             batch.append(0)
#             C.append(c_start+i)
#             H.append(h)
#             W.append(w)
#     if mode == "row":
#         c = random.randint(0, c_range-1)
#         if fault_num == h_range*w_range or fault_num > h_range*w_range:
#             w_start = 0
#             fault_num = h_range*w_range
#         else:
#             w_start = (random.randint(0, h_range*w_range-fault_num)//fault_num)*fault_num
#         for i in range(w_start, w_start+fault_num):
#             h = i // w_range
#             w = i % w_range
#             batch.append(0)
#             C.append(c)
#             H.append(h)
#             W.append(w)
#     return batch, C, H, W


# googlenet 专用

def random_neuron_location(pfi, layer, mode, fault_num):

    if layer < 21:
        shape = pfi.get_layer_shape(layer)
    # if layer == 21:
    #     return [], [], [], []
    if layer >= 21 and layer < 39:
        shape = pfi.get_layer_shape(layer+1)
    # if layer == 40:
    #     return [], [], [], []
    if layer >= 39:
        shape = pfi.get_layer_shape(layer+2)
    
    # print(f"layer:{layer}")
    # print(f"shape:{shape}")

    # 故障注入的位置通过list返回
    batch = [] #batch
    C = [] #out_channel
    H = [] #height
    W = [] #width

    c_range, h_range, w_range = shape[1], shape[2], shape[3]
    
    if mode == "one_bit":
        c = random.randint(0, c_range-1)
        h = random.randint(0, h_range-1)
        w = random.randint(0, w_range-1)
        batch.append(0)
        C.append(c)
        H.append(h) 
        W.append(w) 
    if mode == "col":
        h = random.randint(0, h_range-1)
        w = random.randint(0, w_range-1)
        if fault_num == c_range or fault_num > c_range:
            c_start = 0
            fault_num = c_range
        else:
            c_start = (random.randint(0, c_range-fault_num)//fault_num)*fault_num
        for i in range(fault_num):
            batch.append(0)
            C.append(c_start+i)
            H.append(h)
            W.append(w)
    if mode == "row":
        c = random.randint(0, c_range-1)
        if fault_num == h_range*w_range or fault_num > h_range*w_range:
            w_start = 0
            fault_num = h_range*w_range
        else:
            w_start = (random.randint(0, h_range*w_range-fault_num)//fault_num)*fault_num
        for i in range(w_start, w_start+fault_num):
            h = i // w_range
            w = i % w_range
            batch.append(0)
            C.append(c)
            H.append(h)
            W.append(w)
    return batch, C, H, W

'''
-根据生成的故障注入位置和自定义的比特翻转方法, 构建injector
'''

def random_neuron_single_bit_inj(pfi, layer, mode, fault_num):

    # batch = random_batch_element(pfi)
    batch, C, H, W = random_neuron_location(pfi, layer, mode, fault_num)
    # batch = [0]
    # C = [41]
    # H = [51]
    # W = [1]

    return pfi.declare_neuron_fi(
        batch= batch,
        layer_num=[layer],
        dim1=C,
        dim2=H, 
        dim3=W,
        function=pfi.single_bit_flip_signed_across_batch
    )


