# -*-coding: utf-8-*-
import numpy as np
from bram import BRAM, BramConfig

class Matmul(object):
    '''矩阵乘法
        Args: uint8, (m, n)
        Args: int8, (n, p)
    '''

    def __init__(self):
        self.systolic_size = 4 # 脉动阵列大小
        self.bram = BRAM()
        pass

    def __call__(self, input: np.uint8, weight: np.int8):
        self.send_data(input, 'input')
        self.send_data(weight, 'weight')
        self.send_instr(input.shape[0], weight.shape[1], input.shape[1])
        self.send_flag()
        output_arr = self.recv_output((input.shape[0], weight.shape[1]))
        return output_arr

    def send_data(self, data, block_name, offset='default'):
        '''写入input或weight至bram

            假设两个矩阵分别是(m,n) x (n,p), m和p的维度需要补全至self.systolic_size的倍数,
            并且写入时需要按照补零的方向写入,例如：  
                1. 矩阵(m, n)是m补零,则m个m个写入BRAM中。(行方向补零,列方向写入）  
                2. 矩阵(n, p)是p补零,则p个p个写入BRAM中。(列方向补零,行方向写入）
            
            Args:
                data: 要写入的数据
                block_name: input, weight
                offset: 偏移地址名称,默认为default
        '''
        if block_name == 'input':
            d = data.T
            z = np.zeros(d.shape[0], dtype=np.uint8)
        else:
            d = data
            z = np.zeros(d.shape[0], dtype=np.int8)
        l = d.shape[1]
        t = l % self.systolic_size
        t = self.systolic_size - t
        t = t % self.systolic_size
        while t > 0:
            d = np.insert(d, d.shape[1], z, axis=1)
            t = t - 1
        d = d.copy()
        self.bram.write(d, block_name=block_name)
        pass

    def send_instr(self, m, p, n):
        '''构建并发送指令

            两个矩阵shape分别为(m,n) x (n,p)
        '''
        instr = np.array([m, p, n, 0], dtype=np.uint16)
        self.bram.write(instr, block_name='ir', offset='instr')
        pass

    def send_flag(self):
        '''发送flag=1信号'''
        self.bram.write( b"\x01\x00\x00\x00" , block_name = "ir" , offset = 'flag' )
        pass

    def recv_flag(self):
        flag = self.bram.read( 1 , block_name = "ir" , offset = 'flag' )[0]
        return flag
        
    def recv_output(self, output_shape: tuple):
        '''接收结果

            Args:
                output_shape: 输出的shape,类型tuple

            Return:
                output_arr: shape为output_shape的np.ndarray
        '''
        value = -1
        while( value != 0):
            value = self.recv_flag()
        output_arr = self.bram.read(4 * output_shape[0] * output_shape[1], block_name = "output" , dtype = np.int32)
        output_arr = output_arr.reshape(output_shape[0], output_shape[1])
        return output_arr


if __name__ == '__main__':
    matmul = Matmul()

    ############ matrix 1
    x = np.random.randint(0, 2, (4,8), dtype=np.uint8)
    w = np.random.randint(-1, 2, (8,4), dtype=np.int8)

    std_output = np.matmul(x, w)
    output = matmul(x, w)

    # err = output - std_output
    assert (output == std_output).all(), 'error'
    print('~~~ demo1 pass ~~~')

    ############ matrix 2
    x = np.random.randint(0, 5, (15,20), dtype=np.uint8)
    w = np.random.randint(-5, 5, (20,10), dtype=np.int8)

    std_output = np.matmul( x , w )
    output = matmul(x, w)

    assert (output == std_output).all(), 'error'
    # err = output - std_output
    assert (output == std_output).all(), 'error'
    print('~~~ demo2 pass ~~~')