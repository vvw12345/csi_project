import numpy as np
import math

#定义read_bf_file功能，解码语言为python
#整个.dat文件是由n个bfee组成的，而csi的采样信息也是n个，因此bfee和采样信息一一对应
#bfee结构：2字节的field_len + 1字节的code + 可变长度的field
def read_bf_file(filename,decoder="python"):
    
    #读取所有bfee的数据并且放进列表bfee_list里面去
    with open(filename,"rb") as f: #只读二进制形式打开该文件
        bfee_list = []
        #读取两字节数据，并转换为一个无符号数（采用大端法）获得field_len
        field_len = int.from_bytes(f.read(2), byteorder='big', signed=False)
        while field_len != 0:
            #读取长度为field_len的数据放进去
            bfee_list.append(f.read(field_len))
            field_len = int.from_bytes(f.read(2),byteorder='big', signed=False)
            
    dicts = []
    
    count = 0 #有效信道包的数量
    broken_perm = 0 #标志位，该csi数据是否是破损的
    triangle = [0,1,3]
    
    len_arr=len(bfee_list[0])
    #对所有的数据循环处理
    for array in bfee_list:
        if len(array)!=len_arr:
            break
        
        code = array[0]
        
        #如果code编码不是187的话，代表这个不是信道的信息，可以忽略
        if code != 187:
            continue
        else:
            count = count + 1
            
            #当code是187的时候，这个时候field的数据格式有如下的字段
            #其中有20个字节的固定头部和不定长的payload
            
            #时间戳间隔和硬件设备以及发包频率有关
            timestamp_low = int.from_bytes(array[1:5], byteorder='little', signed=False) #时间戳，其单位是微秒
            bfee_count = int.from_bytes(array[5:7], byteorder='little', signed=False) #发送波束，也就是接收的数据包个数
            Nrx = array[9] #接收天线数
            Ntx = array[10] #发送天线数
            #从接收端的三根天线得到的RSSI值
            rssi_a = array[11] 
            rssi_b = array[12]
            rssi_c = array[13]
            noise = array[14] - 256  #noise（噪声值）是char类型，-256把其从无符号数换成char类型
            agc = array[15]  #天线自动增益大小
            antenna_sel = array[16]
            b_len = int.from_bytes(array[17:19], byteorder='little', signed=False)
            fake_rate_n_flags = int.from_bytes(array[19:21], byteorder='little', signed=False)
            payload = array[21:]
            
            #通过以下的公式计算出一个长度len
            calc_len = (30 * (Nrx * Ntx * 8 * 2 + 3) + 6) / 8
            perm = [1,2,3] #接收天线的处理顺序
            perm[0] = ((antenna_sel) & 0x3)
            perm[1] = ((antenna_sel >> 2) & 0x3)
            perm[2] = ((antenna_sel >> 4) & 0x3)
            
            #如果给出的len和计算出来的不相等，那么代表出错
            if (b_len != calc_len):
                print("MIMOToolbox:read_bfee_new:size","Wrong beamforming matrix size.")
            
            #如果解码的语言不为python的话就退出
            if decoder=="python":
                csi = parse_csi(payload,Ntx,Nrx)
            else:
                csi = None
                print("decoder name error! Wrong encoder name:",decoder)
                return
            
            if sum(perm) != triangle[Nrx - 1]:
                print('WARN ONCE: Found CSI (', filename, ') with Nrx=', Nrx, ' and invalid perm=[', perm, ']\n')
            else:
                csi[:, perm, :] = csi[:, [0, 1, 2], :]
                
            bfee_dict = {
                'timestamp_low': timestamp_low,
                'bfee_count': bfee_count,
                'Nrx': Nrx,
                'Ntx': Ntx,
                'rssi_a': rssi_a,
                'rssi_b': rssi_b,
                'rssi_c': rssi_c,
                'noise': noise,
                'agc': agc,
                'antenna_sel': antenna_sel,
                'perm': perm,
                'len': b_len,
                'fake_rate_n_flags': fake_rate_n_flags,
                'calc_len': calc_len,
                'csi': csi}
            
            dicts.append(bfee_dict)
            
    return dicts


#parse_csi_new和parse_csi的区别在于数据矩阵的格式不一样
def parse_csi_new(payload,Ntx,Nrx):
    csi = np.zeros(shape=(30,Nrx ,Ntx), dtype=np.dtype(complex))
    index = 0

    for i in range(30):
        index += 3
        remainder = index % 8
        for j in range(Nrx):
            for k in range(Ntx):
                real_bin = (int.from_bytes(payload[int(index / 8):int(index/8+2)], byteorder='big', signed=True) >> remainder) & 0b11111111
                real = real_bin
                imag_bin = bytes([(payload[int(index / 8+1)] >> remainder) | (payload[int(index/8+2)] << (8-remainder)) & 0b11111111])
                imag = int.from_bytes(imag_bin, byteorder='little', signed=True)
                tmp = complex(float(real), float(imag))
                csi[i, j, k] = tmp
                index += 16
    return csi


def parse_csi(payload,Ntx,Nrx):
    #把一个数组初始化为0，这个数组将用来存储复数
    csi = np.zeros(shape=(Ntx,Nrx,30), dtype=np.dtype(complex))
    index = 0
    
    #payload部分是不定长的，用来存储csi数据的30个子载波的信息
    #subc数据结构如下：3bit的subc头部 + （Nrx * Ntx * 2) * 8长度的有效部分(也就是Nrx * Ntx个复数)
    #遍历30个子载波，然后遍历每根发送天线和接收天线，解析出复数值之后组合起来
    #k代表发送天线，j代表接收天线，i代表子载波
    for i in range(30):
        index += 3
        remainder = index % 8
        for j in range(Nrx):
            for k in range(Ntx):
                start = index // 8
                real_bin = bytes([(payload[start] >> remainder) | (payload[start+1] << (8-remainder)) & 0b11111111])
                real = int.from_bytes(real_bin, byteorder='little', signed=True)
                imag_bin = bytes([(payload[start+1] >> remainder) | (payload[start+2] << (8-remainder)) & 0b11111111])
                imag = int.from_bytes(imag_bin, byteorder='little', signed=True)
                tmp = complex(float(real), float(imag))
                csi[k, j, i] = tmp
                index += 16
    return csi