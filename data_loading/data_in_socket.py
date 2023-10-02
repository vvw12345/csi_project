import socket
import struct
import matplotlib.pyplot as plt
import numpy as np
from data_loading.csi_getting import *

def read_bfee(bytes):
    pass

#为了满足实时监听的需求，我们需要使用python对进入socket套接字的csi数据进行实时监听，而不是存储到文件里面去
def read_bf_socket(client_socket):
    csi_entry = None
    index = -1 #我们需要展示的图像编号
    broken_perm = 0 #标志位，确定我们是否收到了破损的csi数据
    triangle = [1,3,6] #天线处理顺序标志
    
    while True:
        try:
            # 读取大小和数据编码号
            field_len = struct.unpack('H', client_socket.recv(2))[0]
            code = client_socket.recv(1)[0]
            
            #当编码是187的时候代表的是csi数据
            if code == 187:
                bytes = client_socket.recv(field_len - 1)
                csi_entry = read_bfee(bytes)
                perm = csi_entry['perm']
                Nrx = csi_entry['Nrx']
                
                if Nrx > 1:
                    if sum(perm) != triangle[Nrx]:
                        if not broken_perm:
                            broken_perm = 1
                            print(f'WARN ONCE: Found CSI with Nrx={Nrx} and invalid perm={perm}')
                else:
                    csi_entry['csi'][:, perm[:Nrx], :] = csi_entry['csi'][:, :Nrx, :]
                    
            index = (index + 1) % 10
            csi = get_scale_csi(csi_entry)
            
            # 更新图形
            p[0].set_xdata(np.arange(1, 31))
            p[0].set_ydata(10 * np.log10(np.abs(csi[0, 0, :])))

            plt.pause(0.01)

        except KeyboardInterrupt:
            break


# 创建tcp连接在线解析（是从端口读入数据，而不是从文件读入数据）
host = '0.0.0.0'
port = 8090

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
    server_socket.bind((host, port))
    server_socket.listen(1)

    print(f"Waiting for connection on port {port}")
    client_socket, client_address = server_socket.accept()
    print(f"Accept connection from {client_address}")
    
    # 设置绘图参数
    plt.figure()
    plt.axis([1, 30, -10, 30])
    t1 = 0
    m1 = np.zeros(30)
    p = plt.plot(t1, m1, 'b-', markersize=5)
    plt.xlabel('Subcarrier index')
    plt.ylabel('SNR (dB)')

    read_bf_socket(client_socket)

    client_socket.close()