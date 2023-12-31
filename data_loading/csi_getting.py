import math
import numpy as np

#计算分贝值
def db(X,U):
    R = 1
    #如果说U的开头的功率power，那么程序保证其大于等于0
    if 'power'.startswith(U):
        assert X >= 0 #插入断言，如果不满足抛出异常
    else: #如果是非功率，那么对其做如下的处理
        X = math.pow(abs(X),2) / R
    
    #最后计算分贝值
    return (10 * math.log10(X) + 300) - 300

#接收一个分贝值作为输入，转换为线性值输出
def dbinv(x):
    return math.pow(10,x / 10)

#计算rssa值，检查rssi字典中三根天线的信号强度
#如果信号强度不为0，则转为线性值累加
def get_total_rss(csi_st):
    rssi_mag = 0 #这里注意初始化为0
    if csi_st['rssi_a'] != 0:
        rssi_mag = rssi_mag + dbinv(csi_st['rssi_a'])
    if csi_st['rssi_b'] != 0:
        rssi_mag = rssi_mag + dbinv(csi_st['rssi_b'])
    if csi_st['rssi_c'] != 0:
        rssi_mag = rssi_mag + dbinv(csi_st['rssi_c'])
    return db(rssi_mag, 'power') - 44 - csi_st['agc']

#处理复数形式的CSI数据，并返回缩放处理之后的CSI数据
def get_scale_csi(csi_st):
    #提取CSI数据
    csi = csi_st['csi']
    
    #计算csi数据幅度的平方
    csi_sq = np.multiply(csi, np.conj(csi)).real #计算csi复数和其共轭复数的乘积
    #计算每个时间戳的CSI幅度平方的总和
    csi_pwr = np.sum(csi_sq, axis=0)
    csi_pwr = csi_pwr.reshape(1, csi_pwr.shape[0], -1)
    #计算总的RSS值并且转换为线性值
    rssi_pwr = dbinv(get_total_rss(csi_st))

    #计算缩放因子（用来缩放CSI数据从而考虑信噪比）
    #RSS总的线性值除以每个时间戳的CSI幅度平方总和的均值
    scale = rssi_pwr / (csi_pwr / 30)

    #根据输入的噪声信息计算热噪声的线性值
    #如果noise为-127，则假定热噪声为-92
    if csi_st['noise'] == -127:
        noise_db = -92
    else:
        noise_db = csi_st['noise']
    thermal_noise_pwr = dbinv(noise_db)

    #计算量化误差的线性值
    #等于缩放因子scale * Nrx * Ntx
    quant_error_pwr = scale * (csi_st['Nrx'] * csi_st['Ntx'])

    #计算总的噪声功率
    #热噪声和量化误差的功率之和
    total_noise_pwr = thermal_noise_pwr + quant_error_pwr
    
    #对原始的CSI数据进行缩放处理
    #根据scale和total_noise_pwr来调整CSI的幅度，最后返回处理之后的CSI数据
    ret = csi * np.sqrt(scale / total_noise_pwr)
    if csi_st['Ntx'] == 2:
        ret = ret * math.sqrt(2)
    elif csi_st['Ntx'] == 3:
        ret = ret * math.sqrt(dbinv(4.5))
    return ret



'''
参考csiread的相位校准函数
phase为未校准相位，k为子载波序列数组
函数首先将phase和k转换为数组。定义了两个切片slice1和slice2，这两个切片用于从phase中提取特定的部分。
计算斜率a，它表示相位的变化率。计算偏移b，它是相位的平均值。
使用斜率和偏移来校准相位。
返回校准后的相位。
'''
def calib_csiread(phase, k, axis=1):
    """Phase calibration

    Args:
        phase (ndarray): Unwrapped phase of CSI.
        k (ndarray): Subcarrier index
        axis (int): Axis along which is subcarrier. Default: 1

    Returns:
        ndarray: Phase calibrated

    Examples:

        >>> csi = csidata.csi[:10]
        >>> phase = np.unwrap(np.angle(csi), axis=1)
        >>> phase = calib(phase, k=scidx(20, 2), axis=1)

    References:
        1. `Enabling Contactless Detection of Moving Humans with Dynamic Speeds
        Using CSI <#>`_
    """
    p = np.asarray(phase)
    k = np.asarray(k)

    slice1 = [slice(None, None)] * p.ndim
    slice1[axis] = slice(-1, None)
    slice1 = tuple(slice1)
    slice2 = [slice(None, None)] * p.ndim
    slice2[axis] = slice(None, 1)
    slice2 = tuple(slice2)
    shape1 = [1] * p.ndim
    shape1[axis] = k.shape[0]
    shape1 = tuple(shape1)

    k_n, k_1 = k[-1], k[0]
    a = (p[slice1] - p[slice2]) / (k_n - k_1)
    b = p.mean(axis=axis, keepdims=True)
    k = k.reshape(shape1)

    phase_calib = p - a * k - b
    return phase_calib
    
'''
改写了calib函数，使其能够对相位进行处理
'''
def calib(phase, k, axis=-1):
    """Phase calibration

    Args:
        phase (ndarray): Unwrapped phase of CSI.
        k (ndarray): Subcarrier index
        axis (int): Axis along which is subcarrier. Default: -1 (last axis)

    Returns:
        ndarray: Phase calibrated

    Examples:

        >>> csi = csidata.csi[:10]
        >>> phase = np.unwrap(np.angle(csi), axis=-1)
        >>> phase = calib(phase, k=scidx(20, 2), axis=-1)

    References:
        1. `Enabling Contactless Detection of Moving Humans with Dynamic Speeds
        Using CSI <#>`_
    """
    p = np.asarray(phase)
    k = np.asarray(k)

    slice1 = [slice(None, None)] * p.ndim
    slice1[axis] = slice(-1, None)
    slice1 = tuple(slice1)
    slice2 = [slice(None, None)] * p.ndim
    slice2[axis] = slice(None, 1)
    slice2 = tuple(slice2)
    shape1 = [1] * p.ndim
    shape1[axis] = k.shape[0]
    shape1 = tuple(shape1)

    k_n, k_1 = k[-1], k[0]
    a = (p[slice1] - p[slice2]) / (k_n - k_1)
    b = p.mean(axis=axis, keepdims=True)
    k = k.reshape(shape1)

    phase_calib = p - a * k - b
    return phase_calib


