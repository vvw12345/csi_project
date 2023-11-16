# csi_project
    这个项目为CSI人体行为识别检测的代码，主要包括四个部分，数据读入->数据预处理->特征提取->行为识别；

## 当前的主要研究手段
    1.CSI幅值角度：发射信号的过程中，CSI的相位偏移是随机的。因此对于单一的天线来说，CSI的相位是呈随机分布的。但是幅值是较为稳定的。
    2.CSI相位差角度：“对于单一天线”的相位是随机分布，但是对于两个不同的天线，他们的相位差的相对稳定的。
    3.处理之后的CSI多普勒表征角度：体坐标速度剖面（Body-coordinate Velocity Profile,BVP）是一种域独立的特征，只和人体运动有关，和周围环境无关。

## 数据集的使用
    本项目采用了自采集数据集和Widar3.0的数据集，其原始数据集会进行预处理，而其预处理之后的数据也会用来做效果的对比。
    UT-HAR数据集：https://github.com/ermongroup/Wifi_Activity_Recognition
    github上的一个项目：https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark

## 数据读入部分