def get_volts(data):
    '''
    函数作用：获取rawdata，即原始电压值
    输入：通过串口所获取到的原始hex字符串
    输出：原始电压数据
    '''
    string_512 = 'AAAA048002' #通过特定的字符检索小包的位置
    volts = []
    for i in range(len(data)):
        if data[i:i+10] == string_512:
            high = data[i+10:i+12]
            low = data[i+12:i+14]
            check = data[i+14:i+16]
            if high == '' or low == '' or check == '':
                break
            sum_ = (int(hex(int('0x80',16)+int('0x02',16)+int(high,16)+int(low,16)),16)^0xFFFFFFFF)&0xFF
            if int(check,16) == sum_:
                rawdata = (int(high,16)<<8)|int(low,16)
                if rawdata > 32768:
                    rawdata -=65536
                volt = (rawdata * (1.8/4096)) / 2000
                volts.append(volt)
            else:
                volts.append(volts[-1])
    return volts


def denoise(rawdata):
    '''
    函数作用：通过小波变换来去除噪声
    输入：原始电压数据
    输出：去噪后的电压数据
    '''
    index = []
    data = []
    for i in range(len(rawdata)-1):
        X = float(i)
        Y = float(rawdata[i])
        index.append(X)
        data.append(Y)

    w = pywt.Wavelet('db8')  # 选用Daubechies8小波
    maxlev = pywt.dwt_max_level(len(data), w.dec_len)
    threshold = 0.04 
    coeffs = pywt.wavedec(data, 'db8', level=maxlev) #将信号进行小波分解
    
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i])) #将噪声滤波

    datarec = pywt.waverec(coeffs, 'db8') #将信号进行小波重构
    return datarec


def fda(x_1,Fstop1,Fstop2,fs):
    '''
    函数作用：带通滤波
    输入：电压信号，下截止频率，上截止频率，采样频率
    输出：带通滤波后的信号
    '''
    b, a = signal.butter(8, [2.0*Fstop1/fs,2.0*Fstop2/fs], 'bandpass')
    filtedData = signal.filtfilt(b,a,x_1)
    return filtedData


def normal_and_downsample(datarec,dw_ratio=8):
    '''
    函数作用：将去噪完的数据进行归一化和降采样处理
    输入：去噪后的数据，降采样率
    输出：归一化和降采样后的数据
    '''
    #归一化
    max_ = max(datarec)
    min_ = min(datarec)
    datarec = (datarec - min_) / (max_ - min_)
    
    #降采样
    data = []
    count, temp = 0, 0
    for i in range(len(datarec)):
        temp += datarec[i]
        count += 1
        if count == dw_ratio:
            data.append(temp / dw_ratio)
            temp = 0
            count = 0
            
    #将眨眼所产生的肌肉点信号用序列均值代替
    mean_ = np.mean(data)
    for i in range(len(data)):
        if data[i] >= mean_+0.12 or data[i] <= mean_-0.12:
            data[i] = mean_
    
    return data


def get_class_data(rawdata,dw_ratio=8):
    '''
    函数作用：获取到分类完成的数据
    输入：去噪之后的电压数据、原数据中所含的类别数
    输出：分类数据
    '''
    data_dict = {}
    data = []
    data[:] = rawdata[3584:5632]
    data = fda(data,10,30,180) #带通滤波器获取10-30hz的信号
    data = normal_and_downsample(data,dw_ratio)
    
    return data


def get_data_matrix_from_raw_hex_str(path,dw_ratio=8
                                     ,save=False,save_root=False,save_len=640):
    files = os.listdir(path) #得到文件夹下的所有文件名称
    rawdata_path = []
    data_matrix = np.zeros((len(files),int(4*512/dw_ratio)+1))
    for i in range(len(files)):
        rawdata_path.append(path + '/' + files[i])
        
    for i in range(len(files)):
        with open(rawdata_path[i], "r") as f:  # 打开文件
            data_original = f.read()  # 读取文件
            
        class_ = int(files[i][7])
                
        volts = get_volts(data_original) #通过hexstr获取电压值数据
        data = denoise(volts) #小波变换去噪
        data = get_class_data(data,dw_ratio) #获取最终分好的数据
            
#         print(files[i],len(data))
        data_matrix[i,:-1] = data
        data_matrix[i,-1] = class_
            
        if save == True:   
            data_save = np.zeros((save_len,len(data_dict[str(j)])))
            for k in range(save_len):
                data_save[k,:] = data_dict[str(j)]
            save_path = save_root + '/' + str(int(j)) + '_' + str(i*2+j) + '.txt'
            np.savetxt(save_path,data_save)
            
    return data_matrix