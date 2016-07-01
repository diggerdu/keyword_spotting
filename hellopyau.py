# -*- coding: utf-8 -*-
import wave
import pylab as pl
import numpy as np

# 打开WAV文档
f = wave.open("swk.wav", "rb")

# 读取格式信息
# (nchannels, sampwidth, framerate, nframes, comptype, compname)
params = f.getparams()
nchannels, sampwidth, framerate, nframes = params[:4]

print framerate
# 读取波形数据
str_data = f.readframes(nframes)
f.close()

#将波形数据转换为数组
wave_data = np.fromstring(str_data, dtype=np.short)
wave_data.shape = -1, 1
wave_data = wave_data.T
time = np.arange(0, nframes/2) * (1.0 / framerate)

#fast fourier tansform
rate = 16000
fs = np.fft.fft(wave_data, rate)
fs = abs(fs)[0][1:rate/2+1]
feq = np.arange(1,rate/2+1)
# 绘制波形
pl.subplot(211) 
pl.plot(feq,fs)
#pl.plot(time, wave_data[0][:nframes/2]) 
pl.xlabel("time (seconds)")
pl.show()
