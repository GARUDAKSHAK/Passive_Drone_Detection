import adi, numpy as np

sdr = adi.ad9361("ip:192.168.3.1")
sdr.sample_rate = 10000000
sdr.rx_rf_bandwidth = 8000000
sdr.rx_buffer_size = 1024
sdr.rx_lo = 2440000000

sdr.rx_enabled_channels = [0, 1]
sdr.gain_control_mode_chan0 = "manual"
sdr.gain_control_mode_chan1 = "manual"
sdr.rx_hardwaregain_chan0 = 30
sdr.rx_hardwaregain_chan1 = 30

data = sdr.rx()
print("Type :", type(data))
print("Shape:", np.shape(data))