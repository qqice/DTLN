import soundfile as sf
import numpy as np
import time
#import onnxruntime
from rknnlite.api import RKNNLite
#from rknn.api import RKNN

model_inputs_1 = [np.zeros([1,1,257],dtype=np.float32), np.zeros([1,2,128,2],dtype=np.float32)]
model_inputs_2 = [np.zeros([1,1,512],dtype=np.float32), np.zeros([1,2,128,2],dtype=np.float32)]
def init_rknn_model(model_path, target, device_id):
    # Create RKNN object
    rknn = RKNNLite()

    # Load RKNN model
    print('--> Loading model')
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        print('Load RKNN model \"{}\" failed!'.format(model_path))
        exit(ret)
    print('done')

    # init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    return rknn

##########################
# the values are fixed, if you need other values, you have to retrain.
# The sampling rate of 16k is also fix.
block_len = 512
block_shift = 128
# load models
interpreter_1 = init_rknn_model('./pretrained_model/model_1.rknn', 'rk3588', None)
# load models
interpreter_2 = init_rknn_model('./pretrained_model/model_2.rknn', 'rk3588', None)

# load audio file
audio,fs = sf.read('./audioset_realrec_airconditioner_2TE3LoA2OUQ.wav')
# check for sampling rate
if fs != 16000:
    raise ValueError('This model only supports 16k sampling rate.')
# preallocate output audio
out_file = np.zeros((len(audio)))
# create buffer
in_buffer = np.zeros((block_len)).astype('float32')
out_buffer = np.zeros((block_len)).astype('float32')
# calculate number of blocks
num_blocks = (audio.shape[0] - (block_len-block_shift)) // block_shift
# iterate over the number of blcoks  
time_array = []      
for idx in range(num_blocks):
    start_time = time.time()
    # shift values and write to buffer
    in_buffer[:-block_shift] = in_buffer[block_shift:]
    in_buffer[-block_shift:] = audio[idx*block_shift:(idx*block_shift)+block_shift]
    # calculate fft of input block
    in_block_fft = np.fft.rfft(in_buffer)
    in_mag = np.abs(in_block_fft)
    in_phase = np.angle(in_block_fft)
    # reshape magnitude to input dimensions
    in_mag = np.reshape(in_mag, (1,1,-1)).astype('float32')
    # set block to input
    model_inputs_1[0] = in_mag
    # run calculation 
    model_outputs_1 = interpreter_1.inference(inputs=model_inputs_1,data_format='NCHW')
    # get the output of the first block
    out_mask = model_outputs_1[0]
    # set out states back to input
    model_inputs_1[1] = model_outputs_1[1]  
    # calculate the ifft
    estimated_complex = in_mag * out_mask * np.exp(1j * in_phase)
    estimated_block = np.fft.irfft(estimated_complex)
    # reshape the time domain block
    estimated_block = np.reshape(estimated_block, (1,1,-1)).astype('float32')
    # set tensors to the second block
    # interpreter_2.set_tensor(input_details_1[1]['index'], states_2)
    model_inputs_2[0] = estimated_block
    # run calculation
    model_outputs_2 = interpreter_2.inference(inputs=model_inputs_2,data_format='NCHW')
    # get output
    out_block = model_outputs_2[0]
    # set out states back to input
    model_inputs_2[1] = model_outputs_2[1]
    # shift values and write to buffer
    out_buffer[:-block_shift] = out_buffer[block_shift:]
    out_buffer[-block_shift:] = np.zeros((block_shift))
    out_buffer  += np.squeeze(out_block)
    # write block to output file
    out_file[idx*block_shift:(idx*block_shift)+block_shift] = out_buffer[:block_shift]
    time_array.append(time.time()-start_time)
    
# write to .wav file 
sf.write('out.wav', out_file, fs) 
print('Processing Time [ms]:')
print(np.mean(np.stack(time_array))*1000)
print('Processing finished.')
