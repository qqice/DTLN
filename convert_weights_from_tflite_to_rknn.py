from rknn.api import RKNN

TFLITE_MODEL = './pretrained_model/model_1.tflite'

RKNN_MODEL = './pretrained_model/model_1.rknn'

QUANTIZE_ON = False

if __name__ == '__main__':
    rknn = RKNN(verbose=True)

    print('--> Config model')
    rknn.config(target_platform='rk3588')
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_tflite(model=TFLITE_MODEL,input_is_nchw=True)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=QUANTIZE_ON)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    print('--> Accuracy analysis')
    ret = rknn.accuracy_analysis(inputs=['input_1.npy','input_2.npy'],target='rk3588',device_id="192.168.1.203:5555")
    if ret != 0:
        print('acc_analysis failed!')
        exit(ret)
    print('done')
    # Export RKNN model
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    # ret = rknn.init_runtime('rk3566')
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    rknn.release()
