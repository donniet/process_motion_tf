from __future__ import print_function
import sys
import os
from argparse import ArgumentParser, SUPPRESS
import cv2
import numpy as np
import logging as log
from time import time
from time import sleep
from openvino.inference_engine import IENetwork, IECore
import signal
import sys
import requests

class SignalHandler:
    interrupted = False

    def __init__(self):
        signal.signal(signal.SIGINT, self._handler)

    def _handler(self, signal, frame):
        self.interrupted = True

def main(args):
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = IENetwork(model=model_xml, weights=model_bin)

    log.info("Loading Inference Engine")

    ie = IECore()

    log.info("Device info:")
    versions = ie.get_versions(args.device)
    print("{}{}".format(" "*8, args.device))
    print("{}MKLDNNPlugin version ......... {}.{}".format(" "*8, versions[args.device].major, versions[args.device].minor))
    print("{}Build ........... {}".format(" "*8, versions[args.device].build_number))
   
    if args.cpu_extension and "CPU" in args.device:
        ie.add_extension(args.cpu_extension, "CPU")
        log.info("CPU extension loaded: {}".format(args.cpu_extension))

    if "CPU" in args.device:
        supported_layers = ie.query_network(net, "CPU")
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(args.device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)

    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))

    n, w, h, c = net.inputs[input_blob].shape
    # net.inputs[input_blob].precision = 'U8'

    # input_blob.precision = 'I8'

    log.info('shape: [{},{},{},{}]'.format(n,w, h,c))

    exec_net = ie.load_network(network=net, device_name=args.device)

    handler = SignalHandler()

    print('model loaded, starting detection loop', flush=True)

    while not handler.interrupted:
        sleep(0.25)

        r = requests.get(args.motion_url, timeout=0.5)

        if not r:
            log.error('error getting motion from url, trying again after a minute')
            sleep(60)

        dat = np.frombuffer(r.content, dtype=np.int8)
        dat = np.reshape(dat, [1, w, h, 4])
        dat = dat[:,:,:,0:2]
        dat = dat.astype(np.float32)  

        res = exec_net.infer(inputs={input_blob: dat})
        res = res[out_blob]
        res = np.reshape(res, [])

        if res > args.total_motion:
            requests.post(args.post_url, data='"on"')
            print('motion detected: {}'.format(res), flush=True)
            sleep(600)
        
    print('exited.')

        # print(res)


    


    

if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)

    args = parser.add_argument_group("Options")
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help="display this message")
    args.add_argument('-m', '--model', help="Required. Path to a model .xml file")
    args.add_argument("-d", "--device",
        help="Optional. Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. Sample will look for a suitable plugin for device specified (CPU by default)",
        default="CPU", type=str)
    args.add_argument("-l", "--cpu_extension",
        help="Optional. Required for CPU custom layers. Absolute path to a shared library with the kernels implementations.",
        type=str, default=None)
    args.add_argument('-u', '--motion_url', default='http://mirror.local:8888/motion.bin', help='url to motion binary')
    args.add_argument('-p', '--post_url', default='http://mirror.local:9080/', help='url to post if motion detected')
    args.add_argument('-t', '--total_motion', default=4, help='total avg motion to consider for motion detection')


    main(parser.parse_args())
