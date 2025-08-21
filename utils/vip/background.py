import sys
import vpi
import numpy as np
from argparse import ArgumentParser
import cv2

# ----------------------------
# Parse command line arguments

parser = ArgumentParser()
parser.add_argument('backend', choices=['cpu','cuda'],
                    help='Backend to be used for processing')

parser.add_argument('input',
                    help='Input video to be denoised')

args = parser.parse_args()

if args.backend == 'cuda':
    backend = vpi.Backend.CUDA
else:
    assert args.backend == 'cpu'
    backend = vpi.Backend.CPU

# -----------------------------
# Open input and output videos

inVideo = cv2.VideoCapture(args.input)

fourcc = cv2.VideoWriter_fourcc(*'MPEG')
inSize = (int(inVideo.get(cv2.CAP_PROP_FRAME_WIDTH)), int(inVideo.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fps = inVideo.get(cv2.CAP_PROP_FPS)

outVideoFGMask = cv2.VideoWriter('fgmask_python'+str(sys.version_info[0])+'_'+args.backend+'.mp4',
                                fourcc, fps, inSize)

outVideoBGImage = cv2.VideoWriter('bgimage_python'+str(sys.version_info[0])+'_'+args.backend+'.mp4',
                                fourcc, fps, inSize)

#--------------------------------------------------------------
# Create the Background Subtractor object using the backend specified by the user
with backend:
    bgsub = vpi.BackgroundSubtractor(inSize, vpi.Format.BGR8)

#--------------------------------------------------------------
# Main processing loop
idxFrame = 0
while True:
    print("Processing frame {}".format(idxFrame))
    idxFrame+=1

    # Read one input frame
    ret, cvFrame = inVideo.read()
    if not ret:
        break

    # Get the foreground mask and background image estimates
    fgmask, bgimage = bgsub(vpi.asimage(cvFrame, vpi.Format.BGR8), learnrate=0.01)

    # Mask needs to be converted to BGR8 for output
    fgmask = fgmask.convert(vpi.Format.BGR8, backend=vpi.Backend.CUDA)

    # Write images to output videos
    with fgmask.rlock_cpu(), bgimage.rlock_cpu():
        outVideoFGMask.write(fgmask.cpu())
        outVideoBGImage.write(bgimage.cpu())