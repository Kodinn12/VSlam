& "C:/Users/aksha/anaconda3/envs/Vslam/python.exe" -c "
import depthai as dai
import time

pipeline = dai.Pipeline()
cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
queue = cam.requestOutput((640, 480), dai.ImgFrame.Type.BGR888p).createOutputQueue(maxSize=4, blocking=False)

pipeline.start()
print('Pipeline started')
time.sleep(2)

for i in range(30):
    frame = queue.get()
    if frame:
        print(f'Frame {i}: {frame.getWidth()}x{frame.getHeight()}')
    else:
        print(f'Frame {i}: None')
    time.sleep(0.1)

pipeline.stop()
