import depthai as dai
print(f"DepthAI version: {dai.__version__}")

with dai.Pipeline() as pipeline:
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    queue = cam.requestOutput((640, 480), dai.ImgFrame.Type.BGR888p).createOutputQueue()

    print("Pipeline started. Press Ctrl+C to stop.")
    while pipeline.isRunning():
        frame = queue.get()
        print(f"Got frame: {frame.getWidth()}x{frame.getHeight()} | ts: {frame.getTimestamp()}")