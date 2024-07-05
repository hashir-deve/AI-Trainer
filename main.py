import math
import matplotlib.pyplot as plt
import tensorflow as tf
from fastapi import FastAPI, WebSocket
import uvicorn
import cv2, base64
import numpy as np
import asyncio
from fastapi.middleware.cors import CORSMiddleware
import io
import os
import tempfile

interpreter=tf.lite.Interpreter(model_path='single-pose-lightning.tflite')
interpreter.allocate_tensors()

EDGES={
    (0,1):'m',
    (0,2):'c',
    (1,3):'m',
    (2,4):'c',
    (0,5):'m',
    (0,6):'c',
    (5,7):'m',
    (7,9):'m',
    (6,8):'c',
    (8,10):'c',
    (5,6):'y',
    (5,11):'m',
    (6,12):'c',
    (11,12):'y',
    (11,13):'m',
    (13,15):'m',
    (12,14):'c',
    (14,16):'c', 
}

def draw_keypoints(frame,keypoints,confidence_threshold):
    y,x,c=frame.shape
    shaped=np.squeeze(np.multiply(keypoints,[y,x,1])) #re coverting from 192x192 --> 480x640  [y,x,1] 1 bsc want color chanel as it is
    
    for kp in shaped:
        ky,kx,kp_conf=kp
        if kp_conf>confidence_threshold: #selecting only part displayed in front of camera obviously part not shown will have some lesser value
            cv2.circle(frame,(int(kx),int(ky)),4,(0,255,0),-1) #4 is size then color then -1 to fill circle


def draw_connections(frame,keypoints,edges,confidence_threshold):
    y,x,c=frame.shape
    shaped=np.squeeze(np.multiply(keypoints,[y,x,1])) #re coverting from 192x192 --> 480x640  [y,x,1] 1 bsc want color chanel as it is
    
    for edge,color in edges.items():
        p1,p2=edge
        y1,x1,c1=shaped[p1]
        y2,x2,c2=shaped[p2]
        if (c1>confidence_threshold) & (c2>confidence_threshold): #selecting only part displayed in front of camera obviously part not shown will have some lesser value
            cv2.line(frame,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0),2) 


def find_angle(frame,keypoints,p1,p2,p3,confidence_threshold):
    y,x,c=frame.shape
    shaped=np.squeeze(np.multiply(keypoints,[y,x,1])) #re coverting from 192x192 --> 480x640  [y,x,1] 1 bsc want color chanel as it is
    y1,x1,c1=shaped[p1]
    y2,x2,c2=shaped[p2]
    y3,x3,c3=shaped[p3]
    
    angle=math.degrees(math.atan2(int(y3)-int(y2),int(x3)-int(x2))-math.atan2(int(y1)-int(y2),int(x1)-int(x2)))
    if angle<0:
        angle*=-1
    
    if (c1>confidence_threshold) & (c2>confidence_threshold) & (c3>confidence_threshold):
       # cv2.line(frame,(int(x1),int(y1)),(int(x2),int(y2)),(255,255,255),3)
        #cv2.line(frame,(int(x3),int(y3)),(int(x2),int(y2)),(255,255,255),3)
        cv2.circle(frame,(int(x1),int(y1)),10,(0,0,255),cv2.FILLED)
        cv2.circle(frame,(int(x1),int(y1)),15,(0,0,255),2)
        cv2.circle(frame,(int(x2),int(y2)),10,(0,0,255),cv2.FILLED)
        cv2.circle(frame,(int(x2),int(y2)),15,(0,0,255),2)                           
        cv2.circle(frame,(int(x3),int(y3)),10,(0,0,255),cv2.FILLED)
        cv2.circle(frame,(int(x3),int(y3)),15,(0,0,255),2)
        cv2.putText(frame,str(int(angle)),(int(x2)-50,int(y2)+50),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)  
        
    return angle    


def squat_reps_counting(frame,rft_hip_angle,lft_hip_angle,stage,reps):
    if lft_hip_angle>=170 and rft_hip_angle>=170:
        stage="up"
    elif (lft_hip_angle<=160 and rft_hip_angle<=160) and stage=="up":
        stage="down"
        reps+=1
    
    cv2.rectangle(frame,(0,0),(225,73),(245,117,16),-1)
    cv2.putText(frame,"REPS",(15,20),cv2.FONT_HERSHEY_PLAIN,2,(0,0,0),1,cv2.LINE_AA)  
    cv2.putText(frame,str(reps),(12,50),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(frame,stage,(12,70),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),1,cv2.LINE_AA)
    
    return stage,reps


def squat_exc(frame,keypoints_with_scores,stage,reps):
    
#     left_arm_angle=find_angle(frame,keypoints_with_scores,5,7,9,0.4)
#     right_arm_angle=find_angle(frame,keypoints_with_scores,6,8,10,0.4)
    left_hip_angle=find_angle(frame,keypoints_with_scores,5,11,13,0.4)
    right_hip_angle=find_angle(frame,keypoints_with_scores,6,12,14,0.4)
    
    stage,reps=squat_reps_counting(frame,right_hip_angle,left_hip_angle,stage,reps)
    
    
    return stage,reps


def arm_raise_reps_counting(frame,rft_arm_angle,lft_arm_angle,rft_hip_angle,lft_hip_angle,stage,reps):
    if (lft_arm_angle > 140 and rft_arm_angle>140) and (lft_hip_angle>=10 and rft_hip_angle>=10):
        stage="up"
    elif (lft_arm_angle > 160 and rft_arm_angle>160) and (lft_hip_angle<=10 and rft_hip_angle<=10) and stage=="up":
        stage="down"
        reps+=1
    
    cv2.rectangle(frame,(0,0),(225,73),(245,117,16),-1)
    cv2.putText(frame,"REPS",(15,20),cv2.FONT_HERSHEY_PLAIN,2,(0,0,0),1,cv2.LINE_AA)  
    cv2.putText(frame,str(reps),(12,50),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(frame,stage,(12,70),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),1,cv2.LINE_AA)
    
    return stage,reps


def arm_raise_exc(frame,keypoints_with_scores,stage,reps):
    
    left_arm_angle=find_angle(frame,keypoints_with_scores,5,7,9,0.4)
    right_arm_angle=find_angle(frame,keypoints_with_scores,6,8,10,0.4)
    left_hip_angle=find_angle(frame,keypoints_with_scores,5,11,7,0.4)
    right_hip_angle=find_angle(frame,keypoints_with_scores,6,12,8,0.4)
    
    stage,reps=arm_raise_reps_counting(frame,right_arm_angle,left_arm_angle,right_hip_angle,left_hip_angle,stage,reps)
    
    
    return stage,reps


def front_raise_reps_counting(frame,rft_angle,lft_angle,stage,reps):
    if lft_angle > 160 and rft_angle>160:
        stage="down"
    elif (lft_angle < 30 and rft_angle < 30) and stage=="down":
        stage="up"
        reps+=1
    
    cv2.rectangle(frame,(0,0),(225,73),(245,117,16),-1)
    cv2.putText(frame,"REPS",(15,20),cv2.FONT_HERSHEY_PLAIN,2,(0,0,0),1,cv2.LINE_AA)  
    cv2.putText(frame,str(reps),(12,50),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(frame,stage,(12,70),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),1,cv2.LINE_AA)
    
    return stage,reps


def front_raise_exc(frame,keypoints_with_scores,stage,reps):
    
    left_arm_angle=find_angle(frame,keypoints_with_scores,5,7,9,0.4)
    right_arm_angle=find_angle(frame,keypoints_with_scores,6,8,10,0.4)
    left_hip_angle=find_angle(frame,keypoints_with_scores,5,11,7,0.4)
    right_hip_angle=find_angle(frame,keypoints_with_scores,6,12,8,0.4)
    
    stage,reps=front_raise_reps_counting(frame,right_arm_angle,left_arm_angle,stage,reps)
    
    
    return stage,reps


def pushups_reps_counting(frame,rft_angle,lft_angle,stage,reps):
    if lft_angle > 160 and rft_angle>160:
        stage="up"
    if (lft_angle <= 65 and rft_angle <= 65) and stage=="up":
        stage="down"
        reps+=1
    
    cv2.rectangle(frame,(0,0),(225,73),(245,117,16),-1)
    cv2.putText(frame,"REPS",(15,20),cv2.FONT_HERSHEY_PLAIN,2,(0,0,0),1,cv2.LINE_AA)  
    cv2.putText(frame,str(reps),(12,50),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(frame,stage,(12,70),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),1,cv2.LINE_AA)
    
    return stage,reps


def pushups_exc(frame,keypoints_with_scores,stage,reps):
    
    left_angle=0
    left_angle=find_angle(frame,keypoints_with_scores,5,7,9,0.4)
    right_angle=find_angle(frame,keypoints_with_scores,6,8,10,0.4)
    
    stage,reps=pushups_reps_counting(frame,right_angle,left_angle,stage,reps)
    
    return stage,reps


def convert_yuv420_to_rgb(yuv_data):
    width, height = 720, 480  # Set these to the actual resolution of the camera
    y_size = width * height
    uv_size = (width // 2) * (height // 2)

    # Ensure the length matches the expected YUV420 size
    if len(yuv_data) != y_size + 2 * uv_size:
        raise ValueError("Invalid YUV420 data length")

    y = np.frombuffer(yuv_data[:y_size], dtype=np.uint8).reshape((height, width))
    u = np.frombuffer(yuv_data[y_size:y_size + uv_size], dtype=np.uint8).reshape((height // 2, width // 2))
    v = np.frombuffer(yuv_data[y_size + uv_size:], dtype=np.uint8).reshape((height // 2, width // 2))

    # Resize U and V planes to match the size of the Y plane
    u = cv2.resize(u, (width, height), interpolation=cv2.INTER_LINEAR)
    v = cv2.resize(v, (width, height), interpolation=cv2.INTER_LINEAR)

    # Stack Y, U, and V planes along the third dimension
    yuv = np.dstack((y, u, v))
    rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)

    return rgb


def rotate_image_anticlockwise(image):
    # Rotate the image 90 degrees anti-clockwise
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)


app=FastAPI()

# Allow CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_video_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    try:
        while True:
            data = await websocket.receive_text()

            if data.startswith("set_exercise"):
                exercise_value = data.split("=")[1]
                exercise = exercise_value

            elif data == "start_video":
                # Receive the video file
                video_bytes = await websocket.receive_bytes()
                temp_video_file.write(video_bytes)
                temp_video_file.close()
                print('Video recieved')

                # Open the video using cv2.VideoCapture
                cap = cv2.VideoCapture(temp_video_file.name)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                out = cv2.VideoWriter(output_video_file.name, fourcc, fps, (width, height))

                reps=0
                stage=None

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    img = frame.copy()
                    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
                    input_img = tf.cast(img, dtype=tf.float32)  # [192,192,3] float32

                    # Input and output arrays
                    input_details = interpreter.get_input_details()
                    output_details = interpreter.get_output_details()

                    # Pass to tensor to get predictions
                    interpreter.set_tensor(input_details[0]['index'], np.array(input_img))
                    interpreter.invoke()
                    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

                    draw_connections(frame, keypoints_with_scores, EDGES, 0.4)
                    draw_keypoints(frame, keypoints_with_scores, 0.4)
                    if exercise=='front_raise':
                        stage,reps=front_raise_exc(frame,keypoints_with_scores,stage,reps)

                    elif exercise=='push_ups':
                        stage,reps=pushups_exc(frame,keypoints_with_scores,stage,reps)

                    elif exercise=='arm_raise':
                        stage,reps=arm_raise_exc(frame,keypoints_with_scores,stage,reps)

                    elif exercise=='squat':
                        stage,reps=squat_exc(frame,keypoints_with_scores,stage,reps)  

                    out.write(frame)

                cap.release()
                out.release()

                # Read the processed video file and send it to Flutter
                with open(output_video_file.name, 'rb') as f:
                    video_data = f.read()
                    await websocket.send_bytes(video_data)
                    print('Video sent')

    finally:
        os.remove(temp_video_file.name)
        os.remove(output_video_file.name)
        await websocket.close()
    
def run_server():
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    loop = asyncio.get_event_loop()
    if loop.is_running():
        # Run in the current running loop
        loop.create_task(server.serve())
    else:
        # Run normally if no loop is running
        loop.run_until_complete(server.serve())

if __name__ == "__main__":
    run_server()