import streamlit as st
from ultralytics import YOLO
import cv2
import socket

st.title("YOLOv8 Object Detection with Streamlit")

model = YOLO("yolov8n.pt")

def yolo_inference():
    cap = cv2.VideoCapture(0) 
    stframe = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image")
            break

        results = model.predict(source=frame)

        annotated_frame = results[0].plot() 
        stframe.image(annotated_frame, channels="BGR")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def get_network_host():
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0)
        try:
            s.connect(('8.8.8.8', 1))
            network_ip = s.getsockname()[0]
        except Exception:
            network_ip = local_ip
        finally:
            s.close()
    except Exception as e:
        network_ip = 'Unable to retrieve network IP'
        st.error(f"Error retrieving network IP: {e}")
    return local_ip, network_ip

local_ip, network_ip = get_network_host()
st.write(f"Local URL: http://{local_ip}:8501")
st.write(f"Network URL: http://{network_ip}:8501")

if st.button("Start Object Detection"):
    yolo_inference()
