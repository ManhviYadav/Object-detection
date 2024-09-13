from ultralytics import YOLO

# Load the model
model = YOLO("best.pt")


try:
    results = model.track(source=0, show=True)
    print(results)
except Exception as e:
    print(f"An error occurred during prediction: {e}")


