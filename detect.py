from ultralytics import YOLO

model = YOLO('yolov8n.pt')
source_img = 'https://ultralytics.com/images/bus.jpg'
results = model(source_img, save=True)
for result in results:
    result.save(filename='results.jpg')

print("Detection complete! Check 'results.jpg'.")