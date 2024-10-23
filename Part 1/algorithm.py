#This model run first just to detect a person! If a person (driver) is detected, i.e "True", then it will send a signal to turn on the second model.
from ultralytics import YOLO
def is_person_detected(model, image_path):
    # Perform inference on the provided image
    results = model(image_path)
    
    # Access the first result (assuming a single image)
    first_result = results[0]
    
    # Access the boxes attribute
    detected_objects = first_result.boxes
    
    # Check if any people (class index 0) were detected
    person_detected = detected_objects.shape[0] > 0  # True if any boxes are detected
    
    return person_detected

def main():
    model = YOLO('Part 1\yolo8-trained.pt')  # Load your trained model
    image_path = 'Part 1\driver_in_a_car.png'  # This will be changed to your live image!
    person_detected = is_person_detected(model, image_path)
    return person_detected

if __name__ == "__main__":
    person_detected = main()
    print(f"Person detected: {person_detected}")  # This will print True or False