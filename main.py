import cv2
import mediapipe as mp

# Mediapipe Configuration
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Start video capture
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Não foi possível abrir a câmera")
            break

        # Convert image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process image using Mediapipe
        results = hands.process(image)

        # Draw hand detection on image
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display image on screen
        cv2.imshow("Hand Detector", image)

        # Exit loop if 'q' key is pressed
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
