import cv2
import mediapipe as mp
import random
from collections import deque

# Constants for moves
ROCK, PAPER, SCISSORS = 0, 1, 2
MOVES = ['Rock', 'Paper', 'Scissors']

# Function to beat a given move
def beat_move(move):
    return (move + 1) % 3

# Function to lose to a given move
def lose_to_move(move):
    return (move - 1) % 3

class RPSBot:
    def __init__(self):
        self.history = deque(maxlen=15)
        self.probabilities = [[1/3 for _ in range(3)] for _ in range(3)]

    def update_probabilities(self, player_move):
        if self.history:
            last_player_move = self.history[-1]
            self.probabilities[last_player_move][player_move] += 0.1
            self.probabilities[last_player_move] = [max(min(p, 1), 0) for p in self.probabilities[last_player_move]]
            
        self.history.append(player_move)

    def predict(self):
        if not self.history:
            return random.choice([ROCK, PAPER, SCISSORS])
        
        last_move = self.history[-1]
        predicted_player_move = self.probabilities[last_move].index(max(self.probabilities[last_move]))
        return beat_move(predicted_player_move)

# Function to check if a finger is extended
def is_finger_extended(tip, dip, pip, mcp):
    # Check if the finger is not folded, indicating it's extended
    return tip.y < pip.y < mcp.y or tip.y < dip.y

# Function to classify the hand shape as rock, paper, or scissors
def classify_hand(hand_landmarks):
    # Get landmarks for thumb and fingers
    thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP]

    index_dip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_DIP]
    middle_dip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_DIP]
    ring_dip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_DIP]
    pinky_dip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_DIP]

    index_pip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP]
    middle_pip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_pip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_PIP]
    pinky_pip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_PIP]

    index_mcp = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP]
    middle_mcp = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_mcp = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_MCP]
    pinky_mcp = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_MCP]

    # Check if each finger is extended
    thumb_extended = thumb_tip.x < index_mcp.x  # For thumb, check horizontal position
    index_extended = is_finger_extended(index_tip, index_dip, index_pip, index_mcp)
    middle_extended = is_finger_extended(middle_tip, middle_dip, middle_pip, middle_mcp)
    ring_extended = is_finger_extended(ring_tip, ring_dip, ring_pip, ring_mcp)
    pinky_extended = is_finger_extended(pinky_tip, pinky_dip, pinky_pip, pinky_mcp)

    # Classify the hand shape based on extended fingers
    if not index_extended and not middle_extended and not ring_extended and not pinky_extended:
        return 'Rock'
    elif all([index_extended, middle_extended, ring_extended, pinky_extended]):
        return 'Paper'
    elif index_extended and middle_extended and not ring_extended and not pinky_extended:
        return 'Scissors'

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Create RPS bot instance
bot = RPSBot()

# Capture video from the first webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            hand_shape = classify_hand(hand_landmarks)
            player_move = MOVES.index(hand_shape) if hand_shape in MOVES else None
            if player_move is not None:
                bot_move = bot.predict()
                bot.update_probabilities(player_move)
                cv2.putText(image, f"Bot chooses: {MOVES[bot_move]}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, hand_shape, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Rock Paper Scissors', image)
    if cv2.waitKey(5) & 0xFF == 27 or cv2.getWindowProperty('Rock Paper Scissors', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
