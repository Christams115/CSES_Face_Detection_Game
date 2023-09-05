import cv2
import numpy as np
import random
import time
import threading
from queue import Queue
from PIL import Image, ImageSequence

# Convert GIF to a list of frames
img = Image.open("CSES_Green_Shark.png")
frames = [cv2.cvtColor(np.array(frame), cv2.COLOR_RGBA2BGRA) for frame in ImageSequence.Iterator(img)]

def main():
    cap = cv2.VideoCapture(0)
    video = cv2.VideoCapture('Underwater World Background Video Animation _  Motion Background Loop _ No Copyright.mp4')

    if not cap.isOpened():
        print("Could not open video device.")
        return

    if not video.isOpened():
        print("Could not open background video.")
        return

    spots = []
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    start_time = None
    speed = 20
    dots_per_second = 10
    frame_rate = 30
    chance_per_frame = dots_per_second / frame_rate
    shark_frame_counter = 0
    game_over = False
    user_input = None
    show_live_feed = False

    # Global variable to communicate between threads
    user_input = None

    input_queue = Queue()
    def get_input(input_queue):
        user_input = input("Press 'y' to start the game or 'q' to quit: ")
        input_queue.put(user_input)

    while True:
        ret, frame = cap.read()
        video_ret, video_frame = video.read()
        frame = cv2.flip(frame, 1)

        if not ret or not video_ret:
            print("Failed to capture image")
            break

        if game_over:
            user_input = input("Press 'y' to restart the game or 'q' to quit or 'v' to show a live video: ")
            if user_input == 'y':
                game_over = False
                start_time = None
                spots.clear()
                show_live_feed = False
            elif user_input == 'q':
                break
            elif user_input == 'v':
                show_live_feed = True

        if show_live_feed:
            live_feed = cv2.VideoCapture(0)
            input_thread = threading.Thread(target=get_input, args=(input_queue,))
            input_thread.daemon = True
            input_thread.start()

            while show_live_feed:
                # Capture frame-by-frame
                ret, frame = live_feed.read()

                # Display the resulting frame
                cv2.imshow('Live Feed', frame)

                # Check the queue for user_input and break the loop if needed
                if not input_queue.empty():
                    user_input = input_queue.get()
                    if user_input == 'y':
                        print("Type q for the Next Game")
                        game_over = False
                        start_time = None
                        spots.clear()
                        show_live_feed = False
                        main()
                    elif user_input == 'q':
                        break

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        if user_input != 'y':
            cv2.imshow('Game', frame)
            user_input = input("Press 'y' to start the game or 'q' to quit: ")
            if user_input == 'q':
                break
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        video_frame = cv2.resize(video_frame, (frame.shape[1], frame.shape[0]))

        if start_time is None:
            start_time = time.time()

        elapsed_time = time.time() - start_time
        cv2.putText(video_frame, f"Time: {elapsed_time:.2f} seconds", (frame.shape[1] - 300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        largest_face = None
        largest_area = 0
        for (x, y, w, h) in faces:
            area = w * h
            if area > largest_area:
                largest_area = area
                largest_face = (x, y, w, h)

        if largest_face:
            x, y, w, h = largest_face
            video_frame[y:y+h, x:x+w] = frame[y:y+h, x:x+w]

        if random.random() < chance_per_frame:
            new_x = random.randint(0, frame.shape[1] - 1)
            spots.append([new_x, 0])

        surviving_spots = []

        for x, y in spots:
            shark_img = frames[shark_frame_counter]
            shark_img = cv2.resize(shark_img, (100, 150))
            shark_x, shark_y = x - shark_img.shape[1] // 2, y - shark_img.shape[0] // 2

            if 0 <= shark_y < frame.shape[0] - shark_img.shape[0] and 0 <= shark_x < frame.shape[1] - shark_img.shape[1]:
                alpha_s = shark_img[:, :, 3] / 255.0
                alpha_l = 1.0 - alpha_s
                for c in range(0, 3):
                    video_frame[shark_y:shark_y + shark_img.shape[0], shark_x:shark_x + shark_img.shape[1], c] = (
                        video_frame[shark_y:shark_y + shark_img.shape[0], shark_x:shark_x + shark_img.shape[1], c] * alpha_l +
                        shark_img[:, :, c] * alpha_s
                    )

            if largest_face:
                lx, ly, lw, lh = largest_face
                if lx <= x <= lx + lw and ly <= y <= ly + lh:
                    print(f"Game Over! Score: {elapsed_time:.2f} seconds")
                    game_over = True
                    start_time = None
                    spots.clear()
                    break

            y += speed
            if y < frame.shape[0]:
                surviving_spots.append([x, y])

        spots = surviving_spots
        shark_frame_counter = (shark_frame_counter + 1) % len(frames)

        cv2.imshow('Game', video_frame)

        if video.get(cv2.CAP_PROP_POS_FRAMES) == video.get(cv2.CAP_PROP_FRAME_COUNT):
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
