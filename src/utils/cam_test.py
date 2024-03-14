import cv2


def find_and_show_camera():
    max_tested = 10
    available_cams = []
    unavailable_cams = []

    for i in range(max_tested):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print("Found camera at index:", i)
            print("Width: ", width, "Height: ", height)

            # Show the camera feed in a window
            ret, frame = cap.read()
            if ret:
                cv2.imshow('Camera Test', frame)
                cv2.waitKey(3000)  # Display the frame for 3 seconds
                cv2.destroyAllWindows()

            cap.release()
            available_cams.append(i)
        else:
            unavailable_cams.append(i)

    # After testing all cameras, print which IDs are available and which are not
    print("Available camera IDs:", available_cams)
    print("Unavailable camera IDs:", unavailable_cams)


find_and_show_camera()
