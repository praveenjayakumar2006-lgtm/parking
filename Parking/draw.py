import cv2
import pickle
import numpy as np

zoom_scale = 1.3
points = []  # Store polygon points
manual_mode = False
violation_mode = False
current_image_index = 0

# Image paths
image_paths = ['carParkPortrait3.png', 'carParkPortrait4.png']
img1 = cv2.imread(image_paths[0])
img2 = cv2.imread(image_paths[1])
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

# Load Saved Positions
posLists_parking = []
posLists_violation = []

for i in range(2):
    try:
        with open(f'CarParkPos_Parking_{i}', 'rb') as f:
            posLists_parking.append(pickle.load(f))
    except:
        posLists_parking.append([])

    try:
        with open(f'CarParkPos_Violation_{i}', 'rb') as f:
            posLists_violation.append(pickle.load(f))
    except:
        posLists_violation.append([])


def draw_polygon(event, x, y, flags, param):
    global points, manual_mode, violation_mode, img

    x = int(x / zoom_scale)
    y = int(y / zoom_scale)

    # Only allow manual or violation drawing
    if (manual_mode or violation_mode) and event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        mode_name = "Violation" if violation_mode else "Parking"
        print(f"{mode_name} Mode: Point {len(points)} selected: {x, y}")

        if len(points) == 4:
            if violation_mode:
                posLists_violation[current_image_index].append(points[:])
                with open(f'CarParkPos_Violation_{current_image_index}', 'wb') as f:
                    pickle.dump(posLists_violation[current_image_index], f)
                print(f"Added VIOLATION polygon: {points}")
            else:
                posLists_parking[current_image_index].append(points[:])
                with open(f'CarParkPos_Parking_{current_image_index}', 'wb') as f:
                    pickle.dump(posLists_parking[current_image_index], f)
                print(f"Added PARKING polygon: {points}")

            points.clear()  # Reset


while True:
    img = cv2.imread(image_paths[current_image_index])
    if current_image_index == 1:
        img = cv2.resize(img, (img1.shape[1], img1.shape[0]))

    # Draw Parking polygons (GREEN)
    for i, poly in enumerate(posLists_parking[current_image_index]):
        pts = np.array(poly, np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], (0, 255, 0))
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
        cv2.polylines(img, [pts], True, (0, 255, 0), 2)
        cx, cy = np.mean(pts, axis=0).astype(int)
        cv2.putText(img, f"B{i+1}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Draw Violation polygons (RED)
    for i, poly in enumerate(posLists_violation[current_image_index]):
        pts = np.array(poly, np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], (0, 0, 255))
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
        cv2.polylines(img, [pts], True, (0, 0, 255), 2)
        cx, cy = np.mean(pts, axis=0).astype(int)
        cv2.putText(img, f"V{i+1}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Draw ongoing polygon points
    for pt in points:
        cv2.circle(img, pt, 4, (255, 255, 0), -1)
    if manual_mode or violation_mode:
        mode_text = "Violation Mode" if violation_mode else "Parking Mode"
        cv2.putText(img, f"{mode_text}: Select {len(points)+1}/4 points", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        for j in range(len(points)-1):
            cv2.line(img, points[j], points[j+1], (255, 255, 0), 1)

    # Zoomed display
    zoomed_img = cv2.resize(img, None, fx=zoom_scale, fy=zoom_scale)
    cv2.imshow("Parking Zone Editor", zoomed_img)
    cv2.setMouseCallback("Parking Zone Editor", draw_polygon)

    key = cv2.waitKey(1)

    # Delete last polygon
    if key == ord('b'):
        if manual_mode and posLists_parking[current_image_index]:
            removed = posLists_parking[current_image_index].pop()
            with open(f'CarParkPos_Parking_{current_image_index}', 'wb') as f:
                pickle.dump(posLists_parking[current_image_index], f)
            print(f"Deleted PARKING polygon: {removed}")
        elif violation_mode and posLists_violation[current_image_index]:
            removed = posLists_violation[current_image_index].pop()
            with open(f'CarParkPos_Violation_{current_image_index}', 'wb') as f:
                pickle.dump(posLists_violation[current_image_index], f)
            print(f"Deleted VIOLATION polygon: {removed}")

    elif key == ord('c'):
        posLists_parking[current_image_index] = []
        posLists_violation[current_image_index] = []
        with open(f'CarParkPos_Parking_{current_image_index}', 'wb') as f:
            pickle.dump(posLists_parking[current_image_index], f)
        with open(f'CarParkPos_Violation_{current_image_index}', 'wb') as f:
            pickle.dump(posLists_violation[current_image_index], f)
        print("Cleared all polygons.")

    elif key == ord('m'):
        manual_mode = True
        violation_mode = False
        points.clear()
        print("Switched to PARKING mode.")

    elif key == ord('v'):
        violation_mode = True
        manual_mode = False
        points.clear()
        print("Switched to VIOLATION mode.")

    elif key == ord('s'):
        current_image_index = 1 - current_image_index
        points.clear()
        print(f"Switched to image: {image_paths[current_image_index]}")

    elif key == ord('q'):
        break

cv2.destroyAllWindows()