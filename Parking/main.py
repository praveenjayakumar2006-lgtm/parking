import cv2
import pickle
import numpy as np
import time
import os
from datetime import datetime
import json
from datetime import timedelta
from datetime import timezone
import threading
from reservation import handle_reservation_snapshots
import subprocess

# Start number_plate.py in background
subprocess.Popen(
    ["python3", "/Users/mithunravi/Documents/Number_Plate/number_plate.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)



parking_active = False
view_mode = 0  # 0 = video, 1 = parking only, 2 = violation only
last_reserved_check = time.time()

violation_count = 0  # ✅ Prevents "NameError" for violation count

def show_popup(img, slot_id, data):
    """Draws a small popup showing reservation details and a close (X) button."""
    global cross_coords

    # Prepare popup text lines
    info = [
    f"Slot: {slot_id}",
    f"Name: {data.get('userName', 'N/A')}",
    f"Email: {data.get('email', 'N/A')}",
    f"Plate: {data.get('vehiclePlate', 'N/A')}",
    f"Start: {data.get('startTime', 'N/A')}",
    f"End: {data.get('endTime', 'N/A')}",
]


    # Popup box position
    x, y, w, h = 1345, 301, 355, 185

    # Draw background rectangle
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)

    # Draw close ("X") button at top right corner
    cross_size = 20
    cross_x1 = x + w - cross_size - 5
    cross_y1 = y + 5
    cross_x2 = cross_x1 + cross_size
    cross_y2 = cross_y1 + cross_size
    cross_coords = (cross_x1, cross_y1, cross_x2, cross_y2)

    cv2.rectangle(img, (cross_x1, cross_y1), (cross_x2, cross_y2), (0, 0, 0), 2)
    cv2.putText(img, "X", (cross_x1 + 4, cross_y2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Draw text lines
    y_offset = 35
    for line in info:
        cv2.putText(img, line, (x + 10, y + y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        y_offset += 25


JSON_PATH = "/Users/mithunravi/MP/data/User_Reservations.json"
RESERVED_TXT_PATH = "reserved.txt"
selected_slot = None
reservation_data = []
cross_coords = None  # stores the (x1, y1, x2, y2) of the cross button
popup_visible = False



def update_reserved_slots():
    global reservation_data
    last_data = None

    while True:
        try:
            if os.path.exists(JSON_PATH):
                with open(JSON_PATH, "r") as file:
                    data = json.load(file)
                    reservation_data = data  # store all reservations

                # 🔹 Get current time in UTC (since JSON uses .000Z)
                now = datetime.utcnow()

                # 🔹 Filter only active reservations (start <= now <= end)
                active_slots = []
                for item in data:
                    try:
                        start = datetime.strptime(item["startTime"], "%Y-%m-%dT%H:%M:%S.%fZ")
                        end = datetime.strptime(item["endTime"], "%Y-%m-%dT%H:%M:%S.%fZ")
                        if start <= now <= end:
                            active_slots.append(item["slotId"])
                    except Exception as t_err:
                        print(f"⚠️ Time parse error for {item.get('slotId')}: {t_err}")

                # 🔹 Update reserved.txt only when changed
                if active_slots != last_data:
                    with open(RESERVED_TXT_PATH, "w") as file:
                        if active_slots:
                            file.write(str(active_slots))
                        else:
                            file.write("[]")

                    last_data = active_slots

            else:
                reservation_data = []
                with open(RESERVED_TXT_PATH, "w") as f:
                    f.write("[]")

        except Exception as e:
            print(f"⚠️ Reservation monitor error: {e}")

        time.sleep(1)


# Start background monitoring thread
thread = threading.Thread(target=update_reserved_slots, daemon=True)
thread.start()


# --- Load Reserved Slots ---
def load_reserved_slots(filename="reserved.txt"):
    if not os.path.exists(filename):
        return []
    try:
        with open(filename, "r") as f:
            content = f.read().strip()
            content = content.replace("[", "").replace("]", "").replace("'", "").replace("\"", "")
            slots = [x.strip().upper() for x in content.split(",") if x.strip()]
            return slots
    except:
        return []

reserved_slots = load_reserved_slots()

# --- Video & Reference Images ---
cap = cv2.VideoCapture('carPark2.mp4')
img_portrait1 = cv2.imread("carParkPortrait3.png")
img_portrait2 = cv2.imread("carParkPortrait4.png")
if img_portrait1 is None or img_portrait2 is None:
    raise Exception("Reference images not found!")

output_size = (img_portrait1.shape[1], img_portrait1.shape[0])
img_portrait2 = cv2.resize(img_portrait2, output_size)

# --- Load Slot Positions ---
def load_positions(name, count=2):
    pos_lists = []
    for i in range(count):
        try:
            with open(f'{name}_{i}', 'rb') as f:
                pos_lists.append(pickle.load(f))
        except:
            pos_lists.append([])
    return pos_lists

posList_parking_video = load_positions('CarParkPos_Parking')
posList_parking_image = load_positions('CarParkPos_Parking')
posList_violation_video = load_positions('CarParkPos_Violation')
posList_violation_image = load_positions('CarParkPos_Violation')

# --- Reference background ---
success, ref_frame = cap.read()
if not success or ref_frame is None:
    raise Exception("Failed to read video or empty frame!")
ref_frame = cv2.resize(ref_frame, output_size)
ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)

# --- State trackers ---
occupy_times = [0]*len(posList_parking_video[0])
snap_taken_parking = [False]*len(posList_parking_video[0])
# track previous reserved state so we can reset timers when reservation is removed
prev_reserved = [ (f"B{i+1}" in reserved_slots) for i in range(len(posList_parking_video[0])) ]
violation_coords_image = posList_violation_image[1] if len(posList_violation_image) > 1 else posList_violation_image[0]
occupy_times_violation = [0]*len(violation_coords_image)
violation_snap_map = {}

VIOLATION_SECONDS = 5
PARK_SECONDS = 5
save_folder = "violations"
os.makedirs(save_folder, exist_ok=True)


# --- Utility Functions ---
def crop_polygon(img, polygon):
    mask = np.zeros_like(img)
    pts = np.array(polygon, np.int32)
    cv2.fillPoly(mask, [pts], (255, 255, 255))
    cropped = cv2.bitwise_and(img, mask)
    x, y, w, h = cv2.boundingRect(pts)
    return cropped[y:y + h, x:x + w]


def detect_vehicle(img_gray, ref_gray, polygon, threshold, strong_checks=False):
    """
    Motion + brightness + edge + variance checks.
    Used for both normal parking and violation zones.
    """
    pts = np.array(polygon, np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(img_gray.shape[1], x + w), min(img_gray.shape[0], y + h)

    crop_gray = img_gray[y1:y2, x1:x2]
    crop_ref = ref_gray[y1:y2, x1:x2]

    if crop_gray.size == 0 or crop_ref.size == 0 or crop_gray.shape != crop_ref.shape:
        return False

    # Reduce noise
    crop_gray_blur = cv2.GaussianBlur(crop_gray, (5, 5), 0)
    crop_ref_blur = cv2.GaussianBlur(crop_ref, (5, 5), 0)

    # Motion difference
    motion_score = np.sum(cv2.absdiff(crop_gray_blur, crop_ref_blur))
    if motion_score <= threshold:
        return False

    # ✅ Basic mode
    if not strong_checks:
        return True

    # ✅ Stronger check mode (for violation zones)
    h_c, w_c = crop_gray_blur.shape[:2]
    cx1, cy1 = int(w_c * 0.25), int(h_c * 0.25)
    cx2, cy2 = int(w_c * 0.75), int(h_c * 0.75)
    center = crop_gray_blur[cy1:cy2, cx1:cx2]
    if center.size == 0:
        return False
    mean_center = np.mean(center)

    edges = cv2.Canny(crop_gray_blur, 50, 150)
    edge_count = cv2.countNonZero(edges)
    variance = np.var(crop_gray_blur)

    # Thresholds
    min_center_brightness = 40
    min_edge_count = 40
    min_variance = 200.0

    center_ok = mean_center > min_center_brightness
    edge_ok = edge_count > min_edge_count
    var_ok = variance > min_variance

    return center_ok and (edge_ok or var_ok)



# --- Get current parking status ---
def get_current_status(frame):
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    current_time = time.time()
    status = []
    for i, polygon in enumerate(posList_parking_video[0]):
        is_occupied = detect_vehicle(img_gray, ref_gray, polygon, threshold=80000)
        if is_occupied:
            if occupy_times[i] == 0:
                occupy_times[i] = current_time
            elapsed = current_time - occupy_times[i]
            timer_text = f"{int(elapsed//60):02}:{int(elapsed%60):02}"
            occupied = True
        else:
            occupy_times[i] = 0
            timer_text = ""
            occupied = False
        status.append({'occupied': occupied, 'timer': timer_text})
    return status

def draw_overlay(img, parking_coords, violation_coords, status, video_frame=None, is_portrait=False):
    global occupy_times_violation, violation_snap_map

    img_display = img.copy()

    # Counters
    green_count = 0
    blue_count = 0
    pink_count = 0

    # --- Parking zones ---
    for i, polygon in enumerate(parking_coords):
        data = status[i]
        is_occupied = data['occupied']
        slot_name = f"B{i+1}"

        # --- Reserved slot (Pink always) ---
        if slot_name in reserved_slots:
            color = (203, 192, 255)  # Pink
            pink_count += 1
        else:
            if is_occupied:
                color = (255, 0, 0)   # Blue (vehicle parked)
                blue_count += 1
            else:
                color = (0, 255, 0)   # Green (free)
                green_count += 1

        # Draw polygon
        pts = np.array(polygon, np.int32)
        overlay = img_display.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.3, img_display, 0.7, 0, img_display)
        cv2.polylines(img_display, [pts], True, color, 2)

        # Label slot
        top_left = polygon[0]
        cv2.putText(img_display, slot_name, (top_left[0]+5, top_left[1]+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        # Show "Occupied" or "Empty" for reserved slots
        if slot_name in reserved_slots:
            status_text = "Occupied" if is_occupied else "Empty"
            color_text = (0, 0, 255) if is_occupied else (0, 200, 0)  # Red or green

            # Find the center of the box
            pts = np.array(polygon, np.int32)
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))

            # Draw shadow + colored text
            cv2.putText(img_display, status_text, (cx - 45, cy + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img_display, status_text, (cx - 45, cy + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_text, 1, cv2.LINE_AA)


    # --- Violation zones ---
    for j, polygon in enumerate(violation_coords):
        color = (0, 200, 255)
        if occupy_times_violation[j] != 0:
            elapsed_seconds = int(time.time() - occupy_times_violation[j])
            if elapsed_seconds >= 5:
                color = (0, 0, 255)

        if video_frame is not None:
    # Use stronger checks for violation zones to avoid shadow / noise false positives
            in_violation = detect_vehicle(
                cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY),
                ref_gray,
                polygon,
                threshold=80000,
                strong_checks=True
            )
            if in_violation and occupy_times_violation[j] == 0:
                occupy_times_violation[j] = time.time()
            elif not in_violation:
                occupy_times_violation[j] = 0


        pts = np.array(polygon, np.int32)
        overlay = img_display.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.3, img_display, 0.7, 0, img_display)
        cv2.polylines(img_display, [pts], True, color, 2)
        top_left = polygon[0]
        cv2.putText(img_display, f"V{j+1}", (top_left[0]+5, top_left[1]+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

    return img_display, green_count, blue_count, pink_count

parking_active = False


# --- Main Loop ---
view_mode = 0  # 0 = video, 1 = parking only, 2 = violation only
last_reserved_check = time.time()

def mouse_callback(event, x, y, flags, param):
    global selected_slot, view_mode, popup_visible, cross_coords

    if event == cv2.EVENT_LBUTTONDOWN:
        # 🔹 Check if the popup close "X" button is clicked
        if cross_coords and popup_visible:
            x1, y1, x2, y2 = cross_coords
            if x1 <= x <= x2 and y1 <= y <= y2:
                popup_visible = False
                selected_slot = None
                cross_coords = None
                return

        # 🔹 Handle reserved slot selection (for view modes 0 and 1)
        if view_mode == 0:
            coords = posList_parking_video[0]
        elif view_mode == 1:
            coords = posList_parking_image[1] if len(posList_parking_image) > 1 else posList_parking_image[0]
        else:
            return  # No clicks in violation view

        for i, polygon in enumerate(coords):
            slot_name = f"B{i+1}"
            if slot_name in reserved_slots:
                pts = np.array(polygon, np.int32)
                if cv2.pointPolygonTest(pts, (x, y), False) >= 0:
                    selected_slot = slot_name
                    popup_visible = True
                    break



while True:
    success, frame = cap.read()
    if not success or frame is None:
        print("Video ended or frame not read.")
        break
    frame = cv2.resize(frame, output_size)
    current_status = get_current_status(frame)

    # --- Handle reservation snapshots (auto capture when reserved slot becomes occupied) ---
    handle_reservation_snapshots(posList_parking_video, posList_parking_image, img_portrait2, current_status, reserved_slots)



       # 🔁 Reload reserved slots every 1 second (live update) and reset timers if reservation removed
    if time.time() - last_reserved_check >= 1:
        new_reserved = load_reserved_slots()  # fresh list from reserved.txt

        # If a slot was reserved before and now is NOT reserved -> reset its timers
        for i in range(len(posList_parking_video[0])):
            slot_name = f"B{i+1}"
            was_reserved = prev_reserved[i]
            is_reserved = slot_name in new_reserved
            if was_reserved and not is_reserved:
                # reservation removed -> start countdown/occupancy detection fresh next time
                occupy_times[i] = 0
                snap_taken_parking[i] = False

            # update prev flag
            prev_reserved[i] = is_reserved

        reserved_slots = new_reserved
        last_reserved_check = time.time()

        
        

    parking_coords_image = posList_parking_image[1] if len(posList_parking_image) > 1 else posList_parking_image[0]
    violation_coords_image = posList_violation_image[1] if len(posList_violation_image) > 1 else posList_violation_image[0]

    display_frame = frame.copy()

    # --- Parking view ---
    if view_mode == 0:
        if view_mode == 0:
            display_frame, green_count, blue_count, pink_count = draw_overlay(
                frame.copy(),
                posList_parking_video[0],
                posList_violation_video[0],
                current_status,
                video_frame=frame
            )

            # 🔹 Recalculate counts fresh every second
            free = green_count
            occupied_count = blue_count
            reserved_count = pink_count

            total_slots = free + occupied_count + reserved_count
            if total_slots != len(posList_parking_video[0]):
                # Optional safety check — should always match
                print(f"⚠️ Slot count mismatch! Expected {len(posList_parking_video[0])}, got {total_slots}")

    # 🧠 No dependency on reservation or parking_active — always real-time counts


        cv2.putText(display_frame, f'Free: {free}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(display_frame, f'Occupied: {occupied_count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        cv2.putText(display_frame, f'Reserved: {reserved_count}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (203,192,255), 2)
        cv2.putText(display_frame, f'Violations: {violation_count}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    elif view_mode == 1:
        display_frame, green_count, blue_count, pink_count = draw_overlay(
            img_portrait2.copy(),
            parking_coords_image,
            [],
            current_status,
            video_frame=None,
            is_portrait=True
        )

        # Real-time counts directly from color tracking
        free = green_count
        occupied_count = blue_count
        reserved_count = pink_count

        cv2.putText(display_frame, f'Free: {free}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(display_frame, f'Occupied: {occupied_count}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        cv2.putText(display_frame, f'Reserved: {reserved_count}', (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (203,192,255), 2)
        




    elif view_mode == 2:
        display_frame = img_portrait2.copy()

        # Create a transparent overlay for filling
        overlay = display_frame.copy()

        for j, box in enumerate(posList_violation_video[1]):
            pts = np.array(box, np.int32)

            # Skip if violation tracking data isn't available
            if j >= len(occupy_times_violation):
                continue

            # Determine color and timer
            if occupy_times_violation[j] != 0:
                elapsed = time.time() - occupy_times_violation[j]
                if elapsed >= VIOLATION_SECONDS:
                    color = (0, 0, 255)   # Red (violation)
                    fill_color = (0, 0, 255, 80)
                    timer_text = "Violation"
                            # --- Save violation snapshot ---
                    if j not in violation_snap_map:
                        violation_snap_map[j] = True  # prevent duplicate snapshots
                        cropped_img = crop_polygon(img_portrait2, box)  # use portrait image for accurate crop


                        violation_dir = "/Users/mithunravi/Documents/Parking/violations"
                        os.makedirs(violation_dir, exist_ok=True)

                        # Save the cropped violation image
                        violation_path = os.path.join(violation_dir, f"violation_{j+1}.jpg")
                        cv2.imwrite(violation_path, cropped_img)
                        print(f"🚨 Violation snapshot saved: {violation_path}")


                else:
                    color = (0, 255, 255) # Yellow (monitored)
                    fill_color = (0, 255, 255, 80)
                    remaining = int(VIOLATION_SECONDS - elapsed)
                    timer_text = f"{remaining}s"
            else:
                color = (0, 255, 255)
                fill_color = (0, 255, 255, 80)
                timer_text = ""

            # Fill the box with transparent color
            cv2.fillPoly(overlay, [pts], fill_color[:3])

            # Draw the border
            cv2.polylines(display_frame, [pts], True, color, 2)

            # Label each violation region
            cv2.putText(display_frame, f"V{j+1}", (pts[0][0], pts[0][1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # 🕒 Draw timer text inside the box (centered)
            if timer_text:
                cx = int(np.mean(pts[:, 0]))
                cy = int(np.mean(pts[:, 1]))
                cv2.putText(display_frame, timer_text, (cx - 35, cy + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(display_frame, timer_text, (cx - 35, cy + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

        # Blend the overlay for transparent effect
        display_frame = cv2.addWeighted(overlay, 0.3, display_frame, 0.7, 0)

        # Count and display total violations
        violation_count = sum(
            1 for j in range(len(posList_violation_video[1]))
            if occupy_times_violation[j] != 0 and (time.time() - occupy_times_violation[j]) >= VIOLATION_SECONDS
        )

        cv2.putText(display_frame, f'Violations: {violation_count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)






    from datetime import timezone

    if view_mode in [0, 1] and selected_slot:
        now = datetime.now(timezone.utc)  # current UTC time

        # Filter only active reservations for the selected slot
        active_res = None
        for res in reservation_data:
            if res.get("slotId") == selected_slot:
                try:
                    start = datetime.fromisoformat(res["startTime"].replace("Z", "+00:00"))
                    end = datetime.fromisoformat(res["endTime"].replace("Z", "+00:00"))
                    if start <= now <= end:
                        active_res = res  # currently active one
                        break
                except Exception as e:
                    print(f"⚠️ Time parse error for {selected_slot}: {e}")

        # ✅ Only show popup if active reservation found
        if active_res:
            try:
                # Convert UTC to IST (+5:30)
                start_dt = datetime.fromisoformat(active_res["startTime"].replace("Z", "+00:00")) + timedelta(hours=5, minutes=30)
                end_dt = datetime.fromisoformat(active_res["endTime"].replace("Z", "+00:00")) + timedelta(hours=5, minutes=30)

                # Format for display
                start_time = start_dt.strftime("%I:%M %p")
                end_time = end_dt.strftime("%I:%M %p")

                data_with_time = active_res.copy()
                data_with_time["startTime"] = start_time
                data_with_time["endTime"] = end_time

                show_popup(display_frame, selected_slot, data_with_time)
            except Exception as e:
                print(f"⚠️ Error displaying active reservation for {selected_slot}: {e}")


    elif view_mode == 2:
        # Do not clear selected_slot — just don't draw the popup
        pass


    # --- 🕒 Real-time Clock Display ---
        # --- 🕒 Real-time Clock (always visible, time + date) ---
    now = datetime.now()
    time_str = now.strftime("Time - %I : %M : %S %p")   # Example: Time - 09 : 26 : 36 PM
    date_str = now.strftime("%d %b %Y")                 # Example: 05 Nov 2025

    # Time (bigger)
    time_x, time_y = 1345, 200
    cv2.putText(display_frame, time_str, (time_x, time_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3)

    # Date (smaller, just below the time)
    date_x, date_y = 1345, time_y + 60
    cv2.putText(display_frame, date_str, (date_x, date_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3)



    cv2.imshow("Parking Monitor", display_frame)
    cv2.setMouseCallback("Parking Monitor", mouse_callback)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        view_mode = (view_mode + 1) % 3

cap.release()
cv2.destroyAllWindows()
