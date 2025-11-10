import cv2
import numpy as np
import os
import time
import json
import difflib
import threading
import pyttsx3
import simpleaudio as sa
import numpy as np


FILES_TO_CLEAR = [
    "/Users/mithunravi/Documents/Parking/reservation_extraction.txt",
    "/Users/mithunravi/Documents/Parking/reserved.txt"
]

for file_path in FILES_TO_CLEAR:
    if os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.truncate(0)  # clear contents

# ✅ Clear contents of reservation directory (delete all images)
RESERVATION_SAVE_DIR = "/Users/mithunravi/Documents/Parking/reservation"
if os.path.exists(RESERVATION_SAVE_DIR):
    for file in os.listdir(RESERVATION_SAVE_DIR):
        file_path = os.path.join(RESERVATION_SAVE_DIR, file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"⚠️ Error deleting {file_path}: {e}")
else:
    os.makedirs(RESERVATION_SAVE_DIR, exist_ok=True)

# --- Configuration ---
# (unchanged from your code)
RESERVATION_EXTRACTION_FILE = "/Users/mithunravi/Documents/Parking/reservation_extraction.txt"
USER_RESERVATIONS_FILE = "/Users/mithunravi/MP/data/User_Reservations.json"

# Comparison similarity threshold (0-1)
SIMILARITY_THRESHOLD = 0.85

# --- Global trackers for snapshots (unchanged) ---
snapshot_counter = {}
last_snap_time = {}
prev_reserved_occupied = {}  # Tracks if the slot was previously reserved+occupied


# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def beep(duration=0.5, freq=1000):
    """Play a beep sound."""
    fs = 44100
    t = np.linspace(0, duration, int(fs * duration), False)
    note = np.sin(freq * t * 2 * np.pi)
    audio = note * (2**15 - 1) / np.max(np.abs(note))
    audio = audio.astype(np.int16)
    sa.play_buffer(audio, 1, 2, fs).wait_done()

def alert_slot(slot_name):
    for _ in range(3):
        beep(0.5)
        time.sleep(0.2)


# --- Helper: load registered plate for a slot ---
def _get_registered_plate(slot_name):
    """Return registered plate (no spaces, uppercase) for a given slot, or None."""
    try:
        if not os.path.exists(USER_RESERVATIONS_FILE):
            return None
        with open(USER_RESERVATIONS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        for entry in data:
            if entry.get("slotId") == slot_name and entry.get("status", "").lower() == "active":
                plate = entry.get("vehiclePlate", "")
                return plate.replace(" ", "").upper()
    except Exception as e:
        print(f"⚠️ Error reading reservations file: {e}")
    return None

# --- Helper: parse extraction line into (filename, plate) ---
def _parse_extraction_line(line):
    """
    Expects lines like:
      reservation_slot_B1_1.jpg - TN63DB5481
    Returns (filename, plate) or (None, None) if not parseable.
    """
    try:
        if "-" not in line:
            return None, None
        filename_part, plate_part = line.strip().split("-", 1)
        filename = filename_part.strip()
        plate = plate_part.strip().replace(" ", "").upper()
        return filename, plate
    except Exception:
        return None, None

# --- Comparator that checks similarity and prints result ---
def _compare_and_print(slot_name, detected_plate):
    """Compare detected_plate (already cleaned) with registered plate and print outcome."""
    try:
        registered_plate = _get_registered_plate(slot_name)
        if not registered_plate:
            print(f"⚠️ No active reservation found for {slot_name}")
            return

        # Clean and normalize
        registered_plate = registered_plate.replace(" ", "").upper()
        detected_plate = detected_plate.replace(" ", "").upper()

        # Use difflib similarity + absolute character difference check
        similarity = difflib.SequenceMatcher(None, registered_plate, detected_plate).ratio()
        diff_count = sum(1 for a, b in zip(registered_plate, detected_plate) if a != b)
        diff_count += abs(len(registered_plate) - len(detected_plate))

        # ✅ Match if at least 85% similar OR ≤ 3 characters differ
        if similarity >= SIMILARITY_THRESHOLD or diff_count <= 3:
            print(f"✅ {slot_name}: Match → Registered: {registered_plate} | Detected: {detected_plate}")
        else:
            print(f"❌ {slot_name}: Mismatch → Registered: {registered_plate} | Detected: {detected_plate}")
            alert_slot(slot_name)


    except Exception as e:
        print(f"⚠️ Error comparing plates for {slot_name}: {e}")

# --- Background watcher: monitors the extraction file for new lines and compares ---
def _comparison_watcher(start_delay=5, poll_interval=1):
    """
    Background thread target.
    - Waits `start_delay` seconds on startup, then polls the extraction file.
    - Processes each new line once (keeps memory of processed lines).
    - For lines that mention reservation_slot_B{n}_..., extracts the slot id and plate and compares.
    """
    processed_lines = set()  # remember processed lines

    # initial delay to let the rest of the program and OCR warm up
    time.sleep(start_delay)

    print(f"🟢 Plate comparator started after {start_delay} seconds. Monitoring {RESERVATION_EXTRACTION_FILE}")

    while True:
        try:
            if os.path.exists(RESERVATION_EXTRACTION_FILE):
                with open(RESERVATION_EXTRACTION_FILE, "r", encoding="utf-8") as f:
                    lines = [ln.rstrip("\n") for ln in f]

                for ln in lines:
                    if not ln or ln in processed_lines:
                        continue

                    # Only process lines that look like our snapshot entries
                    if "reservation_slot_" in ln and "-" in ln:
                        filename, plate = _parse_extraction_line(ln)
                        if filename and plate:
                            # Infer slot name from filename: reservation_slot_B1_1.jpg -> B1
                            slot_name = None
                            if "reservation_slot_" in filename:
                                try:
                                    after = filename.split("reservation_slot_")[1]
                                    slot_name = after.split("_")[0].upper()
                                except Exception:
                                    slot_name = None

                            if slot_name:
                                _compare_and_print(slot_name, plate)
                            else:
                                print(f"⚠️ Could not infer slot from filename: {filename}")

                    processed_lines.add(ln)

            time.sleep(poll_interval)
        except Exception as e:
            print(f"⚠️ Comparator watcher error: {e}")
            time.sleep(poll_interval)

# Public API: start the comparison watcher in background (daemon thread)
def start_comparison_watcher(start_delay=5, poll_interval=1):
    t = threading.Thread(target=_comparison_watcher, args=(start_delay, poll_interval), daemon=True)
    t.start()

# --- Your existing handle_reservation_snapshots function (unchanged) ---
def handle_reservation_snapshots(posList_parking_video, posList_parking_image, img_portrait2, current_status, reserved_slots, target_slot=None):
    global snapshot_counter, last_snap_time, prev_reserved_occupied

    total_slots = len(posList_parking_image[1]) if len(posList_parking_image) > 1 else len(posList_parking_image[0])

    # Initialize trackers
    if not snapshot_counter:
        snapshot_counter = {f"B{i+1}": 0 for i in range(total_slots)}
    if not last_snap_time:
        last_snap_time = {f"B{i+1}": 0 for i in range(total_slots)}
    if not prev_reserved_occupied:
        prev_reserved_occupied = {f"B{i+1}": False for i in range(total_slots)}

    active_coords = posList_parking_image[1] if len(posList_parking_image) > 1 else posList_parking_image[0]

    for i, polygon in enumerate(active_coords):
        slot_name = f"B{i+1}"

        if slot_name not in reserved_slots:
            prev_reserved_occupied[slot_name] = False
            continue
        if target_slot is not None and slot_name != target_slot:
            continue

        is_occupied = current_status[i]['occupied']
        now = time.time()
        was_prev_reserved_occupied = prev_reserved_occupied[slot_name]

        if is_occupied and not was_prev_reserved_occupied and now - last_snap_time[slot_name] >= 5:
            pts = np.array(polygon, np.int32)
            mask = np.zeros_like(img_portrait2, dtype=np.uint8)
            cv2.fillPoly(mask, [pts], (255, 255, 255))
            masked_image = cv2.bitwise_and(img_portrait2, mask)
            x, y, w, h = cv2.boundingRect(pts)
            snapshot = masked_image[y:y+h, x:x+w]
            snapshot_counter[slot_name] += 1
            filename = os.path.join(
                RESERVATION_SAVE_DIR,
                f"reservation_slot_{slot_name}_{snapshot_counter[slot_name]}.jpg"
            )
            cv2.imwrite(filename, snapshot)
            last_snap_time[slot_name] = now

        prev_reserved_occupied[slot_name] = is_occupied

# --- Start the background comparator automatically ---
start_comparison_watcher(start_delay=5, poll_interval=1)