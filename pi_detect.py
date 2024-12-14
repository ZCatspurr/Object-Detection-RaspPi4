import cv2
from nyct_gtfs import NYCTFeed
import time
import pytesseract
import numpy as np
from yolo_pred import YOLO_Pred as YOLO
from smooth_detection import SmoothBBox
from rgbmatrix import RGBMatrix, RGBMatrixOptions
from PIL import Image, ImageDraw, ImageFont
import random

pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

def init_matrix():
    options = RGBMatrixOptions()
    options.rows = 32
    options.cols = 64
    options.chain_length = 1
    options.parallel = 1
    options.hardware_mapping = 'adafruit-hat'
    options.brightness = 50 
    return RGBMatrix(options = options)

def display_train(matrix, train_id, current_station, next_stops_north=None, next_stops_south=None):
    image = Image.new('RGB', (64, 32))
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 8)
    except:
        font = ImageFont.load_default()

    draw.text((2, 0), f"Train {train_id}", fill=(255, 255, 255), font=font)

    if current_station:
        text_width = draw.textsize(current_station, font=font)
        scroll_pos = int(time.time() * 30) % (text_width + 64) 

        draw.text((64 - scroll_pos, 8), current_station, fill=(0, 255, 0), font=font)
        if scroll_pos > text_width:
            draw.text((128 - scroll_pos, 8), current_station, fill=(0, 255, 0), font=font)

    if next_stops_north and len(next_stops_north) > 0:
        draw.text((2, 16), "↑ Next:", fill=(0, 255, 0), font=font)
        time_text = next_stops_north[0]['time']
        draw.text((2, 24), time_text, fill=(255, 0, 0), font=font)
    
    if next_stops_south and len(next_stops_south) > 0:
        draw.text((2, 24), "↓", fill=(255, 255, 255), font=font)
        time_text = next_stops_south[0]['time']
        draw.text((10, 24), time_text, fill=(255, 128, 0), font=font)

    matrix.SetImage(image)


yolo_model = YOLO(
    onnx_model = 'best.onnx',
    data_yaml = 'data.yaml'
)

def detect_train_incl_route(frame):
    processed = yolo_model.predictions(frame)

    if processed is not frame:
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
    
        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )
        
        kernel = np.ones((2,2), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        configuration = r'--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGJLMNQRSWZ1234567'
        padding = cv2.copyMakeBorder(
            thresh,
            10, 10, 10, 10,
            cv2.BORDER_CONSTANT,
            value = [255,255,255]
        )
        text = pytesseract.image_to_string(padding, config = configuration)

        text = text.strip().upper()
        valid_chars = set("ABCDEFGJLMNQRSWZ1234567")
        if text and text[0] in valid_chars:
            print(f"Train ID detected: {text[0]}")
            return True, text[0], processed

    return False, None, frame

def get_matching_route(detected_route, avail_routes):
    route_mappings = {
        'S': ['GS', 'FS', 'H'],  
        '6': ['6', '6X'],        
        '7': ['7', '7X']        
    }
    
    if detected_route in route_mappings:
        for possible_id in route_mappings[detected_route]:
            if possible_id in avail_routes:
                return possible_id
    
    if detected_route in avail_routes:
        return detected_route
        
    return None

def get_random_station(feed, route_id):
    all_stations = set()
    
    for trip in feed.trips:
        if trip.route_id == route_id and hasattr(trip, 'stop_time_updates'):
            for update in trip.stop_time_updates:
                all_stations.add(update.stop_name)
    
    return random.choice(list(all_stations)) if all_stations else None

def main():
    matrix = init_matrix()
    cap = cv2.VideoCapture(0)
    smooth_detection = SmoothBBox(buffer_size = 3)
    last_line = None
    prev_processed = None
    frame_count = 0
    
    while True:
        
        success, frame = cap.read()
        if not success:
            break

        if frame_count % 3 == 0:
            is_train_detected, train_id, processed = detect_train_incl_route(frame)

            if is_train_detected:
                prev_processed = processed
                is_stable, stable_line = smooth_detection.update(is_train_detected, train_id, None)
        
                if is_stable and stable_line and stable_line != last_line:
                    try: 
                        feed = NYCTFeed(stable_line)
                        feed.refresh()

                        time.sleep(0.5)
                        route_ids = set(trip.route_id for trip in feed.trips)
                        actual_route_id = get_matching_route(stable_line, route_ids)

                        if actual_route_id:
                            current_station = get_random_station(feed, actual_route_id)
                            if current_station:
                                print(f"Current Station: {current_station}")

                            northbound_trips = [
                                trip for trip in feed.trips 
                                if trip.route_id == actual_route_id 
                                and hasattr(trip, 'stop_time_updates')
                                and trip.stop_time_updates
                                and trip.stop_time_updates[0].stop_id.endswith('N')
                            ]

                            southbound_trips = [
                                trip for trip in feed.trips 
                                if trip.route_id == actual_route_id 
                                and hasattr(trip, 'stop_time_updates')
                                and trip.stop_time_updates
                                and trip.stop_time_updates[0].stop_id.endswith('S')
                            ]
                            
                            next_south = []
                            next_north= []

                            if northbound_trips:
                                print("\nNorthbound:")
                                current_trip = northbound_trips[0]
                                
                                if hasattr(current_trip, 'stop_time_updates'):
                                    for update in current_trip.stop_time_updates[:3]:
                                        if update.arrival and hasattr(update.arrival, 'time'):
                                            arrival_time = update.arrival.time().strftime('%I:%M %p')
                                            print(f"Next stop → {update.stop_name} at {arrival_time}")
                                            next_north.append({
                                                'stop': update.stop_name,
                                                'time': arrival_time
                                            })
                            
                            if southbound_trips:
                                print("\nSouthbound:")
                                current_trip = southbound_trips[0]
                                if hasattr(current_trip, 'stop_time_updates'):
                                    for update in current_trip.stop_time_updates[:3]:
                                        if update.arrival and hasattr(update.arrival, 'time'):
                                            arrival_time = update.arrival.time().strftime('%I:%M %p')
                                            print(f"Next stop → {update.stop_name} at {arrival_time}")
                                            next_south.append({
                                            'stop': update.stop_name,
                                            'time': arrival_time
                                        })
                                        
                            display_train(
                                matrix,
                                stable_line,
                                current_station,
                                next_north,
                                next_south
                            )

                            last_line = stable_line

                    except Exception as e:
                        print(f"Error with: {str(e)}")

        else:
            processed = prev_processed if prev_processed is not None else processed

        disp_frame = prev_processed if prev_processed is not None else processed        
        cv2.imshow("Train Detection Result", disp_frame)
        
        frame_count += 1
        if frame_count > 1000:
            frame_count = 0

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()