class SmoothBBox:
    def __init__(self, buffer_size = 3):
        self.prev_detections = []
        self.buffer_size = buffer_size
        self.prev_valid_detection = None
        self.stable = 0
        self.stable_count = 0

    def update(self, is_train_detected, train_id, confidence):
        if not is_train_detected or not train_id:
            self.stable_counter = 0
            return False, None
        
        self.prev_detections.append((is_train_detected, train_id))
        if len(self.prev_detections) > self.buffer_size:
            self.prev_detections.pop(0)

        if len(self.prev_detections) == self.buffer_size:
            detection_count = {}
            for detection in self.prev_detections:
                if detection[1]:
                    detection_count[detection[1]] = detection_count.get(detection[1], 0) + 1
            
            for line, count in detection_count.items():
                if count >= self.buffer_size * 0.4: 
                    self.prev_valid_detection = (True, line)
                    self.stable = count
                    self.stable_count += 1
                    if self.stable_count >= 1:
                        return True, line
                    return False, None

        self.stable_count = 0
        return self.prev_valid_detection if self.prev_valid_detection else (False, None)
    
    def reset(self):
        self.prev_detections = []
        self.prev_valid_detection = None
        self.stable = 0
        self.stable_count = 0