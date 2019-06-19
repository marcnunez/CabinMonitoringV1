import json

class Anotations():
    def __init__(self, image_id: int, category_id: int,  score: float):
        self.image_id = image_id
        self.category_id = category_id
        self.keypoints = []
        self.score = score

    def set_keypoints(self, keypoints):
        self.keypoints = keypoints

    def add_keypoints(self, x, y, confidence):
        self.keypoints.append(x)
        self.keypoints.append(y)
        self.keypoints.append(confidence)

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__)


