import json


class ParsingData:
    def __init__(self, file_path):
        json_file = open(file_path)
        self.json_data = json.load(json_file)

    def get_data(self, label):
        return self.json_data[label]
