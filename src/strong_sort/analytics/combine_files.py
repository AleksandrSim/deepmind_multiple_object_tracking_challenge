import os
import json

class CombineStats:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.output_path = os.path.dirname(self.folder_path)
        self.dic = {}

    def iterate_jsons(self):
        jsons = os.listdir(self.folder_path)
        for js in jsons:
            print(js)
            try:
                with open(os.path.join(self.folder_path, js), 'r') as f:
                    data = json.load(f)
                    for key, value in data.items():
                        if key not in self.dic:
                            self.dic[key] = value
                        else:
                            self.dic[key] += value
            except UnicodeDecodeError:
                print(f'Error decoding file: {js}')

    def save(self):
        with open(os.path.join(self.output_path, '3_comb.json'), 'w') as f:
            self.dic = {i:k for i,k in sorted(self.dic.items(), key = lambda x :x[1])}
            json.dump(self.dic, f, indent=2)

if __name__ == "__main__":
    stats = CombineStats('/Users/aleksandrsimonyan/Desktop/deepmind/video_stats_3')
    stats.iterate_jsons()
    stats.save()