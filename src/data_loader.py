import glob

class DataLoader:
    def __init__(self, path):
        self.image_paths = sorted(glob.glob(path))
    
    def __len__(self):
        return len(self.image_paths)
    
    def get_frame(self, index):
        return self.image_paths[index]