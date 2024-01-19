import os
import cv2

class ImageHandler:
    def __init__(self):
        # Specify the main folder path (img folder, two directories up)
        self.main_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'img'))
        # Specify the subfolders to retrieve images from
        self.subfolders = [
            '0-Normal', '1-NoHat', '2-NoFace', '3-NoLeg', '4-NoBodyPrint',
            '5-NoHand', '6-NoHead', '7-NoArm', 'All', 'Combinations', 'Other', 'templates'
        ]

    def show_info(self):
        nb_images = self.get_all_images()
        for folder, paths in nb_images.items():
            print(f"Folder {folder}\nAmount of files: {len(paths)}")

    def get_paths_from_folder(self, folder_name):
        """
        Get all image file paths from a specific folder.

        Parameters:
        - folder_path: Path to the folder containing images.

        Returns:
        - paths: List of image file paths.
        """
        for subfolder in self.subfolders:
            if folder_name in subfolder:
                folder_name = subfolder
                break
        whole_path = os.path.join(self.main_folder_path, folder_name)
        paths = []
        for filename in os.listdir(whole_path):
            if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):
                img_path = os.path.join(whole_path, filename)
                paths.append(img_path)
        return paths

    def get_all_images(self):
        """
        Get all image file paths from multiple subfolders inside the main folder.

        Returns:
        - all_images: Dictionary where keys are subfolder names and values are lists of image file paths.
        """
        all_images = {}
        for subfolder in self.subfolders:
            paths = self.get_paths_from_folder(subfolder)
            all_images[subfolder] = paths
        return all_images


if __name__ == "__main__":
    handler = ImageHandler()
    handler.show_info()
