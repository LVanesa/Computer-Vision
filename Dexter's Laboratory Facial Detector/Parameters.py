import os
import cv2 as cv
import numpy as np
from Preprocessing import *

class Parameters:
    def __init__(self):
        # Setarea directoarelor de lucru
        self.imagini_antrenare = '../antrenare'
        self.base_dir = '341_Lungu_Laura'
        self.task1_dir = os.path.join(self.base_dir, 'task1')
        self.task2_dir = os.path.join(self.base_dir, 'task2')
        
        # Subfolderele din task1
        self.antrenare_dir = os.path.join(self.task1_dir, 'antrenare')
        self.dir_pos_examples = os.path.join(self.antrenare_dir, 'exemplePozitive')
        self.dir_neg_examples = os.path.join(self.antrenare_dir, 'exempleNegative')
        self.descriptori_dir = os.path.join(self.task1_dir, 'descriptori')
        self.dir_save_files = os.path.join(self.task1_dir, 'rezultate')
        
        # Fisierele de test
        self.dir_test_examples = '../testare'
        self.text_examples_path_annotations = '../validare/validare_annotations.txt'
        
        # Crearea directoarelor necesare
        for directory in [
            self.base_dir, self.task1_dir, self.task2_dir, self.antrenare_dir,
            self.dir_pos_examples, self.dir_neg_examples, self.descriptori_dir, self.dir_save_files
        ]:
            os.makedirs(directory, exist_ok=True)
            print(f"Directory {directory} exists or created.")
            
        
        self.characters = ['dad', 'deedee', 'dexter', 'mom', 'unknown']
        self.annotations = {char: os.path.join(self.imagini_antrenare, f'{char}_annotations.txt') for char in self.characters}

        

        # Parametrii de preprocesare
        self.preprocessing = Preprocessing()
        self.margin = 10  # Nr de pixeli aditionali pentru crop in jurul bbox-ului
        self.number_negative_examples_per_image = 5  # Număr total de exemple negative
        self.max_attempts = 50  # Limita de încercări pentru generarea unui exemplu negativ

        self.number_positive_examples = 0  # Inițializare număr exemple pozitive
        self.number_negative_examples = 0  # Inițializare număr exemple negative

        # set the parameters of the HOG
        self.overlap = 0.3
        
        self.hog_params = {
            'horizontal': {
                'dim': (96, 84),
                'pixels_per_cell': (6,6),  # Ajustat conform raportului 2:1
                'cells_per_block': (2, 2),
                'dim_hog_cell': 8,
                'dim_descriptor_cell': 36
            },
            'vertical': {
                'dim': (84, 96),
                'pixels_per_cell': (6,6),  # Ajustat conform raportului 1:2
                'cells_per_block': (2, 2),
                'dim_hog_cell': 8,
                'dim_descriptor_cell': 36
            },
            'square': {
                'dim': (96, 96),
                'pixels_per_cell': (6,6),  # Păstrat proporțional pentru pătrat
                'cells_per_block': (2, 2),
                'dim_hog_cell': 8,
                'dim_descriptor_cell': 36
            }
        }

        # set the parameters of the classifier testing
        self.has_annotations = True
        self.threshold = 0

    def get_positive_examples(self, char, annotations):
        """
        Preprocesează și salvează exemplele pozitive pentru un caracter dat.
        """
        for annotation in annotations:
            image_name = annotation[0]
            bbox = list(map(int, annotation[1:5]))  # xmin, ymin, xmax, ymax
            character = annotation[5]
            image_path = os.path.join(self.imagini_antrenare, char, image_name)

            image = cv.imread(image_path)
            if image is None:
                print(f"Failed to read image: {image_path}")
                continue

            # Crop și preprocesare
            cropped_face = self.preprocessing.crop_with_margin(image, bbox, self.margin)

            # Verificăm aspect ratio-ul crop-ului
            h, w = cropped_face.shape[:2]
            aspect_ratio = w / h

            if aspect_ratio > 1.2:  # Forma dreptunghiulară orizontală
                resized_face = cv.resize(cropped_face, self.hog_params['horizontal']['dim'])
                subfolder = "horizontal"
            elif aspect_ratio < 0.8:  # Forma dreptunghiulară verticală
                resized_face = cv.resize(cropped_face, self.hog_params['vertical']['dim'])
                subfolder = "vertical"
            else:  # Forma patrată
                resized_face = cv.resize(cropped_face, self.hog_params['square']['dim'])
                subfolder = "square"

            # Determinăm directorul corect în funcție de caracter și tipul ferestrei
            if character == 'unknown':
                character_dir = os.path.join(self.dir_pos_examples, 'unknown', subfolder)
            else:
                character_dir = os.path.join(self.dir_pos_examples, character, subfolder)

            # Creăm folderul pentru caracterul respectiv și tipul ferestrei, dacă nu există deja
            if not os.path.exists(character_dir):
                os.makedirs(character_dir)

            # Salvare imagine preprocesată
            """ output_path = os.path.join(character_dir, f'{character}_{image_name}')
            cv.imwrite(output_path, resized_face)
            print(f'Saved positive example to {output_path}') """

            # Aplicăm augmentări și salvăm fiecare variantă
            augmented_images = self.preprocessing.augment_image(resized_face)
            for idx, augmented in enumerate(augmented_images[0:], start=1): 
                image_name = image_name.split('.')[0]  # Eliminăm extensia jpg
                aug_output_path = os.path.join(character_dir, f'{character}_{image_name}_aug{idx}.jpg')
                cv.imwrite(aug_output_path, augmented)
                self.number_positive_examples += 1
                print(f'Saved augmented example to {aug_output_path}')

            # Incrementăm numărul de exemple pozitive
            self.number_positive_examples += 1


    def get_negative_examples(self):
        """
        Generare exemple negative din imagini brute, adică cropuri aleatorii care nu includ fețe.
        """
        annotations_dict = self.load_annotations()

        # Creăm folderul pentru exemple negative
        if not os.path.exists(self.dir_neg_examples):
            os.makedirs(self.dir_neg_examples)

        # Iterăm prin fiecare caracter și imagine
        for char in annotations_dict:
            for image_name, bboxes in annotations_dict[char].items():
                image_path = os.path.join(self.imagini_antrenare, char, image_name)
                image = cv.imread(image_path)

                if image is None:
                    print(f"Failed to read image: {image_path}")
                    continue

                h, w = image.shape[:2]

                # Verificăm dacă dimensiunile imaginii sunt suficiente pentru crop
                if h < 96 or w < 96:
                    print(f"Skipping image {image_path} due to insufficient dimensions.")
                    continue

                neg_example_count = 0
                attempt_count = 0  # Contor pentru numărul de încercări

                # Generăm cropuri negative
                while neg_example_count < self.number_negative_examples_per_image:
                    x = np.random.randint(0, w - 96)
                    y = np.random.randint(0, h - 96)
                    crop = image[y:y + 96, x:x + 96]

                    # Verificăm dacă crop-ul nu se suprapune cu zonele de față
                    valid = True
                    for bbox in bboxes:
                        xmin, ymin, xmax, ymax = bbox
                        if x < xmax and x + 96 > xmin and y < ymax and y + 96 > ymin:
                            valid = False
                            break

                    if valid:     
                        image_name_base = image_name.split('.')[0]  # Eliminăm extensia jpg
                        output_path = os.path.join(self.dir_neg_examples, f'neg_{char}_{image_name_base}_{neg_example_count}.jpg')
                        cv.imwrite(output_path, crop)
                        neg_example_count += 1
                        print(f'Saved negative example to {output_path}')

                    attempt_count += 1
                    if attempt_count >= self.max_attempts:
                        print(f"Skipping image {image_path} after {self.max_attempts} attempts due to invalid crops.")
                        break

                self.number_negative_examples += neg_example_count



    def load_annotations(self):
        """
        Încarcă și procesează fișierele de adnotări.
        Împărțim bounding box-urile pentru fiecare imagine în dicționare per caracter.
        """
        annotations_dict = {}

        for char in self.characters:
            annotations_dict[char] = {}  # Creează un dicționar pentru fiecare caracter

            if char == 'unknown':
                continue

            with open(self.annotations[char], 'r') as f:
                lines = f.readlines()

                for line in lines:
                    parts = line.strip().split()
                    image_name = parts[0]  # Numele imaginii
                    bbox = list(map(int, parts[1:5]))  # Extragem coordonatele bbox-ului
                    # Adăugăm bbox-ul în dicționarul imaginii
                    if image_name not in annotations_dict[char]:
                        annotations_dict[char][image_name] = []
                    annotations_dict[char][image_name].append(bbox)

        return annotations_dict

    def get_positive_and_negative_examples(self):
        """
        Preprocesează imaginile, salvează exemplele pozitive și generează exemple negative.
        """
        # Preprocesăm imaginile 
        for char in self.characters:
            if char == 'unknown':
                continue

            input_dir = os.path.join(self.imagini_antrenare, char)
            annotations_path = self.annotations[char]

            with open(annotations_path, 'r') as f:
                annotations = [line.strip().split() for line in f.readlines()]

            self.get_positive_examples(char, annotations)

        self.get_negative_examples()

        # Afișăm numărul total de exemple pozitive și negative
        print(f"Total positive examples: {self.number_positive_examples}")
        print(f"Total negative examples: {self.number_negative_examples}")
