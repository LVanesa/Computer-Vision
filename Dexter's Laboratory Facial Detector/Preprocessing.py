import cv2 as cv
import numpy as np
import random
import albumentations as A

class Preprocessing:
    def __init__(self):
        pass  # Parametrii vor fi primiți din Parameters

    def crop_with_margin(self, image, bbox, margin=0):
        """
        Crop-ul imaginii initiale de antrenare cu păstrarea unui fundal pentru a imbunatati performanta modelului.
        bbox: (xmin, ymin, xmax, ymax)
        """
        h, w = image.shape[:2]
        xmin, ymin, xmax, ymax = bbox
        xmin = max(0, xmin - margin)
        ymin = max(0, ymin - margin)
        xmax = min(w, xmax + margin)
        ymax = min(h, ymax + margin)
        return image[ymin:ymax, xmin:xmax]

    def augment_image(self, image):
        """
        Generează imagini augmentate dintr-o imagine originală:
        - Ajustare globală de culoare și contrast
        - Flip orizontal
        - Deplasare, scalare și rotire
        - Transformare în perspectivă
        
        :param image: Imaginea originală (array numpy)
        :return: Listă cu imagini augmentate
        """
    
        # Definim transformările specifice augmentărilor
        transforms = [
            A.HorizontalFlip(p=1.0),  # Flip orizontal
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=45, p=1.0),  # Shift, Scale, Rotate
            A.Perspective(scale=(0.05, 0.1), keep_size=True, pad_mode=0, p=1.0)  # Perspective
        ]

        augmented_images = [image]  # Lista cu imagini augmentate

        # Aplicăm fiecare transformare specifică
        for transform in transforms:
            augmented = transform(image=image)
            augmented_images.append(augmented['image'])

        return augmented_images

    
    def preprocess_image(self, image, dimension):
        """
        Preprocesează imaginea înainte de a aplica descriptorii HOG:
        - Egalizarea histogramei (CLAHE)
        - Corecția Gamma
        - Gaussian Blur pentru reducerea zgomotului
        - Normalizare la intervalul [0, 1]
        - Redimensionare la dimensiunea specifică HOG
        
        Folosește biblioteca Albumentations pentru procesare.
        
        :param image: Imaginea originală (array numpy).
        :param dimension: Tipul dimensiunii ('horizontal', 'vertical', 'square').
        :return: Imagine preprocesată (array numpy).
        """
        
        preprocess = A.Compose([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0)
        ])
        image = preprocess(image=image)['image']  # Aplicăm transformarea globală
        
        # Pipeline-ul de preprocesare cu Albumentations
        preprocess_pipeline = A.Compose([
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),  # Egalizare histogramă
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),  # Corecție Gamma
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),  # Gaussian Blur
            A.Normalize(mean=0.0, std=1.0, max_pixel_value=255.0, p=1.0),  # Normalizare
            
        ])

        # Aplică pipeline-ul de preprocesare pe imagine
        preprocessed = preprocess_pipeline(image=image)['image']
    
        return preprocessed


