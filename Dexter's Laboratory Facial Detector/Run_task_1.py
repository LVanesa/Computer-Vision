from Parameters import *
from FacialDetector import *
from Visualize import *


params: Parameters = Parameters()
facial_detector: FacialDetector = FacialDetector(params)

#params.get_positive_and_negative_examples()

# Definim tipurile de dimensiuni
hog_dimensions = ['horizontal', 'vertical', 'square']

# Iterăm prin fiecare dimensiune
for dimension in hog_dimensions:
    print(f"Processing HOG descriptors for {dimension} dimension...")
    
    # Procesăm exemplele pozitive
    pos_files = []
    for char in params.characters:
        pos_path = os.path.join(params.dir_pos_examples, char, dimension, '*.jpg')
        pos_files.extend(glob.glob(pos_path))
    num_pos = len(pos_files)
    print(f"Number of positive examples for {dimension}: {num_pos}")
    
    # Construim descriptorii pentru exemplele pozitive
    positive_features_path = os.path.join(
        params.descriptori_dir,
        f'descriptoriExemplePozitive_{dimension}.npy'
    )
    
    if os.path.exists(positive_features_path):
        positive_features = np.load(positive_features_path)
        print(f"Loaded positive descriptors for {dimension}")
    else:
        print(f"Generating positive descriptors for {dimension}...")
        positive_features = facial_detector.get_positive_descriptors(pos_files, dimension)
        np.save(positive_features_path, positive_features)
        print(f"Saved positive descriptors for {dimension} in {positive_features_path}")
    
    # Procesăm exemplele negative
    neg_path = os.path.join(params.dir_neg_examples, '*.jpg')
    neg_files = glob.glob(neg_path)
    num_neg = len(neg_files)
    print(f"Number of negative examples for {dimension}: {num_neg}")

    # Construim descriptorii pentru exemplele negative
    negative_features_path = os.path.join(
        params.descriptori_dir,
        f'descriptoriExempleNegative_{dimension}.npy'
    )
    if os.path.exists(negative_features_path):
        negative_features = np.load(negative_features_path)
        print(f"Loaded negative descriptors for {dimension}")
    else:
        print(f"Generating negative descriptors for {dimension}...")
        negative_features = facial_detector.get_negative_descriptors(neg_files, dimension)
        np.save(negative_features_path, negative_features)
        print(f"Saved negative descriptors for {dimension} in {negative_features_path}")
        
    print(f"Positive features shape for {dimension}: {positive_features.shape}")
    print(f"Negative features shape for {dimension}: {negative_features.shape}")
    
    # Pasul 4. Invatam clasificatorul liniar
    model_name = f"task1_model"
    training_examples = np.concatenate((np.squeeze(positive_features), np.squeeze(negative_features)), axis=0)
    train_labels = np.concatenate((np.ones(positive_features.shape[0]), np.zeros(negative_features.shape[0])))
    facial_detector.train_classifier(training_examples, train_labels, model_name, dimension)


# Căile pentru fișierele .npy
detections_file = os.path.join(params.task1_dir, 'detections_all_faces.npy')
scores_file = os.path.join(params.task1_dir, 'scores_all_faces.npy')
file_names_file = os.path.join(params.task1_dir, 'file_names_all_faces.npy')

# Verifică dacă fișierele de date există
if os.path.exists(detections_file) and os.path.exists(scores_file) and os.path.exists(file_names_file):
    # Dacă fișierele există, le încarcă
    detections = np.load(detections_file, allow_pickle=True)
    scores = np.load(scores_file, allow_pickle=True)
    file_names = np.load(file_names_file, allow_pickle=True)
    print("Datele au fost încărcate cu succes din folderul 'task_1'.")
else:
    # Dacă fișierele nu există, rulează detecția și salvează datele
    detections, scores, file_names = facial_detector.run()
    
    # Salvează detections, scores și file_names în folderul 'task_1'
    np.save(detections_file, detections)
    np.save(scores_file, scores)
    np.save(file_names_file, file_names)
    print("Datele au fost salvate cu succes în folderul 'task_1'.")



if params.has_annotations:
    facial_detector.eval_detections(detections, scores, file_names)
    show_detections_with_ground_truth(detections, scores, file_names, params)
else:
    show_detections_without_ground_truth(detections, scores, file_names, params)
    
    