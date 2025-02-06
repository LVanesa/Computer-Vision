from Parameters import *
import numpy as np
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import glob
import cv2 as cv
import pickle
import ntpath
from copy import deepcopy
import timeit
from skimage.feature import hog


class FacialDetector:
    def __init__(self, params:Parameters):
        self.params = params
        self.best_models_task_1 = {}
        
    def get_hog_features(self, image, dimension):
        """
        Extrage descriptorii HOG pentru o imagine dată, folosind parametrii specifici dimensiunii.
        """
        hog_settings = self.params.hog_params[dimension]
        dim = hog_settings['dim']
        pixels_per_cell = hog_settings['pixels_per_cell']
        cells_per_block = hog_settings['cells_per_block']

        # Redimensionăm imaginea la dimensiunea corespunzătoare
        resized_image = cv.resize(image, dim)

        # Calculăm descriptorii HOG
        hog_features = hog(
            resized_image,
            orientations=9,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            block_norm='L2-Hys',
            feature_vector=True,
            transform_sqrt=True
        )
        return hog_features

    def get_positive_descriptors(self, files, dimension):
        """
        Calculează descriptorii HOG pentru exemplele pozitive pentru dimensiunea specificată.
        :param files: Lista fișierelor de imagini.
        :param dimension: Tipul dimensiunii ('horizontal', 'vertical', 'square').
        :return: Numpy array cu descriptorii pozitivi.
        """
        positive_descriptors = []
        print(f'Calculăm descriptorii pentru {len(files)} imagini pozitive în dimensiunea {dimension}...')
        
        for i, file in enumerate(files):
            print(f'Procesăm exemplul pozitiv numărul {i+1}/{len(files)}...')
            img = cv.imread(file, cv.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Eroare la citirea fișierului {file}")
                continue
            
            processed_img = self.params.preprocessing.preprocess_image(img, dimension)
            # Extragem descriptorii HOG pentru imagine
            hog_features = self.get_hog_features(processed_img, dimension)
            positive_descriptors.append(hog_features)

        return np.array(positive_descriptors)

    def get_negative_descriptors(self, files, dimension):
        """
        Calculează descriptorii HOG pentru exemplele negative pentru dimensiunea specificată.
        :param files: Lista fișierelor de imagini.
        :param dimension: Tipul dimensiunii ('horizontal', 'vertical', 'square').
        :return: Numpy array cu descriptorii negativi.
        """
        negative_descriptors = []
        print(f'Calculăm descriptorii pentru {len(files)} imagini negative în dimensiunea {dimension}...')
        
        for i, file in enumerate(files):
            print(f'Procesăm exemplul negativ numărul {i+1}/{len(files)}...')
            img = cv.imread(file, cv.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Eroare la citirea fișierului {file}")
                continue
            processed_img = self.params.preprocessing.preprocess_image(img, dimension)
            # redimensionam imaginea la dimensiunea specificată de dimension
            img = cv.resize(processed_img, self.params.hog_params[dimension]['dim'])
            
            # Extragem descriptorii HOG pentru imagine
            hog_features = self.get_hog_features(img, dimension)
            negative_descriptors.append(hog_features)
        return np.array(negative_descriptors)


    def train_classifier(self, training_examples, train_labels, model_name, dimension):
        svm_file_name = os.path.join(self.params.descriptori_dir, f'{model_name}_{dimension}')
        if os.path.exists(svm_file_name):
            self.best_models_task_1[dimension] = pickle.load(open(svm_file_name, 'rb'))
            return

        best_accuracy = 0
        best_c = 0
        best_model = None
        Cs = [10 ** -5, 10 ** -4,  10 ** -3,  10 ** -2, 10 ** -1, 10 ** 0]
        for c in Cs:
            print('Antrenam un clasificator pentru c=%f' % c)
            model = LinearSVC(C=c)
            model.fit(training_examples, train_labels)
            acc = model.score(training_examples, train_labels)
            print(acc)
            if acc > best_accuracy:
                best_accuracy = acc
                best_c = c
                best_model = deepcopy(model)

        print('Performanta clasificatorului optim pt c = %f' % best_c)
        # salveaza clasificatorul
        pickle.dump(best_model, open(svm_file_name, 'wb'))
        self.best_models_task_1[dimension] = best_model

        # vizualizeaza cat de bine sunt separate exemplele pozitive de cele negative dupa antrenare
        # ideal ar fi ca exemplele pozitive sa primeasca scoruri > 0, iar exemplele negative sa primeasca scoruri < 0
        scores = best_model.decision_function(training_examples)
        positive_scores = scores[train_labels > 0]
        negative_scores = scores[train_labels <= 0]


        plt.plot(np.sort(positive_scores))
        plt.plot(np.zeros(len(positive_scores)))
        plt.plot(np.sort(negative_scores))
        plt.xlabel('Nr example antrenare')
        plt.ylabel('Scor clasificator')
        plt.title('Distributia scorurilor clasificatorului pe exemplele de antrenare')
        plt.legend(['Scoruri exemple pozitive', '0', 'Scoruri exemple negative'])
        plt.show()

    def intersection_over_union(self, bbox_a, bbox_b):
        x_a = max(bbox_a[0], bbox_b[0])
        y_a = max(bbox_a[1], bbox_b[1])
        x_b = min(bbox_a[2], bbox_b[2])
        y_b = min(bbox_a[3], bbox_b[3])

        inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

        box_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
        box_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)

        iou = inter_area / float(box_a_area + box_b_area - inter_area)

        return iou

    def non_maximal_suppression(self, image_detections, image_scores, image_size):
        """
        Detectiile cu scor mare suprima detectiile ce se suprapun cu acestea dar au scor mai mic.
        Detectiile se pot suprapune partial, dar centrul unei detectii nu poate
        fi in interiorul celeilalte detectii.
        :param image_detections:  numpy array de dimensiune NX4, unde N este numarul de detectii.
        :param image_scores: numpy array de dimensiune N
        :param image_size: tuplu, dimensiunea imaginii
        :return: image_detections si image_scores care sunt maximale.
        """

        # xmin, ymin, xmax, ymax
        x_out_of_bounds = np.where(image_detections[:, 2] > image_size[1])[0]
        y_out_of_bounds = np.where(image_detections[:, 3] > image_size[0])[0]
    
    
        image_detections[x_out_of_bounds, 2] = image_size[1]
        image_detections[y_out_of_bounds, 3] = image_size[0]
        sorted_indices = np.flipud(np.argsort(image_scores))
        sorted_image_detections = image_detections[sorted_indices]
        sorted_scores = image_scores[sorted_indices]

        is_maximal = np.ones(len(image_detections)).astype(bool)
        iou_threshold = 0.4
        for i in range(len(sorted_image_detections) - 1):
            if is_maximal[i] == True:  # don't change to 'is True' because is a numpy True and is not a python True :)
                for j in range(i + 1, len(sorted_image_detections)):
                    if is_maximal[j] == True:  # don't change to 'is True' because is a numpy True and is not a python True :)
                        if self.intersection_over_union(sorted_image_detections[i],sorted_image_detections[j]) > iou_threshold:is_maximal[j] = False
                        else:  # verificam daca centrul detectiei este in mijlocul detectiei cu scor mai mare
                            c_x = (sorted_image_detections[j][0] + sorted_image_detections[j][2]) / 2
                            c_y = (sorted_image_detections[j][1] + sorted_image_detections[j][3]) / 2
                            if sorted_image_detections[i][0] <= c_x <= sorted_image_detections[i][2] and \
                                    sorted_image_detections[i][1] <= c_y <= sorted_image_detections[i][3]:
                                is_maximal[j] = False
        return sorted_image_detections[is_maximal], sorted_scores[is_maximal]
    
    def run(self):
        test_images_path = os.path.join(self.params.dir_test_examples, '*.jpg')
        test_files = glob.glob(test_images_path)

        detections = None  # array cu toate detectiile pe care le obtinem
        scores = np.array([])  # array cu toate scorurile pe care le obtinem
        file_names = np.array([])  # array cu fisierele, in aceasta lista fisierele vor aparea de mai multe ori

        num_test_images = len(test_files)

        # Scalele folosite pentru detectie
        scales = [0.7]
        current_scale = 0.7
        while current_scale * 0.8 > 0.1:
            current_scale *= 0.8
            scales.append(current_scale)

        for i in range(num_test_images):
            start_time = timeit.default_timer()
            print('Procesam imaginea de testare %d/%d..' % (i, num_test_images))
            img = cv.imread(test_files[i], cv.IMREAD_GRAYSCALE)
            
            if img is None:
                continue
            

            image_scores = []
            image_detections = []

            for scale in scales:
                scaled_img = cv.resize(img, None, fx=scale, fy=scale)

                # Procesăm fiecare model ('horizontal', 'vertical', 'square')
                for dimension, model in self.best_models_task_1.items():
                    w = model.coef_.T  # Transpunem pentru compatibilitate
                    bias = model.intercept_[0]
                    
                    processed_img = self.params.preprocessing.preprocess_image(scaled_img, dimension)

                    hog_descriptors = hog(
                        processed_img,
                        pixels_per_cell=self.params.hog_params[dimension]['pixels_per_cell'],
                        cells_per_block=self.params.hog_params[dimension]['cells_per_block'],
                        feature_vector=False
                    )

                    num_cols = scaled_img.shape[1] // self.params.hog_params[dimension]['pixels_per_cell'][1] - 1
                    num_rows = scaled_img.shape[0] // self.params.hog_params[dimension]['pixels_per_cell'][0] - 1
                    num_cells_in_template_h = self.params.hog_params[dimension]['dim'][0] // self.params.hog_params[dimension]['pixels_per_cell'][0] - 1
                    num_cells_in_template_w = self.params.hog_params[dimension]['dim'][1] // self.params.hog_params[dimension]['pixels_per_cell'][1] - 1

                    step_size = self.params.hog_params[dimension]['pixels_per_cell'][0]
                    # Scanăm ferestrele HOG
                    for y in range(0, num_rows - num_cells_in_template_h + 1):
                        for x in range(0, num_cols - num_cells_in_template_w + 1):
                            descr = hog_descriptors[
                                y:y + num_cells_in_template_h,
                                x:x + num_cells_in_template_w
                            ].flatten()

                            score = np.dot(descr, w)[0] + bias
                            if score > self.params.threshold:
                                x_min = int(x * step_size / scale)
                                y_min = int(y * step_size / scale)
                                x_max = int((x * step_size + self.params.hog_params[dimension]['dim'][1]) / scale)
                                y_max = int((y * step_size + self.params.hog_params[dimension]['dim'][0]) / scale)
                                image_detections.append([x_min, y_min, x_max, y_max])
                                image_scores.append(score)

            if len(image_scores) > 0:
                image_detections, image_scores = self.non_maximal_suppression(
                    np.array(image_detections),
                    np.array(image_scores),
                    img.shape
                )
                if detections is None:
                    detections = image_detections
                else:
                    detections = np.concatenate((detections, image_detections))
                scores = np.append(scores, image_scores)
                short_name = ntpath.basename(test_files[i])
                image_names = [short_name for _ in range(len(image_scores))]
                file_names = np.append(file_names, image_names)

            end_time = timeit.default_timer()
            print('Timpul de procesare al imaginii de testare %d/%d este %f sec.' % (i, num_test_images, end_time - start_time))

        return detections, scores, file_names



    def compute_average_precision(self, rec, prec):
        # functie adaptata din 2010 Pascal VOC development kit
        m_rec = np.concatenate(([0], rec, [1]))
        m_pre = np.concatenate(([0], prec, [0]))
        for i in range(len(m_pre) - 1, -1, 1):
            m_pre[i] = max(m_pre[i], m_pre[i + 1])
        m_rec = np.array(m_rec)
        i = np.where(m_rec[1:] != m_rec[:-1])[0] + 1
        average_precision = np.sum((m_rec[i] - m_rec[i - 1]) * m_pre[i])
        return average_precision

    def eval_detections(self, detections, scores, file_names):
        ground_truth_file = np.loadtxt(self.params.text_examples_path_annotations, dtype='str')
        ground_truth_file_names = np.array(ground_truth_file[:, 0])
        ground_truth_detections = np.array(ground_truth_file[:, 1:5], int)

        num_gt_detections = len(ground_truth_detections)  # numar total de adevarat pozitive
        gt_exists_detection = np.zeros(num_gt_detections)
        # sorteazam detectiile dupa scorul lor
        sorted_indices = np.argsort(scores)[::-1]
        file_names = file_names[sorted_indices]
        scores = scores[sorted_indices]
        detections = detections[sorted_indices]

        num_detections = len(detections)
        true_positive = np.zeros(num_detections)
        false_positive = np.zeros(num_detections)
        duplicated_detections = np.zeros(num_detections)

        for detection_idx in range(num_detections):
            indices_detections_on_image = np.where(ground_truth_file_names == file_names[detection_idx])[0]

            gt_detections_on_image = ground_truth_detections[indices_detections_on_image]
            bbox = detections[detection_idx]
            max_overlap = -1
            index_max_overlap_bbox = -1
            for gt_idx, gt_bbox in enumerate(gt_detections_on_image):
                overlap = self.intersection_over_union(bbox, gt_bbox)
                if overlap > max_overlap:
                    max_overlap = overlap
                    index_max_overlap_bbox = indices_detections_on_image[gt_idx]

            # clasifica o detectie ca fiind adevarat pozitiva / fals pozitiva
            if max_overlap >= 0.3:
                if gt_exists_detection[index_max_overlap_bbox] == 0:
                    true_positive[detection_idx] = 1
                    gt_exists_detection[index_max_overlap_bbox] = 1
                else:
                    false_positive[detection_idx] = 1
                    duplicated_detections[detection_idx] = 1
            else:
                false_positive[detection_idx] = 1

        cum_false_positive = np.cumsum(false_positive)
        cum_true_positive = np.cumsum(true_positive)

        rec = cum_true_positive / num_gt_detections
        prec = cum_true_positive / (cum_true_positive + cum_false_positive)
        average_precision = self.compute_average_precision(rec, prec)
        plt.plot(rec, prec, '-')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Average precision %.3f' % average_precision)
        plt.savefig(os.path.join(self.params.dir_save_files, 'precizie_medie.png'))
        plt.show()
