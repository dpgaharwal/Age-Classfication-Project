
<div align="center">    
 
# Age Classification Using Transfer Learning     

![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
	![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
 ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)


<!--  
Conference   
-->   
</div>

![1_d6pG5pm8BKBe5uz16sp4zg-ezgif com-webp-to-jpg-converter](https://github.com/ThisaraWeerakoon/Age-Classification/assets/83450623/e02f049b-806e-41a5-9871-bdab48dfefd2)
# Age Prediction from Facial Images using VGG16 Architecture

## Description
This project explores the use of deep learning to predict a person's age from facial images. We apply transfer learning using the VGG16 model, a convolutional neural network pre-trained on the ImageNet dataset, to classify age groups.


## Why
Human age estimation from facial features showcases our brain's remarkable capabilities. Translating this skill to machines using deep learning techniques unlocks numerous applications:

- Enhanced Security: Improve surveillance by recognizing age-specific behaviors.
- Medical Applications: Estimate age for planning treatments and predicting health trends.

## Problem Solved
This project tackles the complex computer vision task of classifying a person's age from their facial image, using transfer learning techniques to approach the problem.

---

## Methodology

1. **Data Visualization**
    - Used the UTKFace dataset with over 20,000 facial images annotated with age, gender, and ethnicity.
    - Visualized images to understand the dataset distribution and labels.

2. **Data Preprocessing**
    - Resized images to 224x224 pixels to match the VGG16 model input size.
    - Normalized pixel values to scale them between 0 and 1.
    - Categorized age labels into 5 groups: 
      - 0-24
      - 25-49
      - 50-74
      - 75-99
      - 100-124

3. **Transfer Learning with VGG16**
    - Used the VGG16 model pre-trained on ImageNet as the base model.
    - Frozen the model's layers and added dense layers with dropout and L2 regularization.
    - The final output layer used softmax activation to classify into 5 age groups.

4. **Model Training**
    - Compiled the model with categorical cross-entropy loss and Adam optimizer.
    - Employed early stopping and model checkpoint callbacks to prevent overfitting and monitor validation performance.
    - Split the data: 90% for training, 10% for validation.

5. **Model Evaluation**
    - Evaluated the model's performance on the test set, checking accuracy and loss.
    - Plotted training and validation loss curves to visualize the learning process and detect overfitting.

6. **Age Prediction**
    - Developed a function to predict the age group of new images:
      - Preprocess input images.
      - Make predictions using the trained model.
      - Map the predictions to appropriate age groups.


## Visualization of the model used
![0_cV6Ciyjm0pdebW_2-ezgif com-webp-to-jpg-converter](https://github.com/ThisaraWeerakoon/Age-Classification/assets/83450623/e3bf3776-6907-4240-987d-5707abcb6ee9)

## Code Implementation

The project's code is organized in a Jupyter notebook, which includes detailed steps for data preprocessing, model training, and evaluation. Key libraries used in the project include:

- `numpy` for numerical operations
- `matplotlib` for data visualization
- `cv2` (OpenCV) for image processing
- `keras` for building and training the neural network
- `visualkeras` for visualizing the model architecture

## Example Usage

To test the trained model on new images, follow these steps:

1. **Preprocess the Image:**
   ```python
   def image_preprocessing(img_path):
       img = cv2.imread(img_path)
       resized_img = cv2.resize(img, (224, 224))
       normalized_img = resized_img / 255.0
       return normalized_img
2.**Predict Age Group:**
 ```python
  def predict_on_image(img_path):
    preprocessed_img = image_preprocessing(img_path)
    reshaped_img = np.reshape(preprocessed_img, (1, 224, 224, 3))
    predicted_labels_probabilities = model.predict(reshaped_img)
    class_index = np.argmax(predicted_labels_probabilities)
    age_class = str(class_index * 25) + "-" + str((class_index + 1) * 25 - 1)
    return age_class
```

3.**Visualize Prediction:**
 ```python
  new_sample_img_rgb = cv2.cvtColor(new_sample_img_bgr, cv2.COLOR_BGR2RGB)
  cv2.putText(new_sample_img_rgb, predicted_age_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
  plt.imshow(new_sample_img_rgb)
```

## Credits

We used several third-party assets and tutorials, including:

- [Tensorflow](https://www.tensorflow.org/api_docs)
- [VGG16 Model](https://keras.io/api/applications/vgg/)
- [Blog](https://medium.com/@thisara.weerakoon2001/age-classification-using-transfer-learning-vgg16-d2f240f67d26)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Badges

![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
	![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
 ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
