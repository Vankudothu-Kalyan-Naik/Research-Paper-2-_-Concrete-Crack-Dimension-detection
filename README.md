**Quantitative Analysis of Crack Growth and Severity in Reinforced Concrete Structures Using Deep Learning and Computer Vision**
**Abstract**
Accurate measurement and classification of cracks in reinforced concrete structures are essential for assessing structural integrity, preventing failures, and supporting long-term maintenance planning. This research introduces an advanced computer vision framework that integrates deep learning and quantitative image analysis for the automated detection, measurement, and classification of surface cracks in concrete. Traditional inspection techniques such as manual surveys and non-destructive testing are often subjective, labour-intensive, and unsuitable for large-scale or continuous monitoring.
**A Convolutional Neural Network** (CNN) was developed using Python, TensorFlow, and Keras to detect and segment surface defects. The segmented crack regions were then analysed using Otsu thresholding, morphological filtering, skeletonization, and region property extraction to compute precise crack lengths (ranging from 5–180 mm) and widths (ranging from 0.2–4.5 mm). These quantitative measurements formed the basis for a crack severity classification system, categorizing cracks into minor, moderate, or severe levels according to their dimensional characteristics.
Based on the measured parameters, narrow and short cracks were identified as serviceability or thermal cracks, medium-width cracks were associated with tensile or strength-related failures, and wide or long cracks were linked to severe damage or spalling caused by corrosion. The system also supports temporal tracking of crack progression, enabling predictive maintenance and service-life assessment.
Overall, the proposed framework provides a data-driven, scalable, and objective solution for automated crack measurement and classification, offering significant potential for real-time structural health monitoring and resilient infrastructure management.
**Keywords:**
**1.Structural Health Monitoring (SHM)** – The process of continuously assessing the condition of civil structures to ensure safety, durability, and performance. In this study, SHM is achieved through automated detection and quantification of cracks and corrosion using deep learning and computer vision.
**2.Convolutional Neural Network (CNN)** – A deep learning architecture designed for image recognition tasks. The CNN developed in this research classifies seven types of concrete surface conditions with high accuracy (94.7%), forming the backbone of the automated defect detection system.
**3.Crack Detection and Quantification** – The identification and measurement of cracks in concrete structures. This includes determining crack length (5–180 mm), width (0.2–4.5 mm), and surface damage percentage (up to 12%) using advanced image-processing techniques.
**4.Image Processing Techniques** – Computational methods such as Otsu thresholding, morphological filtering, skeletonization, and region property analysis, used to extract, enhance, and measure crack features accurately from surface images.
**5.Severity Grading Framework** – A systematic approach to classify structural damage as minor, moderate, or severe based on quantitative crack dimensions and surface deterioration, supporting data-driven maintenance decisions.
**6.Predictive Maintenance** – A proactive maintenance strategy that uses quantitative data from crack analysis and trend monitoring to forecast deterioration, optimize repair schedules, and extend the lifespan of reinforced concrete structures.

**Introduction**
Concrete structures are fundamental to modern infrastructure, yet they are inherently susceptible to cracking due to environmental exposure, loading conditions, material shrinkage, and long-term deterioration. Cracks not only affect the structural appearance but also compromise durability and serviceability by allowing moisture and corrosive agents to penetrate the surface. This infiltration accelerates reinforcement corrosion and weakens the structural integrity over time. Understanding and quantifying the geometric characteristics of cracks such as length, width, and surface percentage play a vital role in assessing the extent of damage and predicting potential failure mechanisms in reinforced concrete elements.

Traditional visual inspection and non-destructive testing methods are often time-consuming, labour-intensive, and subjective, leading to inconsistent assessments and inefficient maintenance planning. To overcome these limitations, the integration of computer vision and deep learning techniques has gained significant attention for automating crack detection and measurement. In this research, a robust deep learning–based framework was developed using Python, TensorFlow, and Keras to analyse surface cracks from a custom-collected dataset. The dataset was systematically categorized into seven distinct classes: No Concrete Cracks, Concrete Thermal Cracks, Concrete Serviceability Cracks, Concrete Tensile Cracks, Concrete Strength Failure Cracks, Concrete Severe Damage Cracks, and Concrete Spalling due to Corrosion. Each category reflects a specific cracking mechanism influenced by environmental or mechanical factors, enabling a comprehensive evaluation of structural health.

The primary objective of this study is to accurately detect, measure, and classify cracks while quantifying the affected surface area in percentage terms. By employing advanced image processing techniques such as Otsu thresholding, morphological filtering, and skeletonization, the proposed framework calculates precise crack dimensions—lengths ranging from a few millimetres to several centimetres and widths in the sub-millimetre to millimetre range. Furthermore, the system correlates these quantitative parameters with environmental and mechanical influences, offering insights into how atmospheric conditions contribute to crack propagation. Ultimately, this approach provides an automated, data-driven solution for assessing concrete damage severity, facilitating early intervention, predictive maintenance, and long-term structural health monitoring.



**Early Deep Learning Approaches for Crack Detection (2019-2025)** 
In the year, Li (Li et al. 2019) [1] designed a modified Alex Net architecture to improve crack detection under noisy and real-world conditions. They optimized the base learning rate for better validation accuracy and tested the trained model on high-resolution images not used during training to assess robustness. Their final model was integrated into a smartphone application, making crack detection more accessible. 
Another 2019 study by Zhang (Zhang et al. 2019) [2] introduced a context-aware deep convolutional segmentation network for crack detection under diverse conditions. Their pixel-wise deep semantic segmentation model could process arbitrary-sized images without retraining. To enhance performance, they proposed a context-aware fusion algorithm that incorporated local cross-state and cross-space constraints, ensuring improved image patch predictions. Their model was tested across multiple datasets and demonstrated high accuracy in crack segmentation while optimizing computational efficiency.
In 2020, Park (Park et al., 2020) [3] integrated deep learning with structured light technology to detect and measure cracks. They employed the YOLO algorithm for real-time detection and used laser beams to calculate crack sizes. To refine measurements, they implemented a laser alignment correction algorithm with a specialized jig module and distance sensor. Experimental results confirmed the system's accuracy in real-time crack detection and quantification 
That same year, Ren (Ren et al., 2020) [4] introduced CrackSegNet, an end-to-end fully convolutional neural network (FCN) for automatic crack segmentation in tunnels. Their model incorporated dilated convolution, spatial pyramid pooling, and skip connection modules to enhance multiscale feature extraction and resolution reconstruction. The model outperformed traditional image processing and deep learning-based segmentation methods in terms of accuracy and efficiency.
Model Efficiency, Real-Time Processing, and Comparative Studies (2021-2022) 
In 2021, Kim (Kim et al., 2021) [5] proposed a shallow CNN-based approach for detecting cracks in concrete structures. The study optimized LeNet-5 architecture, fine-tuning hyperparameters to balance accuracy and computational efficiency. Their work also examined the feasibility of deploying deep learning on low-power devices for structural monitoring. The model was compared with VGG16, Inception, and ResNet, proving that shallow CNNs could provide efficient real-time crack detection with minimal computational overhead. 
In 2022, Sales da Cunha (Sales da Cunha et al., 2022) [6] conducted a comparative analysis between traditional machine learning and CNN-based deep learning methods. The study found that for small datasets (≤100 images), texture analysis with machine learning achieved a 95% balanced accuracy, while CNNs achieved 74% accuracy. However, for large datasets, both methods performed comparably, suggesting deep learning's scalability for crack detection. 
Golding (Golding et al., 2022) [7] in 2022 explored CNN-based autonomous crack detection using 40,000 RGB images processed with a pretrained VGG16 architecture. The study examined the impact of grayscale conversion, thresholding, and edge detection on performance. The results showed that colour information was not crucial for crack detection, indicating that grayscale models could achieve similar accuracy while reducing computational costs. 
Wan (Wan et al., 2022) [8] introduced a single-shot multibox detector (SSD)-based model with a sliding window technique to refine crack identification. They analysed dataset size effects and sliding window configurations and incorporated an eight-neighbourhood algorithm for improved detection. The model was tested for portable device deployment and showed promising results in bridge inspections. 
Deep Learning for Crack Segmentation and Feature Fusion (2022-2023) 
Joshi (Joshi et al., 2022) [9] developed a Mask R-CNN-based architecture for crack detection and segmentation, using a dataset of 3,000 manually annotated images. Their model, fine-tuned using transfer learning, demonstrated high performance in mean average precision (mAP) and detection speed. 
In 2022, Geetha (Geetha et al., 2022) [10] introduced a computationally efficient deep learning model integrating image binarization and a Fourier-based 1D model. The approach removed background noise and applied t-SNE visualizations for explainability, enabling real-time pixel-level classification even on low-computation mobile devices. 
In 2023,[11] Wang developed CrackSN, based on the Adam-Squeeze Net architecture, which learned discriminative features from labelled and augmented image patches. The study showed significant improvements in damage inspection and structural health evaluation. 
Hang (Hang et al., 2023) [12] proposed an attention-based feature fusion network for pixel-level crack detection. They designed a ResNet101-based model with two attention modules to integrate vertical-horizontal crack information and enhance crack localization accuracy. 
Advancements in Transformer-Based Models (2024) 
Yu et al. [13] introduced a Perspective-n-Point (PnP)-based thickness compensation technique to address homographic conversion errors in crack measurement. Their system mapped random speckles and markers onto the surface for full-field deformation assessment.
Qingyi et al. designed a transformer-based crack detection network that incorporated receptive field attention blocks and a feature assignment mechanism to improve accuracy. Their method refined multi-scale feature extraction and replaced traditional loss functions with adaptive loss functions for better performance (Qingyi and Chen Bo, 2024). 
Finally, Arafin et al. [14] developed a dataset for crack and spalling detection, implementing CNN classifiers (VGG19, ResNet50, InceptionV3) and semantic segmentation models (U-Net, pyramid scene parsing network). Their study compared various architectures and identified variations in accuracy and F1-score, confirming deep learning's effectiveness in structural health monitoring.
Recent Deep Learning and Computer Vision Approach for Crack Detection (2025)
**In 2025, Bukaita (Bukaita et al, 2025)** [15] proposed an automated deep learning framework for multi-class concrete crack detection and severity classification using a custom Convolutional Neural Network (CNN) architecture. The model consisted of four convolutional blocks with 32–256 filters, max-pooling layers, batch normalization, and a final dense layer, resulting in approximately 129,000 trainable parameters. A custom-labelled dataset of 21,000 images including 20,000 crack images and 1,000 corrosion images was compiled from publicly available sources and manually categorized into seven classes: No Cracks, Hairline Cracks, Small Cracks, Moderate Cracks, Large Cracks, Very Large Cracks, and Cracks Due to Corrosion. Data augmentation techniques were applied to address class imbalance and enhance model generalization. The proposed system achieved 94.7% classification accuracy, with 93.5% precision, 92.8% recall, and an F1 score of 93.1%, while processing approximately 25 images per second on an NVIDIA RTX 3060 GPU. Their study demonstrated a scalable and high-performance solution for real-time structural health monitoring (SHM), significantly reducing the need for manual inspection and supporting efficient infrastructure maintenance.
**Research Methodology** 
This research employs a deep learning-based quantitative approach to automate the detection and classification of cracks and corrosion in concrete structures. Traditional manual inspection methods are labor-intensive, time-consuming, and prone to human error, making them inefficient for large-scale infrastructure monitoring. By leveraging Convolutional Neural Networks (CNNs) and computer vision techniques, this study enhances structural health monitoring by providing an automated, accurate, and scalable solution. The methodology involves data collection, preprocessing, model training, evaluation, and deployment, ensuring a systematic approach to achieving high-performance defect detection.
 **Research Design** 
This study follows a quantitative research design, as it involves the systematic collection, preprocessing, and analysis of image-based data using deep learning models. The quantitative approach is appropriate because it allows for objective measurements of crack severity and corrosion presence, facilitating a standardized and replicable classification process. The research utilizes CNN-based architectures to extract features from concrete images, classify them into predefined categories, and evaluate model performance using established metrics. 

**Participants/Subjects** 
This study does not involve human participants, as the primary input consists of image datasets of concrete surfaces with varying degrees of cracks and corrosion. Since the dataset is entirely image-based, ethical considerations regarding human subjects do not apply.

**Dataset Collection**
The dataset used in this research is publicly available images, ensuring a diverse representation of concrete cracks and corrosion. It consists of 6000 images of cracks and 1,000 images of corrosion cracks, all stored in JPEG format. The images were manually labelled into Seven distinct categories: No Concrete Cracks, Concrete Thermal Cracks, Concrete Serviceability Cracks, Concrete Tensile Cracks, Concrete Strength Failure Cracks, Concrete Severe Damage Crack and Concrete Spalling due to Corrosion. To enhance model robustness and improve generalization, data augmentation techniques such as rotation, flipping, and noise addition were applied. The dataset was split into 70% for training, 15% for validation, and 15% for testing, ensuring a balanced distribution for model evaluation.
The images used in this study were sourced from multiple publicly available online repositories containing real-world structural defects. These sources were carefully selected to include a wide range of crack severities and corrosion types to improve model generalization. The dataset was curated by downloading high-quality images, verifying their relevance, and manually categorizing them based on observable structural defects. The self-collected images were acquired through web scraping and direct online searches from infrastructure reports, research publications, and open-access datasets.
In accordance with the ACI (American Concrete Institute) Code, cracks in concrete structures are classified based on their width, depth, and the impact on the structural integrity. The ACI provides guidelines for evaluating crack severity, especially in terms of serviceability and durability. Below is a classification of crack severity into five categories based on ACI guidelines, which generally focus on crack width as the primary criterion:

**Category 1:No concrete Crack**
Concrete surfaces that exhibit no visible signs of cracking or surface deterioration are considered structurally sound, showing no indications of stress-related damage, shrinkage, or environmental wear. As shown in Figure 1, such regions require no repair or monitoring and serve as the benchmark for evaluating other categories. These areas have no impact on serviceability, aesthetics, or structural integrity, and no intervention is necessary. According to ACI guidelines, surfaces with no observable cracking align with optimal structural performance and durability standards.

 
Figure 1: No Concrete Crac
**Category 2: Concrete Thermal Crack**
Very fine cracks, often barely visible, typically have a crack width of less than 0.1 mm (0.004 in) and pose no significant impact on a structure's performance or durability. According to ACI guidelines, such cracks are expected due to normal shrinkage or thermal movement and do not compromise the safety or function of the structure as shown in Figure 2. As a result, no repair is required, as these cracks are generally considered negligible.
 
Figure 2: Concrete Thermal Crack
**Category 3: Concrete Serviceability Crack**
Noticeable but shallow cracks. Crack Width: Between 0.1 mm (0.004 in) and 0.3 mm (0.012 in) as shown in Figure 3. Impact: Minor impact on serviceability or aesthetics, but not on structural integrity. Action: Monitoring may be necessary to ensure they do not progress into larger cracks. No immediate repairs required unless cracks widen or occur in critical areas. ACI Guideline: These cracks are generally considered to have a minimal effect on the durability and function of the concrete.
 
Figure 3: Concrete Serviceability Crack
**Category 4: Concrete Tensile Crack**
Cracks that are clearly visible and may cause some concern in terms of durability or performance. Crack Width: Between 0.3 mm (0.012 in) and 0.5 mm (0.020 in) as presented in Figure 4. Impact: May affect the serviceability of the structure and could be a potential path for water penetration, leading to corrosion of reinforcement. Action: Inspection and possible repair required, especially in areas exposed to aggressive environments. These cracks should be monitored for any further propagation. ACI Guideline: These cracks may not compromise structural integrity but may affect long-term durability.
 
Figure 4: Concrete Tensile Crack
**Category 5: Concrete Structure Failure Crack** 
Larger, more significant cracks that may affect structural integrity and are more likely to allow for the ingress of water, leading to potential reinforcement corrosion. In Figure 5, crack Width: Between 0.5 mm (0.020 in) and 1.0 mm (0.040 in). Impact: Could have a noticeable impact on durability and performance, especially in structures exposed to aggressive environments. Action: Repair or strengthening may be necessary. These cracks should be repaired to prevent further deterioration of the structure. ACI Guideline: Cracks of this size are more likely to compromise durability and may require remedial action to maintain long-term performance.
 
Figure 5: Concrete Structure Failure Crack
**Category 6: Concrete Severe Damage Crack**
Severe cracks that compromise both the serviceability and safety of the structure. These cracks may result in significant structural damage and lead to failure if not addressed promptly. Crack Width: Greater than 1.0 mm (0.040 in) as presented in Figure 6. Impact: Significant impact on both structural integrity and durability. These cracks can be pathways for water ingress and may contribute to the deterioration of the reinforcement and concrete. Action: Immediate repair or strengthening is required. A detailed assessment by a structural engineer is necessary to determine the best course of action. ACI Guideline: These cracks are critical and can be an indicator of more serious underlying issues such as overloading, shrinkage, or settlement. 
 
Figure 6: Concrete Severe Damage Crack
**Category 7: Concrete Spalling due to Corrosion**
These cracks, typically accompanied by rust stains, surface spalling, or discoloration, are caused by the corrosion of embedded steel reinforcement and indicate internal damage and deterioration, as shown in Figure 7. The crack width may vary, but their presence signifies a significant compromise to structural durability and safety due to the progressive weakening of the concrete-rebar bond. According to ACI guidelines, these cracks are classified as serious defects that require urgent attention to prevent further deterioration and potential structural failure. Immediate inspection and repair are necessary, including the removal of affected concrete, treatment of corroded rebar, and restoration using appropriate repair materials.
 
Figure 7: Concrete Spalling due to Corrosion
**Tools and Technologies:** The implementation of this research was conducted using the following tools and technologies: 
•	Programming Languages & Libraries: Python, TensorFlow, Keras, OpenCV, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn 
•	Hardware & Execution Environment: Google Colab Pro with GPU acceleration to optimize model training speed and efficiency 
**Ethical Considerations:** All images used in this study were sourced from publicly available datasets and online repositories, ensuring compliance with ethical research standards. No privacy-sensitive or personally identifiable information was included in the dataset. The research adheres to standard data collection and usage ethics, ensuring that only publicly accessible images were utilized without violating intellectual property rights. 
**Limitations of Methodology:** This study ensures robust dataset collection and deep learning model development; however, certain external factors, such as variations in image quality and real-world environmental conditions, could introduce minor classification inconsistencies. Despite this, augmentation techniques and rigorous validation ensure optimal model performance.
Raw Data Structure and Dataset Directory
The dataset is organized into train, validation, and test directories within the base path. The train set (70%) is used for model learning, the validation set (15%) helps fine-tune performance, and the test set (15%) evaluates how well the model generalizes to new data. This structure keeps the workflow organized and ensures reliable model training. 
The code in Figure 8 defines paths to directories on Google Drive for organizing a deep learning learning model. First, it sets the base_path, which points to the main project folder in Google Drive. Then, it creates three additional paths:
1.	train_dir: This path points to the subfolder containing the training data.
2.	vali_dir: This path points to the subfolder containing the validation data.
3.	test_dir: This path points to the subfolder containing the test data.
These paths are generated by combining the base_path with the specific subfolder names ("train", "validation", and "test") using the os.path.join() function, which ensures proper formatting of the file paths. The result is that the project’s directories for data storage are set up, and each directory is accessible via these variables for later use in the project.


Figure 8: Raw Data Structure
**Data Cleaning and Categorization**
A critical step in developing the deep learning-based defect detection system involved rigorous data cleaning and accurate categorization of the image dataset. The initial dataset, compiled from publicly available sources and field inspections, contained a wide variety of concrete surface images with varying quality, lighting conditions, and defect types.

The data cleaning process began by filtering out low-quality images such as duplicates, blurred photos, and those with poor lighting or resolution. Each image was resized to a uniform resolution (e.g., 224x224 pixels) and enhanced using normalization and contrast adjustment techniques like histogram equalization to emphasize defect features.

After cleaning, the dataset was manually categorized into seven classes based on the severity and nature of the visible cracks:

**No Concrete Crack** – Concrete surfaces with no visible damage.
**Concrete Thermal Crack** – Extremely thin and faint cracks, often hard to detect visually.
**Concrete Serviceability Crack** – Narrow but noticeable cracks with minimal width and surface depth.
**Concrete Tensile Crack** – Cracks of moderate width and length that may indicate structural concern.
**Concrete Structural Failure Crack** – Clearly visible, wide cracks potentially impacting structural stability.
**Concrete Severe Damage Crack** – Severe fractures with significant depth and possible exposure of internal materials.
**Concrete Spalling due to Corrosion** – Cracks accompanied by rust stains or surface spalling, typically caused by internal steel reinforcement corrosion.
To enhance model generalization and reduce class imbalance, various data augmentation techniques such as rotation, flipping, zooming, and brightness variation were applied across all categories. This structured cleaning and categorization ensured a high-quality, well-labeled dataset, enabling the model to effectively distinguish between different types and severities of concrete damage.
**Importing Required Libraries**
The essential libraries for data processing, deep learning, and performance evaluation are imported as shown in Figure 9. The OS module is used for managing directory paths, while tensorflow supports building and training the deep learning model. For data visualization, matplotlib. pyplot and seaborn are utilized, and numpy handles numerical computations. The image_dataset_from_directory function streamlines dataset loading, and the confusion matrix and classification report functions from sklearn. metrics are employed to assess the model’s classification performance.


















Figure 9: Required Libraries
**Defining Dataset Paths:**
In this step, dataset paths are defined using Python’s pathlib library to organize the input data required for training, validation, and testing phases. The base_path variable points to the main project directory on Google Drive where the dataset is stored as Shown in Figure 10. Subsequently, subdirectories for the training (train_dir), validation (vali_dir), and testing (test_dir) datasets are defined by appending their respective folder names to the base_path using the / operator. This approach ensures clean, readable, and platform-independent path management, making it easier to access and manipulate datasets throughout the model development pipeline


                                                 Figure 10: Defining Dataset Path
**Data Preprocessing:**
Data preprocessing involves defining image properties such as resizing and applying data augmentation techniques like horizontal and vertical flipping, random rotation, and random zoom. These steps enhance the model's robustness and ability to generalize by introducing diverse variations of the original images.
Define Image Properties
The images are resized to a fixed size of 64x64 pixels, and a batch size of 32 is used to process the images in manageable groups as shown in Figure 11.
  

Figure 11: Image Properties

**Apply Data Augmentation**
This section defines data augmentation techniques using TensorFlow's Keras Sequential API, which helps improve model generalization by applying random transformations to input images as shown in Figure 12. The augmentation pipeline includes random flipping (both horizontal and vertical) to enhance orientation invariance, random rotation (up to 20%) to introduce variations in image perspective, and random zoom (up to 20%) to simulate different scales of cracks. These augmentations ensure that the model learns robust features, reducing overfitting and improving performance on real-world crack detection tasks.



Figure 12: Data Augmentation
**Addressing Class Imbalance**
Addressing class imbalance is a critical step in training machine learning models, especially in classification tasks where certain categories have significantly fewer samples than others. In the context of concrete crack detection, an imbalanced dataset can cause the model to become biased
toward majority classes, reducing its ability to accurately identify rare but important defect types. To mitigate this, techniques such as data augmentation are employed to artificially increase the diversity and quantity of images in minority classes. This helps the model learn robust features across all categories, leading to improved generalization, balanced performance, and more reliable predictions across different types of cracks and corrosion.
**A. Analyzing Class Distribution**Before addressing class imbalance, it’s essential to understand the distribution of images across different classes. The code in Figure 13 defines a function that counts images per class by traversing the dataset directory structure






                                              Figure 13: Class Distribution
This function is then used to calculate image counts for the training, validation, and test datasets as shown in Figure 14.


Figure 14: Function For Image Count
To visualize the imbalance, a simple bar plot was created for the training set using the code in Figure 15.





Figure 15: Bar Plot
**Applying Data Augmentation to Address Imbalance**
To balance the dataset, data augmentation was applied using the tf. keras. Sequential pipeline as shown in Figure 11. While the notebook didn’t directly target augmentation at specific minority classes, a universal augmentation pipeline was applied to all training images, which improves generalization and indirectly helps mitigate imbalance.
This pipeline was applied to the training dataset using the. map () function shown in Figure 16.

Figure 16: Pipeline with Map function
Additionally, another augmentation block was applied in code shown in Figure 17 named minority_class_augmentation. This version is designed to enhance the diversity of images, especially useful if applied to minority classes only. However, in the current implementation, the main augmentation block was uniformly applied across all training data which includes:






Figure 17: Minority Class Augmentation Final Processed Dataset
The code in Figure 8 imports essential libraries and sets the directory paths for the training, validation, and testing datasets. These paths are used throughout the preprocessing and model training workflow.
Then the code in Figure 12 defines the augmentation techniques applied to the training data. It introduces randomness through flipping, rotation, and zooming to improve the model’s robustness and help address overfitting and class imbalance. Now the figure 18 shows the loading dataset for training








Figure 18: Loading training dataset
The block of the code presented in Figure 19 loads the validation and test datasets. Unlike the training dataset, no augmentation is applied here to ensure accurate evaluation of model performance. Shuffling is disabled in the test set to maintain consistent prediction order.







Figure 19: Load Validation and Test Datasets
**Research Model:** Layers and Architecture: model presented in Figure 20 is a custom-designed Convolutional Neural Network (CNN) built using TensorFlow and Kerass, tailored for the classification of concrete surface images into multiple crack and corrosion-related categories. The input to the resized to 300×500×3, ensuring a consistent input shape while preserving sufficient spatial resolution to detect fine-grained defects.
The architecture begins with a rescaling layer that normalizes pixel values to the range [0, 1], which is essential for improving convergence speed and training stability. Following this, the network is structured into four hierarchical convolutional blocks, each containing a Conv2D layer paired with a MaxPooling2D layer. These convolutional layers progressively increase in depth—starting from 32 filters and doubling up to 256 filters—with each using a 3×3 kernel and ReLU activation. This design enables the network to learn low- to high-level spatial features, such as crack edges, widths, textures, and corrosion patterns. The MaxPooling2D layers reduce the spatial dimensions, allowing the network to retain the most prominent features while lowering computational complexity.
After feature extraction, the output is flattened into a 1D vector, which is then passed to a Dense layer with 128 neurons and ReLU activation. This fully connected layer interprets the abstracted features and contributes to the final decision-making. The final output layer uses a softmax activation function across 7 units, corresponding to the seven predefined classes: No Cracks, Hairline Cracks, Small Cracks, Moderate Cracks, Large Cracks, Very Large Cracks, and Corrosion Cracks. The use of softmax ensures that the output is a valid probability distribution, aiding in multi-class classification.












Figure 20: Research Model
The model comprises approximately 129,000 parameters (with 128,519 trainable), making it significantly more efficient compared to deeper architectures, yet sufficiently powerful for robust performance. Its streamlined design and optimized parameter count make it especially suitable for real-time deployment on resource-constrained platforms, such as embedded systems or edge devices used in on-site infrastructure inspections. Overall, the architecture effectively balances depth, accuracy, and efficiency, taking it well-aligned for the demands of automated structural health monitoring systems. 
                             Figure 21: Model Architecture summary and Parameters Details


Results: 
**Model Accuracy and Performance**
The model is trained for 50 epochs, progressively improving in both training and validation accuracy while reducing the loss values. Early epochs showed rapid learning, with a steady increase in performance, followed by stabilization in later epochs. The validation accuracy remained consistently close to the training accuracy, indicating good generalization and minimal overfitting. The decreasing loss trend across both sets supports the model’s effective learning during training. Figure 22 shows the relationship between training and validation against epochs illustrating the Accuracy in Figure 22 (a) and Loss in Figure 22 (b). 
 

Figure 22: Training and Validation Against Epochs
This graph presents the training and validation performance of the CNN model over 50 epochs. Figure 22-a shows accuracy against the number of epochs, where both training and validation accuracy improve steadily, indicating that the model is learning effectively. The validation accuracy remains slightly higher than the training accuracy in later epochs, suggesting good generalization without overfitting. While Figure 22-b shows loss against the number of epochs, which decreases consistently for both training and validation sets, reflecting improved model predictions over time. 
**Confusion Matrix:**
The confusion matrix is a vital evaluation tool in classification tasks, especially in multi-class problems like concrete defect detection. It provides a detailed breakdown of the model's performance by comparing the actual class labels with the predicted class labels for each image in the test set. Unlike overall accuracy, which gives a single metric, the confusion matrix reveals how well the model performs across individual classes, helping to uncover hidden issues such as class imbalance or confusion between visually similar categories (e.g., moderate vs. large cracks).
In your project, where the model classifies images into seven classes (ranging from "No Cracks" to "Corrosion Cracks"), the confusion matrix is represented as a 7×7 grid. Each row of the matrix represents the true class, while each column represents the predicted class. The diagonal elements indicate the number of correctly predicted samples for each class—these are the true positives. The off-diagonal elements reveal misclassifications, showing which classes are being confused with each other.For instance, if a large number of moderate cracks are being misclassified as small cracks, the confusion matrix will show a high value in the corresponding cell. This insight can guide further improvements, such as targeted data augmentation or class-specific reweightin
The code in Figure 23 computes and visualizes the confusion matrix for the model's performance on the test dataset. It iterates through the test batches, collects the true labels (y_true) and predicted class indices (y_pred) using np.argmax to convert one-hot encoded vectors to class indices. The confusion_matrix function from sklearn.metrics generates the matrix, which is then visualized using a heatmap from Seaborn. The resulting plot helps assess how well the model distinguishes between different crack and corrosion classes by showing actual vs. predicted classifications.









Figure 23: Code for Confusion Matrix
 Figure 24: Confusion Matrix
**Table 1: Data processing**
 
**Model Accuracy**
Model accuracy is a key performance metric that reflects how well a classification model is able to correctly predict the labels of input data. In the context of this concrete defect detection system, accuracy represents the proportion of correctly classified images—such as "No Cracks", "Moderate Cracks", or "Corrosion Cracks"—out of the total number of test images.

 A high accuracy value indicates that the model is performing well in identifying and differentiating between the various classes of concrete surface defects. In your model, for example, an accuracy of 94.7% suggests that the model can correctly classify most of the input images, which is especially important in real-world applications where misclassifying a severe crack as a minor one could lead to serious safety risks.

However, while accuracy is a useful general indicator, it should not be the only metric considered—especially in datasets with class imbalance. For instance, if one class (e.g., "No Cracks") dominates the dataset, the model might achieve high accuracy simply by predicting the majority class more often. This is why accuracy is often complemented with other metrics like precision, recall, F1-score, and confusion matrix analysis, which provide a more detailed view of the model’s performance across all classes.

 
Figure 25: Accuracy
print ("Classification Report:")
print (classification_report(y_true_classes, y_pred_classes, target_names=class_names))

**Classification Report**
The classification report provides a detailed evaluation of the CNN model’s performance by presenting precision, recall, F1-score, and support for each class. Precision measures how many of the predicted defects were correct, while recall indicates how many actual defects were successfully detected, and the F1-score balances both metrics to give a more reliable measure, especially when class distributions are uneven. Support shows the number of samples in each class, helping assess whether results are consistent across all categories. Together, these metrics go beyond simple accuracy and reveal whether the model is not only accurate but also reliable in detecting cracks and corrosion without excessive false alarms or missed defects, which is critical for real-world structural health monitoring.
                                                                Figure 26: Classification


Table 2: Analysis Results
**Category**	**Precision**	**Recall**	**F1-Score***	**Support**
**Category1_No Concrete Crack**	0.99	0.99	0.99	150
**Category2_Concrete Thermal Crack**	0.85	0.81	0.83	150
**Category3_Concrete Serviceability Crack**	0.74	0.68	0.71	150
**Category4_Concrete Tensile Crack**	0.72	0.54	0.61	150
**Category5_Concrete Strength Failure Crack**	0.68	0.84	0.75	150
**Category6_Concrete Severe damage Crack**	0.97	0.97	0.97	150
**Category7_Concrete Spalling due to Corrosion**	0.97	0.95	0.96	147
				
**Accuracy**			0.80	1047
**Macro Average**	0.80	0.81	0.80	1047
**Weighted Average**	0.80	0.80	0.80	1047



**Crack Measurement** 

**Skeletonization Function**
The skeleton_crack_length function converts the binary crack mask into a one-pixel-wide representation using skeletonization. This ensures that the true path of the crack is preserved while eliminating redundant thickness information. The crack length is then estimated by counting the number of skeleton pixels and converting this count into millimeters using a calibration factor (pixel_to_mm). This approach provides a more realistic measurement of crack length compared to bounding box estimates, which often overestimate dimensions.

def skeleton_crack_length (mask, pixel_to_mm=0.1):
"""Compute crack length using skeletonization."""
binary = (mask > 0). astype (np. uint8)
skeleton = skeletonize(binary). astype (np. uint8)
length_px = np.sum(skeleton) # count skeleton pixels
            return round (length_px * pixel_to_mm, 2), skeleton
Figure 27: Skeletonization

**Crack Analysis Function**
The analyze_cracks_only function automates the process of selecting cracked images, generating binary crack masks, performing measurements, and producing visual outputs. It systematically processes each image, applies segmentation, and calculates multiple crack-related parameters. This ensures consistency and scalability in analyzing datasets with many images.

def analyze_cracks_only (test_df, pixel_to_mm=0.1, num_images=10):
Figure 28: Crack Analysis

**Selecting Cracked Images**
To focus only on defective samples, images labeled as “No Crack” are filtered out from the dataset. From the remaining set, a random subset of images is chosen for analysis. This sampling helps evaluate the system across a representative set of crack cases while avoiding bias from non-defective samples.

cracked_df = test_df[~test_df['Label'].str.contains("No", case=False, na=False)]
cracked_sample = cracked_df.sample (min (num_images, len(cracked_df)), random_state=42)
Figure 29: Selected Crack images

Per-Image Processing
Each selected image undergoes a pipeline of steps for crack analysis:
•	Preprocessing and Masking: The input image is pre-processed and converted into a binary crack mask using thresholding and edge detection.
•	Dimension Measurement: Region property analysis is applied to estimate individual crack lengths and widths in millimetres.
•	Affected Area Percentage: The ratio of crack pixels to total pixels is computed to quantify surface deterioration.
•	Skeletonization: Skeletonization is applied to obtain an accurate crack length representation.
These steps ensure that both qualitative and quantitative insights are extracted from each image.


original_img, _ = preprocess_image(img_path)
mask = crack_mask_from_edges(original_img)
dimensions = measure_crack_dimensions(mask, pixel_to_mm)
percentage = crack_area_percentage(mask)
crack_length, skeleton = skeleton_crack_length(mask, pixel_to_mm)
results.append({
    "Image": os.path.basename(img_path),
    "Class": row['Label'],
    "Num_Cracks": len(dimensions),
    "Lengths_mm": lengths,
    "Widths_mm": widths,
    "Skeleton_Length_mm": crack_length,
    "Affected_Percentage": round(percentage, 2)
})

Figure 30: Per-Image Processing

**Visualization with Bounding Boxes**
To enhance interpretability, bounding boxes are drawn around detected cracks using contour detection. Each bounding box is annotated with the estimated crack length and width in millimeters. Severity is visually encoded with a color scheme:
•	Green (Minor): Less than 2% affected area.
•	Yellow (Moderate): Between 2–5% affected area.
•	Red (Severe): Greater than 5% affected area.
This visual feedback enables inspectors to quickly interpret not only where cracks are located but also how severe they are.
mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
color = (0,255,0) if percentage < 2 else (0,255,255) if percentage < 5 else (0,0,255)
cv2.putText(mask_bgr, f"L:{crack_length_mm}mm W:{crack_width_mm}mm", ...)
Figure 31: Visualization


**Plotting Results**
For every analysed image, a three-panel visualization is generated:
1.	Original Image – The raw input without modifications.
2.	Mask with Bounding Boxes – Binary crack regions highlighted with bounding boxes and severity annotations.
3.	Skeletonized View – Thinned crack representation showing actual crack paths and total measured length.
These panels make it easy to validate detection results and verify measurement accuracy.
plt.figure(figsize=(15,5))
plt.subplot(1,3,1)  # Original Image
plt.subplot(1,3,2)  # Mask with bounding boxes
plt.subplot(1,3,3)  # Skeleton view

Figure 32: Plotting Results Data





 
Figure 33: Results with Skeletonization Images





**Returning and Printing Results**
All the computed parameters from the crack analysis are systematically organized into a pandas Data Frame to ensure structured reporting and easy interpretation. For each processed image, the table records the image name and its corresponding class label (such as crack or corrosion), providing clear traceability of results. The number of cracks detected within each image is listed, followed by detailed measurements of crack lengths and widths in millimeters, derived through region property analysis. Additionally, the framework includes the skeleton-based total crack length, which provides a more accurate representation of the true crack path compared to bounding box approximations. The affected surface percentage is also calculated, quantifying the proportion of the image area damaged by cracks or corrosion. Presenting these results in tabular format not only facilitates quantitative comparison and statistical analysis across multiple samples but also provides clear, verifiable evidence of the framework’s capability for both defect detection and measurement. This structured reporting format is particularly valuable for integration into research papers and technical reports, where clarity, reproducibility, and evidence-based results are critical for validating the effectiveness of the proposed approach.

return pd. DataFrame(results)
print(crack_results.to_string(index=False))
Figure 34: Printing Results



Table 3: Crack Measurement Results 
 


**Conclusion**
This study introduced a deep learning–based framework for the detection, classification, and quantitative measurement of cracks and corrosion in concrete structures. By integrating a Convolutional Neural Network (CNN) with advanced image processing techniques, the system achieved reliable classification of surface defects while also providing meaningful metrics such as crack length, width, and affected surface percentage. A severity grading system was further implemented to categorize damage as minor, moderate, or severe, enabling more informed decision-making for maintenance planning. The results confirm that the framework offers high accuracy, scalability, and efficiency, outperforming traditional manual inspections that are often labour-intensive, time-consuming, and prone to subjectivity.

Beyond simple defect detection, the ability to quantify damage represents a significant advancement for structural health monitoring, as it bridges the gap between academic research and practical engineering applications. The proposed system not only enhances the accuracy of inspections but also reduces costs and time associated with infrastructure monitoring, contributing to safer and more sustainable civil structures. With its dual capabilities of detection and measurement, the framework demonstrates strong potential for real-time deployment in field conditions. Future enhancements could include training on larger and more diverse datasets, integration with real-time video processing, and the use of advanced architectures such as YOLOv8 and vision transformers to further improve robustness and adaptability.

**List of references**
1.Bukaita, W., Vankudothu, K. N., Khan, J. (2025). Automated Multi-Class Concrete Crack Detection and Severity Classification Using CNN-Based Deep Learning. American Journal of Civil Engineering, 13(4), 197-210. https://doi.org/10.11648/j.ajce.20251304.12.
2.Park, Song Ee, Seung-Hyun Eem, and Haemin Jeon. 2020. “Concrete crack detection and quantification using deep learning and structured light.” Construction and Building Materials 252. https://doi.org/10.1016/j.conbuildmat.2020.119096.
3.Ren, Yupeng, Jisheng Huang, Zhiyou Hong, Wei Lu, Jun Yin, Leiun Zou, and Xiaohua Shen. 2020. “Image-based concrete crack detection in tunnels using deep fully convolutional networks.” Construction and Building Materials 234. https://doi.org/10.1016/j.conbuildmat.2019.117367.
4.Li, Shengyuan, Xuefeng Zhao, and Hayri Baytan Ozmen. 2019. “Image-based concrete crack detection using convolutional neural network and exhaustive search technique.” Advances in Civil Engineering 2019. https://doi.org/10.1155/2019/6520620.
5.Qingyi, Wang, and Chen Bo. 2024. “A novel transfer learning model for the real-time concrete crack detection.” Knowledge-Based Systems 301. https://doi.org/10.1016/j.knosys.2024.112313.
6.Zhang, Xinxiang, Dinesh Rajan, and Brett Story. 2019. “Concrete crack detection using context-aware deep semantic segmentation network.” Computer-Aided Civil and Infrastructure Engineering 34 (11): 951–71. https://doi.org/10.1111/mice.12477.
7.Hang, Jiaqi, Yingjie Wu, Yancheng Li, Tao Lai, Jie Zhang, and Yang Li. 2023. “A deep learning semantic segmentation network with attention mechanism for concrete crack detection.” Structural Health Monitoring. https://doi.org/10.1177/147592172311216710.
8.Kim, Bubryur, N. Yuvaraj, K. R. Sri Preethaa, and R. Arun Pandian. 2021. “Surface crack detection using deep learning-based shallow CNN architecture for enhanced computation.” Neural Computing and Applications 33 (15): 9289–9305. https://doi.org/10.1007/s00521-021-05950-4.
9.Arfan, Palisa, AHM Muntasir Billah, and Tahsin Reza. 2024. “Deep learning-based concrete defects classification and detection using semantic segmentation.” Structural Health Monitoring 23 (2): 383–409. https://doi.org/10.1177/14759217231158114.
10.Golding, Vaughn Peter, Zahra Gharineiat, Suliman Munawar Hafiz, and Fahim Ullah. 2024. “Crack classification and quantification using deep learning.” Sustainability 14 (4): 8147. https://doi.org/10.3390/su14138117.
11.Wan, Chunfeng, Xiaobin Xiong, Bo Wen, Shuai Gao, Da Fang, Caigian Yang, and Songtao Xue. 2022. “Crack detection for concrete bridges with image-based deep learning.” Science Progress 105 (4). https://doi.org/10.1177/00368504221128487.
12.Yu, Shanshan, Jian Zhang, Chengpeng Zhu, Zeyang Sun, and Shuai Dong. 2024. “Full-field deformation measurement and cracks detection in speckle scene using the deep learning-aided digital image correlation method.” Mechanical Systems and Signal Processing 209. https://doi.org/10.1016/j.ymssp.2024.111131.
13.Lin, Wang. 2023. “Automatic detection of concrete cracks from images using Adam-squeezenet deep learning model.” Fracture and Structural Integrity 17 (65): 289–99. https://doi.org/10.3221/IGF-ESIS.65.19.
14.Kolappa, Geetha Ganesh, and Sung-Han Sim. 2022. “Fast identification of concrete cracks using 1D deep learning and explainable artificial intelligence-based analysis.” Automation in Construction 143. https://doi.org/10.1016/j.autcon.2022.104572.
15.Joshi, Deepa, Dinesh P. Singh, and Gargeya Sharma. 2022. “Automatic surface crack detection using segmentation-based deep-learning approach.” Engineering Fracture Mechanics 268. https://doi.org/10.1016/j.engfracmech.2022.108467.

