# DATA-FUSION-USING-MACHINE-LEARNING-AND-AN-APPROACH-FOR-IMAGE-CLASSIFICATION

For more info and code-related queries:- shivramdube329@gmail.com

All data from the training and testing with the dataset is available on:- https://drive.google.com/drive/folders/17uFMw_Qlc60bDo5aWv1xrB8HxokLpb47?usp=sharing

Aim and Objectives:-
Based on the problem statement mentioned, the aim of the proposed work is identified and formulated as follows.


Aim:-
Main goal of this project is to create an innovative technique for classifying the fused image of CT and MRI of brain tumors in two classes whether brain tumor disease is there or not. Particularly, goal is to improve the accuracy and beneficial value of brain tumor imaging by utilizing modern methods of machine learning, such as linear regression and fusion algorithms, to combine the complementing data from both modalities. By doing this study, the sole aim is to transform the area of brain tumor imaging and enhance patient care and medical outcomes by giving doctors a potent tool for precise tumor characterization, treatment planning, and monitoring. Objectives associated with the aim are summarized and described as follows.


Objectives:-

•	To develop a simple data fusion mechanism by using Discrete Wavelet Transforms (DWT).

•	To evaluate the DWT-based developed mechanism by using performance-identified evaluation parameters.

•	To develop a data fusion mechanism using machine learning by exploring linear regression.

•	To evaluate the linear regression-based developed model by using performance-identified evaluation parameters.

•	To develop a fused image dataset by utilizing existing CT and MRI images of normal brain and brain tumor.

•	To develop a classification model for brain tumor presence using fused image dataset and deep learning.

•	To evaluate the deep learning based developed classification model by using identified performance evaluation parameters.



Image Fusion Fundamentals:-
Image fusion is the process of combining multiple images of the same scene or object to create a single composite image that contains more comprehensive and enhanced information. Generally, the features of both the input images are extracted and later, these features are merged to get the fused image. The general way to fuse two images is depicted in Figure 3.1.


![image](https://github.com/shivram-dube/DATA-FUSION-USING-MACHINE-LEARNING-AND-AN-APPROACH-FOR-IMAGE-CLASSIFICATION/assets/88313584/4116305b-ddcd-49b6-a2d3-4cb5eaf0d8ed)





![image](https://github.com/shivram-dube/DATA-FUSION-USING-MACHINE-LEARNING-AND-AN-APPROACH-FOR-IMAGE-CLASSIFICATION/assets/88313584/05a24772-801e-4c0f-8562-6720d207e651)

Pseudocode:-

Pseudocode for the implementation of above procedure is as follows:-

Discrete Wavelet based Image Fusion- 
Input: image-1=CT image, image-2=MRI image, and GI=Groundtruth image
Output: FI=Fused Image, PSNR, SSIM, MSE, NRMSE, SD, E, CR and AG

Import necessary libraries- 
(Pywavelet, Opencv, and PIL)
Define a function to perform wavelet transformation of inputs- 
{
wavelet_transform(image, wavelet family type)
{
Do this for image-1 and image-2;
}
Return the values of LL, LH, HL, and HH for both the images;
Record the results of LL, LH, HL, and HH for both the images;
}
Define a function for coefficient fusion- 
{
fusecoeff(coeff for image-1, coeff for image-2, fusion method)
{
	Select the fusion method from min, max and mean;
	{
Do this for LL of image-1 and LL of image-2;
		Do this for LH of image-1 and LH of image-2;
		Do this for HL of image-1 and HL of image-2;
		Do this for HH of image-1 and HH of image-2;
	}
Return the values of LL, LH, HL, and HH for fused image;
Record the results of LL, LH, HL, and HH for fused image;
}		  Define a function to perform inverse wavelet transformation of coefficients generated after coefficient fusion- 
{
inverse_wavelet_transform(fused- LL; LH; HL; and HH, wavelet family type)
{
Reconstructed fused image FI;
}
}
Evaluation of FI and GI-
{
Evaluate PSNR, SSIM, MSE, NRMSE, SD, E, CR and AG between the fused image (FI) and a ground truth image (GI).
Return and Record the values of PSNR, SSIM, MSE, NRMSE, SD, E, CR and AG;
}
Display the fused image;

The pseudocode described above utilizes image processing libraries or frameworks to perform the DWT, coefficient fusion, image reconstruction, and evaluation steps. These libraries provide functions and tools for efficient implementation of image fusion algorithms and evaluation metrics. By leveraging these fundamentals, the code performs the steps of image decomposition, coefficient fusion, image reconstruction, and evaluation to achieve DWT-based image fusion and assessment.


Experimental Setup and Results :-
Experiments are carried out on the 38 CT and MRI images from dataset [34]. In this dataset, total 38 CT and 38 MRI images along with 38 Ground Truth images are available. Experimentations are carried out on the System Type- x64-based PC, Intel(R) Core(TM) i3-2350M CPU@2.30GHz, 2300 Mhz, 2 Core(s), 4 Logical Processor(s).
Available dataset images are shown in Figure 3.3 (a) through Figure 3.3 (e). Obtained results are presented in Figure 3.3 (a) through Figure 3.3 (e) and Figure 3.4. Performance evaluation parameters for the comparison of obtained fused images and ground truth images are presented in Table 3.1.

![image](https://github.com/shivram-dube/DATA-FUSION-USING-MACHINE-LEARNING-AND-AN-APPROACH-FOR-IMAGE-CLASSIFICATION/assets/88313584/19e8f454-c41e-4195-86a0-3c123913ee41)

![image](https://github.com/shivram-dube/DATA-FUSION-USING-MACHINE-LEARNING-AND-AN-APPROACH-FOR-IMAGE-CLASSIFICATION/assets/88313584/78963e73-b80f-4c4a-8b70-5293e631b5c8)

![image](https://github.com/shivram-dube/DATA-FUSION-USING-MACHINE-LEARNING-AND-AN-APPROACH-FOR-IMAGE-CLASSIFICATION/assets/88313584/d37a9c9a-664d-4d5f-b69f-af771d400d73)

![image](https://github.com/shivram-dube/DATA-FUSION-USING-MACHINE-LEARNING-AND-AN-APPROACH-FOR-IMAGE-CLASSIFICATION/assets/88313584/6ce21279-8a3b-462c-bdfc-3346b6827e9c)

![image](https://github.com/shivram-dube/DATA-FUSION-USING-MACHINE-LEARNING-AND-AN-APPROACH-FOR-IMAGE-CLASSIFICATION/assets/88313584/06fa1c73-181e-410a-9d01-ed03335f6bc4)

![image](https://github.com/shivram-dube/DATA-FUSION-USING-MACHINE-LEARNING-AND-AN-APPROACH-FOR-IMAGE-CLASSIFICATION/assets/88313584/fadd8ad9-fda8-4e2f-bc58-18c8e6303ca7)



LINEAR REGRESSION-BASED IMAGE FUSION:-

Linear Regression Fundamentals :-
Linear regression is a statistical technique used to model the relationship between a dependent variable and one or more independent variables. In this context, multiple linear regression is used to establish a linear relationship between the pixel values of the input CT and MRI images and the pixel values of the fused image.

4.1.1 Mathematical Representation of Multiple Linear Regression-based Image Fusion Model:-
To implement linear regression of output or response or dependent variable Y on the set of input or independent variables X = (𝑥₁, …, 𝑥n), where n is the number of inputs or predictors. Then, a linear relationship between Y and X is represented by the following equation and this equation is called as the regression equation.
Y = 𝛽₀ + 𝛽₁𝑥₁ + ⋯ + 𝛽n𝑥n + 𝜀
Where 𝛽₀, 𝛽₁, …, 𝛽n are the regression coefficients, and 𝜀 is the random error.

Linear regression calculates the estimators of the regression coefficients or simply the predicted weights, denoted with 𝑏₀, 𝑏₁, …, 𝑏n. These estimators define the following estimated regression function 
𝑓(X) = 𝑏₀ + 𝑏₁𝑥₁ + ⋯ + 𝑏n𝑥n.

This function should capture the dependencies between the inputs and output sufficiently well.
The estimated or predicted response, (Xi), for each observation 𝑖 = 1, …, 𝑛, should be as close as possible to the corresponding actual response Yi. The differences Yi - (Xi) for all observations 𝑖 = 1, …, 𝑛, are called the residuals. Regression is about determining the best predicted weights—that is, the weights corresponding to the smallest residuals.

Here, in the image fusion, the different modality images like CT and MRI images are fused. Problem formulation from image fusion point of view is as follows.

Consider the set CT consist of 38 images of CT scan and the set MRI consist of 38 images of MRI, then the response variable would be FI which will consist of 38 fused images. 

CT = (CT1…38, MRI1…38)

FI= (FI1…38)

Hence, estimated regression function (CT, MRI) would be

(CT1…38, MRI1…38) = 𝛽₀ + 𝛽1CT1…38 + 𝛽2MRI1…38  + 𝜀


Linear Regression-Based Image Fusion Model:-
The image fusion process described here is based on the Linear Regression (LR) algorithm. The process implements image fusion using multiple linear regression for combining CT and MRI images. Following are some fundamental concepts that are used in the process.

Image Fusion: Image fusion is the process of combining multiple images of the same scene or object to create a single composite image that contains more comprehensive and enhanced information. The CT and MRI images are fused to generate a fused image that combines the complementary information from both modalities.

Image Preprocessing: Preprocessing of the input images is carried out, such as resizing and normalizing. Resizing ensures that the input images have the same dimensions, which is necessary for performing regression. Normalizing the pixel values brings them to a common scale, which aids in the training process.

Training: The multiple linear regression model is trained using the CT and MRI images as inputs and the corresponding fused images as targets. The model learns the relationship between the input images and the fused images.

Evaluation Metrics: Different evaluation metrics is used to assess the quality of the model predicted fused images. These metrics include Structural Similarity Index (SSIM), Peak Signal-to-Noise Ratio (PSNR), Mean Squared Error (MSE), Normalized Root Mean Squared Error (NRMSE), Mean (MEAN), Standard Deviation (SD), Entropy (E), Cross Entropy (CR), and Average Gradient (AG). These metrics provide quantitative measures of the similarity, fidelity, and information content of the predicted fused images compared to the ground truth fused images.
By combining these preliminaries and fundamentals, the code performs linear regression-based image fusion, evaluates the quality of the predictions, and analyzes the image characteristics and dissimilarity between the input and predicted fused images.

![image](https://github.com/shivram-dube/DATA-FUSION-USING-MACHINE-LEARNING-AND-AN-APPROACH-FOR-IMAGE-CLASSIFICATION/assets/88313584/8c979f00-8a27-41db-8de7-c901873d2588)

Pseudocode for Multiple Linear Regression-based Image Fusion Model:-


Pseudocode for the implementation of above procedure is as follows:

Linear Regression based Image Fusion- 
Input: image-1=CT image, image-2=MRI image, and GI=Groundtruth image
Ct_dir, mri_dir, fused_dir 
Output: FI=New Fused Image, PSNR, SSIM, MSE, NRMSE, SD, E, CR and AG

Import necessary libraries- 
(pandas as pd, numpy as np, os  Opencv2, PIL import Image from sklearn.linear_model import LinearRegression, tensorflow as tf)

Set the directory paths for the CT, MRI, and fused image directories 
{
(Ct_dir, Mri_dir, Fused_dir)
}
Load CT images from ct_dir 
{
-Load CT images from ct_dir 
- Normalize and resize each CT image 
-Store normalized CT images in the ct_images list
}
Load MRI images from Mri_dir 
{
-Load MRI images from MRI_dir 
- Normalize and resize each MRI image 
-Store normalized MRI images in the mri_images list
}
Load Fused images from Fused_dir 
{
-Load fused images from Fused_dir 
- Normalize and resize each fused image 
-Store normalized fused images in fused_images list
}
Reshape CT, MRI, and fused images to 1D arrays
{
 	{
 ct_reshape = ct_images.reshape((-1, 1)) mri_reshape = mri_images.reshape((-1, 1)) fused_reshape = fused_images.reshape((-1, 1)) 
}
Record the results of ct_reshape, mri_reshape, and fused_reshape 
}
Train a linear regression model using ct_reshape and mri_reshape images as input and fused_reshape images as targets
{
X = np.concatenate((ct_reshape, mri_reshape), axis=1) y = fused_reshape model = LinearRegression().fit(X, y)
}
Read new Image-1 and Image-2 
{
{
Read Image-1 and image-2 
- Normalize and resize the Image-1 and image-2 
- Reshape the Image-1 and image-2 to 1D arrays 
}
Record the results of Image-1,and Image-2;
}
 Predict fused images for the Image-1 and Image-2  using the trained model
{
Concatenate the reshape images and predicted new fused image
{
new_fused = np.concatenate((Image-1_reshape, Image-2_reshape), axis=1) new_fused_reshape = model.predict(new_fused)
}
}
Reshape and rescale the predicted fused image
{
Reshape the predicted fused image to the original shape[256,256] 
Rescale the predicted fused image to [0, 255]
}
Evaluation of 
{
Evaluate PSNR, SSIM, MSE, NRMSE, SD, E, CR and AG for(FI) 
Return and Record the values of PSNR, SSIM, MSE, NRMSE, SD, E, CR and AG;
}
Display the fused image;



Experimental Setup and Results :-
Experiments are carried out on the 38 CT and MRI images from dataset [34]. In this dataset, total 38 CT and 38 MRI images along with 38 Ground Truth images are available. Experimentations are carried out on the System Type- x64-based PC, Intel(R) Core(TM) i3-2350M CPU@2.30GHz, 2300 Mhz, 2 Core(s), 4 Logical Processor(s).
Available dataset images and obtained results are shown in Figure 4.2 (a) through Figure 4.2 (f). Performance evaluation parameters for the comparison of obtained fused images and ground truth images are presented in Table 3.1.

![image](https://github.com/shivram-dube/DATA-FUSION-USING-MACHINE-LEARNING-AND-AN-APPROACH-FOR-IMAGE-CLASSIFICATION/assets/88313584/ae0d7280-b2d5-47bb-8375-c111069d7621)

![image](https://github.com/shivram-dube/DATA-FUSION-USING-MACHINE-LEARNING-AND-AN-APPROACH-FOR-IMAGE-CLASSIFICATION/assets/88313584/822fb98f-f39f-454d-bce5-922a29ca0569)

![image](https://github.com/shivram-dube/DATA-FUSION-USING-MACHINE-LEARNING-AND-AN-APPROACH-FOR-IMAGE-CLASSIFICATION/assets/88313584/b83a2a85-da70-403f-8ab1-f4f9c5a3112c)

![image](https://github.com/shivram-dube/DATA-FUSION-USING-MACHINE-LEARNING-AND-AN-APPROACH-FOR-IMAGE-CLASSIFICATION/assets/88313584/f0f99fc5-5ecb-4126-8e3b-acf4acdab26d)

![image](https://github.com/shivram-dube/DATA-FUSION-USING-MACHINE-LEARNING-AND-AN-APPROACH-FOR-IMAGE-CLASSIFICATION/assets/88313584/75bab006-6c09-4436-b55c-cbeee7adee9e)

![image](https://github.com/shivram-dube/DATA-FUSION-USING-MACHINE-LEARNING-AND-AN-APPROACH-FOR-IMAGE-CLASSIFICATION/assets/88313584/2fceef52-b816-4c57-bcba-d973d4be3fd1)


Dataset Generation Using Linear Regression-Based Image Fusion:-

Dataset Generation Using Linear Regression-Based Image Fusion Fundamentals :

The image fusion process described here is based on the Linear Regression(LR) algorithm ,: The process implements image fusion using linear regression for combining CT and MRI images and generated fused images saved in a specified directory. Following are some fundamental concepts that are used in the process.

Procedure
The complete procedure for Linear Regression based image fusion and dataset generation is described below.

Step-1:	Import necessary libraries.
Step-2:	 Set directory paths for CT, MRI, and fused image folders.
Step-3:	 Load CT, MRI, and fused images into numpy arrays.
Step-4:	Reshape images for linear regression.
Step-5:	Train a linear regression model using CT and MRI images as input and fused images as the target.
Step-6:	Load a new CT image and preprocess it.
Step-7:	Iterate over MRI images:
Step-8:	   - Load an MRI image and preprocess it.
Step-9:	   - Use the trained model to predict the fused image.
Step-10:	   - Reshape and rescale the predicted fused image.
Step-11:	   - Generate a filename and save the image to the output folder.
Step-12:	   - Display the predicted fused images



Pseudocode Pseudocode for the implementation of above procedure is as follows:

Linear Regression based Image Fusion- 
Input: image-1=CT image, image-2=MRI image, and GI=Groundtruth image
Ct_dir, mri_dir, fused_dir , output_folder, new_ct_path ,new_mri_folder_path
Output: FI=New Fused Image, PSNR, SSIM, MSE, NRMSE, SD, E, CR and AG

Import necessary libraries- 
(pandas as pd, numpy as np, os  Opencv2, PIL import Image from sklearn.linear_model import LinearRegression, tensorflow as tf)

Set the directory paths for the CT, MRI, and fused image directories 
{
(Ct_dir, Mri_dir, Fused_dir, new_ct_path, new_mri_folder_path,                    output_folder)
}
Load CT images from ct_dir 
{
-Load CT images from ct_dir 
- Normalize and resize each CT image 
-Store normalized CT images in the ct_images list
}
Load MRI images from Mri_dir 
{
-Load MRI images from MRI_dir 
- Normalize and resize each MRI image 
-Store normalized MRI images in the mri_images list
}
Load Fused images from Fused_dir 
{
-Load fused images from Fused_dir 
- Normalize and resize each fused image 
-Store normalized fused images in fused_images list
}
Reshape CT, MRI, and fused images to 1D arrays
{
 	{
 ct_reshape = ct_images.reshape((-1, 1)) mri_reshape = mri_images.reshape((-1, 1)) fused_reshape = fused_images.reshape((-1, 1)) 
}
Record the results of ct_reshape, mri_reshape, and fused_reshape 
}
Train a linear regression model using ct_reshape and mri_reshape images as input and fused_reshape images as targets
{
X = np.concatenate((ct_reshape, mri_reshape), axis=1) y = fused_reshape model = LinearRegression().fit(X, y)
}
Read  Image-1 and new MRI directory
{
Iterate the process for all image in new MRI directory
{
           {
           Read Image-1 and images-2 in directory 
           - Normalize and resize the Image-1 and Image-2 
          - Reshape the Image-1and Image 2 to 1D arrays 
}
              Record the results of Image-1, and Image-2;
}

 Predict fused images for the Image-1 and Image-2  using the trained model
{
Concatenate the reshape images and predicted new fused image
{
new_fused = np.concatenate((Image-1_reshape, Image-2_reshape), axis=1) new_fused_reshape = model.predict(new_fused)
}
}

Reshape and rescale the predicted fused image and save in output folder
{
Reshape the predicted fused image to the original shape[256,256] 
Rescale the predicted fused image to [0, 255]
Save in output folder
}
Display the fused image;


Experimental Setup and Results :-
Experiments are carried out on the 38 CT and MRI images from dataset [35-39]. In this dataset, total 38 CT and 38 MRI images along with 38 Ground Truth images are available. Experimentations are carried out on the System Type- x64-based PC, Intel(R) Core(TM) i3-2350M CPU@2.30GHz, 2300 Mhz, 2 Core(s), 4 Logical Processor(s).
Generated fused images are shown in Figure 5.1 (a) through Figure 5.1 (c). 

![image](https://github.com/shivram-dube/DATA-FUSION-USING-MACHINE-LEARNING-AND-AN-APPROACH-FOR-IMAGE-CLASSIFICATION/assets/88313584/0fb70731-054d-436f-a819-de67e05b8c40)

![image](https://github.com/shivram-dube/DATA-FUSION-USING-MACHINE-LEARNING-AND-AN-APPROACH-FOR-IMAGE-CLASSIFICATION/assets/88313584/ccc37bf3-3885-44da-b8bc-0647b2a4eb35)


DEEP LEARNING-BASED CLASSIFICATION:-

Image Classification Fundamentals:
Image classification involves categorizing images into predefined classes. The process includes data collection, preprocessing, and training of model. Later, validation and evaluation of classification model with post-processing and deployment. The goal is to accurately classify new, unseen fused images based on the knowledge learned from a labeled dataset. The general block schematic of classification model building and evaluation is depicted in Figure 6.1.

![image](https://github.com/shivram-dube/DATA-FUSION-USING-MACHINE-LEARNING-AND-AN-APPROACH-FOR-IMAGE-CLASSIFICATION/assets/88313584/f832111e-ceb6-42db-8258-9abfb6333287)


Deep Learning-Based Image Classification:
The image classification process described here is based on the Deep Learning CNN algorithm. Deep learning CNN is widely used for image classification, leveraging their ability to automatically learn hierarchical features from raw pixel values. These models are trained on large datasets, extracting meaningful patterns, and achieving high accuracy in classifying images. 


Procedure
The complete procedure for the Classification of fused image is described below and relevant block schematic is given in Figure 6.2.

![image](https://github.com/shivram-dube/DATA-FUSION-USING-MACHINE-LEARNING-AND-AN-APPROACH-FOR-IMAGE-CLASSIFICATION/assets/88313584/63e5274d-4367-4d46-8fca-f9e22732f953)


Pseudocode:-

The Pseudocode for the implementation of the above procedure is as follows:

Image Classification based on Deep Learning- 
Input: image=Fused image,
Output: Label=[yes,no],  Precision,Recall,F-1 score, Accuracy
 Import necessary libraries-
{
(import numpy as np, import pandas as pd, import os,  import keras  import Conv2D, Flatten, Dense, MaxPooling2D,Dropout  PIL, import cv2,from sklearn.model_selection import train_test_split import tensorflow as tf)
}
 Define paths to training and testing folders-
{
train_folder_path = 'DATASET FOR CLASSIFICATION/Training' test_folder_path = 'DATASET FOR CLASSIFICATION/Testing'
}
 Load training dataset-
{
{
For each class in ['yes', 'no']:
        Read images from the training folder for the class
        Resize each image to a fixed size
        Add resized image to training data array
}
Record corresponding labels
}
Load test dataset-
{
{
		For each class in ['yes', 'no']:
        Read images from the testing folder for the class
        Resize each image to a fixed size
        Add resized image to test data array
	}
        Record corresponding labels
}
Convert image data and labels to NumPy arrays-
{
X_train = np.array(train_images) Y_train = np.array(train_labels) X_test = np.array(test_images) Y_test = np.array(test_labels)
}
 Convert labels to integers and shuffle training data-
{
Y_train = Y_train.astype(int) Y_test = Y_test.astype(int) X_train,Y_train = shuffle(X_train,Y_train,random_state=101)
}
Define ImageDataGenerator for data augmentation and preprocessing-
{
         For training data generator
{
        Apply various transformations (rotation, shift, shear, flip)
        Rescale pixel values to [0, 1]
}
    For test data generator
	   {
       Rescale pixel values to [0, 1]
	}
}
Split training data into training and validation sets-
{
train_test_split(X_train,Y_train,test_size=0.1,random_state=101)
}
Define deep learning model architecture –
{
	   Add convolutional layers with relu activation
    Add max pooling layers
    Add dropout layers for regularization
    Add dense layers with softmax activation for classification
}
Compile the model-
{ 
categorical cross-entropy loss, 
Adam optimizer
}
Train the model using fit function-
{
    Specify training data, batch size, epochs, and validation split
}
Evaluate the model on the test data-
{
	Generate a classification report using the test data
{
    Calculate precision, recall, F1-score, and support for each class
    Print the classification report
}
}
Load a sample image for prediction-
{
Resize and reshape the image
}
Pass the image through the model for prediction-
{
    Prediction=model.predict(image)
}
Display the predicted label with image;


Experimental Setup and Results :-
Experiments are carried out on the 169 CT and MRI fused images from generated dataset. Experimentations are carried out on the System Type- x64-based PC, Intel(R) Core(TM) i3-2350M CPU@2.30GHz, 2300 Mhz, 2 Core(s), 4 Logical Processor(s).
Obtained results are presented in Figure 6.3 through Figure 6.8. Performance evaluation parameters for the comparison of obtained fused images and ground truth images are presented in Table 6.1 through Table 6.2.

![image](https://github.com/shivram-dube/DATA-FUSION-USING-MACHINE-LEARNING-AND-AN-APPROACH-FOR-IMAGE-CLASSIFICATION/assets/88313584/8ce8e2e1-ced6-4143-8808-71bc37079c50)

![image](https://github.com/shivram-dube/DATA-FUSION-USING-MACHINE-LEARNING-AND-AN-APPROACH-FOR-IMAGE-CLASSIFICATION/assets/88313584/b90229d0-f67b-4071-ab51-edbd39f42f9a)

![image](https://github.com/shivram-dube/DATA-FUSION-USING-MACHINE-LEARNING-AND-AN-APPROACH-FOR-IMAGE-CLASSIFICATION/assets/88313584/0090cd1f-36f8-4e9e-8eb2-5c7584394ad9)

![image](https://github.com/shivram-dube/DATA-FUSION-USING-MACHINE-LEARNING-AND-AN-APPROACH-FOR-IMAGE-CLASSIFICATION/assets/88313584/c47d1eb1-764f-4a2d-880f-214cb848a6e7)

![image](https://github.com/shivram-dube/DATA-FUSION-USING-MACHINE-LEARNING-AND-AN-APPROACH-FOR-IMAGE-CLASSIFICATION/assets/88313584/c6c2d434-690d-451a-9e44-0e0fd9aa3cc2)

![image](https://github.com/shivram-dube/DATA-FUSION-USING-MACHINE-LEARNING-AND-AN-APPROACH-FOR-IMAGE-CLASSIFICATION/assets/88313584/bcc8f339-ff1b-4ca4-951d-fee35e7faf58)

![image](https://github.com/shivram-dube/DATA-FUSION-USING-MACHINE-LEARNING-AND-AN-APPROACH-FOR-IMAGE-CLASSIFICATION/assets/88313584/50753868-39bc-431b-bc10-21589e2c7b5b)

![image](https://github.com/shivram-dube/DATA-FUSION-USING-MACHINE-LEARNING-AND-AN-APPROACH-FOR-IMAGE-CLASSIFICATION/assets/88313584/993eaf1c-6473-442d-9ac4-617d5c631a16)

![image](https://github.com/shivram-dube/DATA-FUSION-USING-MACHINE-LEARNING-AND-AN-APPROACH-FOR-IMAGE-CLASSIFICATION/assets/88313584/559e62b1-f4e2-4c4f-996b-86ea7e03e22c)


gui for the model:-

![image](https://github.com/shivram-dube/DATA-FUSION-USING-MACHINE-LEARNING-AND-AN-APPROACH-FOR-IMAGE-CLASSIFICATION/assets/88313584/54682557-1409-41dc-86c1-218c332be48f)

![image](https://github.com/shivram-dube/DATA-FUSION-USING-MACHINE-LEARNING-AND-AN-APPROACH-FOR-IMAGE-CLASSIFICATION/assets/88313584/08bb9dc8-b1cf-40cd-a18e-0e11a6fb0feb)

Prediced output of fused images with their classification report and evaluation parameters has been presented.



Results:-

Obtained results are presented in respective chapters. Figure 3.3 (a) to Figure 3.3 (e), Figure 4.2 (a) to Figure 4.2 (f), Figure 5.1 (a) to Figure 5.1 (c), and Figure 6.3 to Figure 6.8 shows the obtained results for the DWT based approach, Linear Regression based approach, Generated fused images, and the Classification model results.

Along with this, the Table 3.1, Table 4.1, Table 6.1 and Table 6.2 shows the performance evaluation parameter values, particularly related to Structural Similarity Index (SSIM), Peak Signal-to-Noise Ratio (PSNR), Mean Squared Error (MSE), Normalized Root Mean Squared Error (NRMSE), Mean (MEAN), Standard Deviation (SD), Entropy (E), Cross Entropy (CR), and Average Gradient (AG). These parameters are used to evaluate the DWT based approach and linear regression based approach. 

The performance evaluation parameters associated with the classification model developed based on deep learning are- raining Accuracy, Validation Accuracy, Training Loss and Validation Loss and overall model is evaluated based on the Confusion Matrix which includes True Positive, True Negative, False Positive, False Negative, Accuracy, Precision, Recall, F1 score, Support, macro avg, and weighted avg.
























