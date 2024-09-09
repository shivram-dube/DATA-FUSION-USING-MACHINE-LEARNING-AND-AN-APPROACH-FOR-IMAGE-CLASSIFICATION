import tkinter as tk
from tkinter import *
from tkinter import filedialog

import numpy as np
import os
import cv2
from PIL import Image
from sklearn.linear_model import LinearRegression
import tensorflow as tf

global predicted_label_name, fused_image_resize, resized_image
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# from skimage.metrics import mean_squared_error as mse
# from skimage.metrics import normalized_root_mse as nrmse

win = Tk()
win.title('Data Fusion Using ML and an Approach for image classification')
win.geometry('1500x700')
win.config(bg='aqua')
win.resizable(False, False)

# Set the directory paths for the CT, MRI, and fused image directories
ct_dir = "D:\\project\\my dataset\\CTMRI\\linear regression dataset\\New folder\\CT"
mri_dir = "D:\\project\\my dataset\\CTMRI\\linear regression dataset\\New folder\\MRI"
fused_dir = "D:\\project\\my dataset\\CTMRI\\linear regression dataset\\New folder\\FUSED"

# Load the CT images
ct_images = []
for filename in os.listdir(ct_dir):
    img = cv2.imread(os.path.join(ct_dir, filename), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))
    img = img.astype(np.float32) / 255.0
    ct_images.append(img)
ct_images = np.array(ct_images)

# Load the MRI images
mri_images = []
for filename in os.listdir(mri_dir):
    img = cv2.imread(os.path.join(mri_dir, filename), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))
    img = img.astype(np.float32) / 255.0
    mri_images.append(img)
mri_images = np.array(mri_images)

# Load the fused images
fused_images = []
for filename in os.listdir(fused_dir):
    img = cv2.imread(os.path.join(fused_dir, filename), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))
    img = img.astype(np.float32) / 255.0
    fused_images.append(img)
fused_images = np.array(fused_images)

# Reshape the images to 1D arrays for
ct_reshape = ct_images.reshape((-1, 1))
mri_reshape = mri_images.reshape((-1, 1))
fused_reshape = fused_images.reshape((-1, 1))

# Train a linear regression model to predict fused images from CT and MRI images
X = np.concatenate((ct_reshape, mri_reshape), axis=1)
y = fused_reshape
model = LinearRegression().fit(X, y)

# Global variables to store image paths
ct_image_path = ""
mri_image_path = ""



# Function to handle CT button click event
def browse_ct_image():
    global ct_image_path

    # Open a file dialog to select CT image
    ct_image_path = filedialog.askopenfilename(initialdir=ct_dir, title="Select CT Image")

    # Load and display the CT image in a label
    ct_image = Image.open(ct_image_path)
    ct_image = ct_image.resize((256, 256))
    ct_image = ct_image.resize((185, 195))
    ct_image_tk = ImageTk.PhotoImage(ct_image)
    lb5.configure(image=ct_image_tk)
    lb5.image = ct_image_tk


# Function to handle MRI button click event
def browse_mri_image():
    global mri_image_path

    # Open a file dialog to select MRI image
    mri_image_path = filedialog.askopenfilename(initialdir=mri_dir, title="Select MRI Image")

    # Load and display the MRI image in a label
    mri_image = Image.open(mri_image_path)
    mri_image = mri_image.resize((256, 256))
    mri_image = mri_image.resize((185, 195))
    mri_image_tk = ImageTk.PhotoImage(mri_image)
    lb6.configure(image=mri_image_tk)
    lb6.image = mri_image_tk


# Function to handle Fusion button click event
from PIL import ImageTk

ssim_val=0.0
psnr_val=0.0
mean=0.0
std=0.0
std=0.0
mse=0.0
nrmse=0.0
avg_grad=0.0
cross_entropy_ct_fused=0.0
cross_entropy_mri_fused=0.0
entropy=0.0





def generate_fused_image():
    global fused_image_pil
    global ssim_val
    global psnr_val, mean, std, mse, nrmse, avg_grad, cross_entropy_ct_fused, cross_entropy_mri_fused
    global entropy
    global new_ct_resize, new_fused_resize, new_mri
    global ct_image_path, mri_image_path

    # Generate a new CT image
    new_ct = cv2.imread(ct_image_path, cv2.IMREAD_GRAYSCALE)
    new_ct = cv2.resize(new_ct, (256, 256))
    new_ct_resize = new_ct.astype(np.float32) / 255.0

    # Generate a new MRI image
    new_mri = cv2.imread(mri_image_path, cv2.IMREAD_GRAYSCALE)
    new_mri = cv2.resize(new_mri, (256, 256))
    new_mri_resize = new_mri.astype(np.float32) / 255.0

    # Reshape the new images for PCA
    new_ct_reshape = new_ct_resize.reshape((-1, 1))
    new_mri_reshape = new_mri_resize.reshape((-1, 1))

    # Predict the new fused image using the linear regression model
    new_X = np.concatenate((new_ct_reshape, new_mri_reshape), axis=1)
    new_fused_reshape = model.predict(new_X)

    # Reshape the predicted fused image to match the original image shape
    new_fused1 = new_fused_reshape.reshape((256, 256))

    # Rescale the predicted fused image to the range [0, 255]
    new_fused = cv2.normalize(new_fused1, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Convert the fused image to PIL format for displaying in tkinter
    # fused_image_pil = Image.fromarray(new_fused1)
    global fused_image_pil
    fused_image_pil = Image.fromarray(cv2.resize(new_fused, (185, 195)))

    # Convert PIL image to Tkinter PhotoImage
    fused_image_tk = ImageTk.PhotoImage(fused_image_pil)

    # Create a label to display the fused image
    lb7.configure(image=fused_image_tk)
    lb7.image = fused_image_tk

    # Calculate SSIM and PSNR for the fused image
    ssim_val = ssim(new_ct_resize, new_fused, data_range=new_fused.max() - new_fused.min(), multichannel=True)
    psnr_val = psnr(new_mri_resize, new_fused, data_range=new_fused.max() - new_fused.min())

    print("SSIM: ", ssim_val)
    print("PSNR: ", psnr_val)

    # Calculate the mean and standard deviation of the fused image
    mean = np.mean(new_fused)
    std = np.std(new_fused)

    # Print the results
    print("Mean: ", mean)
    print("Standard deviation: ", std)

    # Calculate MSE and NRMSE
    mse = np.mean((new_ct.astype(np.float32) - new_fused.astype(np.float32)) ** 2)
    nrmse = np.sqrt(mse) / (np.max(new_ct) - np.min(new_ct))

    # Print the metrics
    print("MSE: ", mse)
    print("NRMSE: ", nrmse)

    # Calculate the gradients in x and y directions using Sobel operator
    grad_x = cv2.Sobel(new_fused, cv2.CV_32F, 1, 0)
    grad_y = cv2.Sobel(new_fused, cv2.CV_32F, 0, 1)

    # Calculate the magnitude of the gradients
    grad_mag = cv2.magnitude(grad_x, grad_y)

    # Calculate the average gradient
    avg_grad = np.mean(grad_mag)

    print("Average gradient:", avg_grad)

    # Calculate the histograms of the CT and MRI images
    ct_hist = cv2.calcHist([new_ct_resize], [0], None, [256], [0, 256])
    ct_hist_normalized = ct_hist / np.sum(ct_hist)
    mri_hist = cv2.calcHist([new_mri_resize], [0], None, [256], [0, 256])
    mri_hist_normalized = mri_hist / np.sum(mri_hist)

    fused_hist = cv2.calcHist([new_fused], [0], None, [256], [0, 256])
    fused_hist_normalized = fused_hist / np.sum(fused_hist)
    cross_entropy_ct_fused = 0.0
    cross_entropy_mri_fused = 0.0
    for i in range(256):
        if ct_hist_normalized[i] > 0 and fused_hist_normalized[i] > 0:
            cross_entropy_ct_fused -= ct_hist_normalized[i] * np.log2(fused_hist_normalized[i] + 1e-7)
        if mri_hist_normalized[i] > 0 and fused_hist_normalized[i] > 0:
            cross_entropy_mri_fused -= mri_hist_normalized[i] * np.log2(fused_hist_normalized[i] + 1e-7)
    print("Cross entropy between CT and fused images:", cross_entropy_ct_fused)
    print("Cross entropy between MRI and fused images:", cross_entropy_mri_fused)

    histogram = cv2.calcHist([new_fused], [0], None, [256], [0, 256])
    histogram_normalized = histogram / np.sum(histogram)
    # plt.plot(histogram)
    # plt.show()
    entropy = -np.sum(histogram_normalized * np.log2(histogram_normalized + 1e-7))

    print('Entropy', entropy)


# Create a CT button
'''ct_button = tk.Button(win, text="Browse CT Image", command=browse_ct_image)
ct_button.pack()

# Create an MRI button
mri_button = tk.Button(win, text="Browse MRI Image", command=browse_mri_image)
mri_button.pack()

# Create a Fusion button
fusion_button = tk.Button(win, text="Fuse Images", command=generate_fused_image)
fusion_button.pack()

# Create labels to display the images
ct_image_label = tk.Label(win)
ct_image_label.pack()

mri_image_label = tk.Label(win)
mri_image_label.pack()

fused_image_label = tk.Label(win)
fused_image_label.pack()'''




















def classify():
    # Load the trained model and labels
    model = load_model("C:\\Users\MAHESH\\PycharmProjects\\Mahesh\\myprojectbraintumor2.h5")

    labels = ['no', 'yes']


    global predicted_label_name, resized_image

    global predicted_label_name, fused_image_pil

    img = fused_image_pil.resize((150, 150), Image.ANTIALIAS)
    img_array = np.array(img)

    # Reshape the image array to (150, 150, 1)
    reshaped_image = img_array.reshape((150, 150, 1))

    # Duplicate the grayscale channel to create three identical channels
    rgb_image = np.concatenate([reshaped_image] * 3, axis=2)

    # Reshape the image array to match the model's input shape
    resized_image = rgb_image.reshape((1, 150, 150, 3))

    # Normalize the resized image array
    resized_image = resized_image.astype('float32') / 255.0

    # Perform prediction on the resized image
    prediction = model.predict(resized_image)
    predicted_label_index = np.argmax(prediction)
    predicted_label_name = labels[predicted_label_index]

    lb16.configure(text=predicted_label_name)





    from sklearn.utils import shuffle



    from keras.preprocessing.image import ImageDataGenerator

    # Define folder paths
    train_dir = "C:\\Users\\MAHESH\\Downloads\\sample dataset for classification-20230531T044721Z-001\\sample dataset for classification\\Training"
    test_dir = "C:\\Users\\MAHESH\\Downloads\\sample dataset for classification-20230531T044721Z-001\\sample dataset for classification\\Testing"

    # Define image dimensions
    img_height = 128
    img_width = 128

    # Define ImageDataGenerator for training data
    train_datagen = ImageDataGenerator(
        # Data augmentation and rescaling parameters
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        rescale=1. / 255
    )

    # Define ImageDataGenerator for test data
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # Generate data batches from folders; target size affects the size of imgs in data batches...
    # ...but does not affect the actual image data that is loaded from the disk...
    # ...actual img from disk (that is fed in model) is affected by image size = 150 in preprocessing
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        class_mode='categorical')

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_height, img_width),
        class_mode='categorical')

    # Load/Read the Dataset
    labels = ['no', 'yes']
    image_size = 150
    train_folder_path = "C:\\Users\\MAHESH\\Downloads\\sample dataset for classification-20230531T044721Z-001\\sample dataset for classification\\Training"
    test_folder_path = "C:\\Users\\MAHESH\\Downloads\\sample dataset for classification-20230531T044721Z-001\\sample dataset for classification\\Testing"

    # Load the training dataset
    train_images = []
    train_labels = []
    for i, label in enumerate(labels):
        folder_path = os.path.join(train_folder_path, label)
        for filename in os.listdir(folder_path):
            img = cv2.imread(os.path.join(folder_path, filename))
            img = cv2.resize(img, (image_size, image_size))
            train_images.append(img)
            train_labels.append(i)

    # Load the test dataset for use in model.evaluate
    test_images = []
    test_labels = []
    for i, label in enumerate(labels):
        folder_path = os.path.join(test_folder_path, label)
        for filename in os.listdir(folder_path):
            img = cv2.imread(os.path.join(folder_path, filename))
            img = cv2.resize(img, (image_size, image_size))
            test_images.append(img)
            test_labels.append(i)

    # Convert the image data and label arrays to NumPy arrays
    X_train = np.array(train_images)
    Y_train = np.array(train_labels)
    X_test = np.array(test_images)
    Y_test = np.array(test_labels)

    # Convert the label arrays to integers
    Y_train = Y_train.astype(int)
    Y_test = Y_test.astype(int)

    # shuffle training dataset
    X_train, Y_train = shuffle(X_train, Y_train, random_state=101)
    X_train.shape
    model = load_model('C:\\Users\MAHESH\\PycharmProjects\\Mahesh\\myprojectbraintumor2.h5')

    labels = ['no', 'yes']
    # Classification Report
    from sklearn.metrics import classification_report

    Y_true = test_generator.classes
    Y_pred = model.predict(X_test)
    predicted_labels = np.argmax(Y_pred, axis=-1)

    print(classification_report(Y_true, predicted_labels, target_names=labels))


def Analysis():
    global ssim_val
    global psnr_val, mean, std, mse, nrmse, avg_grad, cross_entropy_ct_fused, cross_entropy_mri_fused
    global entropy

    # str1="{:.4f}".format(ssim_val)
    # SSIM_val.config(text=""+str(str1))
    SSIM_val.config(text="{:.4f}".format(ssim_val))
    PSNR_val.config(text="{:.4f}".format(psnr_val))

    Mean.config(text="{:.4f}".format(mean))

    Entropy.config(text="{:.4f}".format(entropy))

    STD_dev.config(text="{:.4f}".format(std))

    MSE.config(text="{:.4f}".format(mse))

    NRMSE.config(text="{:.4f}".format(nrmse))

    AVG_grad.config(text="{:.4f}".format(psnr_val))

    cr_Entropy_CT.config(text="{:.4f}".format(float(cross_entropy_ct_fused)))

    #cr_Entropy_CT.config(text="{:.4f}".format(cross_entropy_ct_fused))

    cr_Entropy_MRI.config(text="{:.4f}".format(float(cross_entropy_mri_fused)))
    #cr_Entropy_MRI.config(text="{:.4f}".format(cross_entropy_mri_fused))
    # Calculate SSIM and PSNR for the fused image

    '''ssim_val = ssim(new_ct_resize, new_fused, data_range=new_fused.max() - new_fused.min(), multichannel=True)
    ssim_val = "{:.4f}".format(ssim_val)
    SSIM_val.config(text="" + str(ssim_val))

    psnr_val = psnr(new_mri_resize, new_fused, data_range=new_fused.max() - new_fused.min())
    psnr_val = "{:.4f}".format(psnr_val)
    PSNR_val.config(text="" + str(psnr_val))



    # Calculate the mean and standard deviation of the fused image
    mean = np.mean(new_fused)
    Mean = "{:.4f}".format(mean)
    Mean.config(text="" + str(Mean))


    std = np.std(new_fused)
    STD_dev = "{:.4f}".format(std)
    STD_dev.config(text="" + str(STD_dev))


    # Calculate MSE and NRMSE
    mse = np.mean((new_ct_resize.astype(np.float32) - new_fused.astype(np.float32)) ** 2)
    MSE = "{:.4f}".format(mse)
    MSE.config(text="" + str(MSE))

    nrmse = np.sqrt(mse) / (np.max(new_ct_resize) - np.min(new_ct_resize))
    NRMSE = "{:.4f}".format(nrmse)
    NRMSE.config(text="" + str(NRMSE))




    # Calculate the gradients in x and y directions using Sobel operator
    grad_x = cv2.Sobel(new_fused, cv2.CV_32F, 1, 0)
    grad_y = cv2.Sobel(new_fused, cv2.CV_32F, 0, 1)

    # Calculate the magnitude of the gradients
    grad_mag = cv2.magnitude(grad_x, grad_y)

    # Calculate the average gradient
    avg_grad = np.mean(grad_mag)
    AVG_grad = "{:.4f}".format(avg_grad)
    AVG_grad.config(text="" + str(AVG_grad))



    # Calculate the histograms of the CT and MRI images
    ct_hist = cv2.calcHist([new_ct_resize], [0], None, [256], [0, 256])
    ct_hist_normalized = ct_hist / np.sum(ct_hist)
    mri_hist = cv2.calcHist([new_mri_resize], [0], None, [256], [0, 256])
    mri_hist_normalized = mri_hist / np.sum(mri_hist)

    fused_hist = cv2.calcHist([new_fused], [0], None, [256], [0, 256])
    fused_hist_normalized = fused_hist / np.sum(fused_hist)
    cross_entropy_ct_fused = 0.0
    cross_entropy_mri_fused = 0.0
    for i in range(256):
        if ct_hist_normalized[i] > 0 and fused_hist_normalized[i] > 0:
            cross_entropy_ct_fused -= ct_hist_normalized[i] * np.log2(fused_hist_normalized[i] + 1e-7)
        if mri_hist_normalized[i] > 0 and fused_hist_normalized[i] > 0:
            cross_entropy_mri_fused -= mri_hist_normalized[i] * np.log2(fused_hist_normalized[i] + 1e-7)

    cr_Entropy_CT = "{:.4f}".format(cross_entropy_ct_fused)
    cr_Entropy_CT.config(text="" + str(cr_Entropy_CT))
    cr_Entropy_MRI = "{:.4f}".format(cross_entropy_ct_fused)
    cr_Entropy_MRI.config(text="" + str(cr_Entropy_MRI))

    histogram = cv2.calcHist([new_fused], [0], None, [256], [0, 256])
    histogram_normalized = histogram / np.sum(histogram)
    # plt.plot(histogram)
    # plt.show()
    entropy = -np.sum(histogram_normalized * np.log2(histogram_normalized + 1e-7))
    Entropy = "{:.4f}".format(entropy)
    Entropy.config(text="" + str(Entropy))'''


def clear_all():
    # Clear the image
    lb5.config(image=None)
    lb5.image = None

    lb6.config(image=None)
    lb6.image = None

    lb7.config(image=None)
    lb7.image = None
    lb16.config(text="")

    SSIM_val.config(text="")
    PSNR_val.config(text="")
    Mean.config(text="")
    Entropy.config(text="")
    STD_dev.config(text="")
    MSE.config(text="")
    NRMSE.config(text="")
    AVG_grad.config(text="")
    cr_Entropy_CT.config(text="")
    cr_Entropy_MRI.config(text="")

def Exit():
    win.destroy()
    win.quit()



# Button bind and over
def on_enter1(p):
    btn1.config(bg='red', fg='black')


def on_enter2(p):
    btn2.config(bg='red', fg='black')


def on_enter3(p):
    btn3.config(bg='red', fg='black')


# def on_enter4(p):
# btn4.config(bg='red', fg='black')


def on_enter5(m):
    btn5.config(bg="red", fg="black")


def on_enter6(n):
    btn6.config(bg="red", fg="black")


def on_enter7(n):
    btn7.config(bg="red", fg="black")


def on_enter8(n):
    btn8.config(bg="red", fg="black")


def on_leave1(p):
    btn1.config(bg='light slate blue', fg='black')


def on_leave2(p):
    btn2.config(bg='light slate blue', fg='black')


def on_leave3(p):
    btn3.config(bg='light slate blue', fg='black')


# def on_leave4(p):
# btn4.config(bg='light slate blue', fg='black')


def on_leave5(m):
    btn5.config(bg="light slate blue", fg='black')


def on_leave6(n):
    btn6.config(bg="light slate blue", fg='black')


def on_leave7(n):
    btn7.config(bg="light slate blue", fg='black')


def on_leave8(n):
    btn8.config(bg="light slate blue", fg='black')


lb1 = tk.Label(win, text='Data fusion Using ML  And An Aproach For Image Classification', width=150, font=('bold', 20),
               borderwidth=2,
               relief="solid", )
lb1.pack()

lb2 = tk.Label(win, text='Menu', width=8, borderwidth=2, relief="solid", )
lb2.place(x=50, y=50)

lb3 = tk.Label(win, text='', width=26, height=37, borderwidth=2, relief="solid")
lb3.place(x=51, y=70, )
lb3.config(bg='white')

# blank background widget of images
lb4 = tk.Label(win, text='', width=160, height=22, borderwidth=2, relief="solid")
lb4.place(x=290, y=70)

lbseg = tk.Label()
lbseg.place(x=1170, y=105)

# Create a Frame  border input CT image
border_color = tk.Frame(win, background="black")

# Label Widget inside the Frame
label_1 = tk.Label(border_color, width=30, height=14, bd=0)

# Place the widgets with border Frame
label_1.pack(padx=2, pady=2)
border_color.place(x=330, y=95)

# Create a Frame  border for MRI image
border_color = tk.Frame(win, background="black")

# Label Widget inside the Frame
label_2 = tk.Label(border_color, width=30, height=14, bd=0)

# Place the widgets with border Frame for preprocessed image
label_2.pack(padx=2, pady=2)
border_color.place(x=605, y=95)

# Create a Frame  border for Fused  image
border_color = tk.Frame(win, background="black")
# Label Widget inside the Frame for
label_3 = tk.Label(border_color, width=30, height=14, bd=0)

# Place the widgets with border Frame
label_3.pack(padx=2, pady=2)
border_color.place(x=880, y=95)

# Create a Frame  border for segmented fused  image
border_color = tk.Frame(win, background="black")
# Label Widget inside the Frame for
# label_4 = tk.Label(border_color, width=30, height=14, bd=0)

# Place the widgets with border Frame
# label_4.pack(padx=2, pady=2)
# border_color.place(x=1155, y=95)


lb5 = tk.Label()
lb5.place(x=345, y=105)

lb6 = tk.Label()
lb6.place(x=625, y=105)

lb7 = tk.Label()
lb7.place(x=895, y=105)

lb8 = tk.Label(win, text='CT Image', font=('bold', 13), borderwidth=2, relief="solid")
lb8.place(x=390, y=330)

lb9 = tk.Label(win, text='MRI Image', font=('bold', 13), borderwidth=2, relief="solid")
lb9.place(x=670, y=330)

lb10 = tk.Label(win, text='Fused Image', font=('bold', 13), borderwidth=2, relief="solid")
lb10.place(x=940, y=330)

# segment = tk.Label(win, text='segmented Image', font=('bold', 13), borderwidth=2, relief="solid")
# segment.place(x=1200, y=330)

lb11 = tk.Label(win, text='', width=35, height=5, borderwidth=2, relief="solid")
lb11.place(x=290, y=440)

lb12 = tk.Label(win, text='', width=122, height=5, borderwidth=2, relief="solid")
lb12.place(x=560, y=440)

lb13 = tk.Label(win, text='Image Display', width=25, height=1, font=('bold', 15), borderwidth=2, relief="solid")
lb13.place(x=700, y=51)

lb14 = tk.Label(win, text='TUMOR(YES/NO)', font=('calibri', 14, 'bold'), borderwidth=2, relief="solid")
lb14.place(x=350, y=430)

lb15 = tk.Label(win, text='Analysis', font=('calibri', 14, 'bold'), borderwidth=2, relief="solid")
lb15.place(x=560, y=425)

lb16 = tk.Label(win, text="", bg='gray', width=16, font=('calibri', 17, 'bold'), borderwidth=2, relief="solid")
lb16.place(x=320, y=465)

lb17 = tk.Label(win, text='SSIM', font=('calibri', 14, 'bold'), borderwidth=2, relief="solid")
lb17.place(x=640, y=450)

lb18 = tk.Label(win, text='PSNR', font=('calibri', 14, 'bold'), borderwidth=2, relief="solid")
lb18.place(x=860, y=450)

lb19 = tk.Label(win, text='Mean', font=('calibri', 14, 'bold'), borderwidth=2, relief="solid")
lb19.place(x=1040, y=450)
lb20 = tk.Label(win, text='Entropy', font=('calibri', 14, 'bold'), borderwidth=2, relief="solid")
lb20.place(x=1240, y=450)
Mean = tk.Label(win, text='', width=19, height=2, bg='dodger blue', font=('calibri', 10, 'bold'), borderwidth=2,
                relief="solid")
Mean.place(x=1000, y=480)

PSNR_val = tk.Label(win, text='', width=19, height=2, bg='dodger blue', font=('calibri', 10, 'bold'), borderwidth=2,
                    relief="solid")
PSNR_val.place(x=810, y=480)

SSIM_val = tk.Label(win, text='', width=19, height=2, bg='dodger blue', font=('calibri', 10, 'bold'), borderwidth=2,
                    relief="solid")
SSIM_val.place(x=600, y=480)

Entropy = tk.Label(win, text='', width=19, height=2, bg='dodger blue', font=('calibri', 10, 'bold'), borderwidth=2,
                   relief="solid")
Entropy.place(x=1200, y=480)

# background widget for fusion evaluation parameters
lb21 = tk.Label(win, text='', width=160, height=5, borderwidth=2, relief="solid")
lb21.place(x=290, y=550)

lb22 = tk.Label(win, text='Analysis', font=('calibri', 14, 'bold'), borderwidth=2, relief="solid")
lb22.place(x=700, y=530)
# PRINT EVALUATION PARAMETERS FOR FUSION

lb23 = tk.Label(win, text='STD.Dev', font=('calibri', 14, 'bold'), borderwidth=2, relief="solid")
lb23.place(x=380, y=560)
lb24 = tk.Label(win, text='MSE', font=('calibri', 14, 'bold'), borderwidth=2, relief="solid")
lb24.place(x=580, y=560)

lb25 = tk.Label(win, text='NRMSE', font=('calibri', 14, 'bold'), borderwidth=2, relief="solid")
lb25.place(x=740, y=560)
lb26 = tk.Label(win, text='AVG.Grad', font=('calibri', 14, 'bold'), borderwidth=2, relief="solid")
lb26.place(x=920, y=560)

lb27 = tk.Label(win, text='cr_Entropy_CT', font=('calibri', 14, 'bold'), borderwidth=2, relief="solid")
lb27.place(x=1082, y=560)
lb28 = tk.Label(win, text='cr_Entropy_MRI', font=('calibri', 14, 'bold'), borderwidth=2, relief="solid")
lb28.place(x=1260, y=560)

STD_dev = tk.Label(win, text='', width=17, height=2, bg='dodger blue', font=('calibri', 10, 'bold'), borderwidth=2,
                   relief="solid")
STD_dev.place(x=360, y=590)

MSE = tk.Label(win, text='', width=17, height=2, bg='dodger blue', font=('calibri', 10, 'bold'), borderwidth=2,
               relief="solid")
MSE.place(x=540, y=590)

NRMSE = tk.Label(win, text='', width=17, height=2, bg='dodger blue', font=('calibri', 10, 'bold'), borderwidth=2,
                 relief="solid")
NRMSE.place(x=720, y=590)

AVG_grad = tk.Label(win, text='', width=17, height=2, bg='dodger blue', font=('calibri', 10, 'bold'), borderwidth=2,
                    relief="solid")
AVG_grad.place(x=900, y=590)

cr_Entropy_CT = tk.Label(win, text='', width=17, height=2, bg='dodger blue', font=('calibri', 10, 'bold'),
                         borderwidth=2, relief="solid")
cr_Entropy_CT.place(x=1080, y=590)

cr_Entropy_MRI = tk.Label(win, text='', width=17, height=2, bg='dodger blue', font=('calibri', 10, 'bold'),
                          borderwidth=2, relief="solid")
cr_Entropy_MRI.place(x=1260, y=590)

# Create a CT button
btn1 = tk.Button(win, text='Browse CT Image', bg='light slate blue', fg='black', width=15, height=2, font=('bold', 10),
                 borderwidth=2, relief="solid", command=browse_ct_image)
btn1.place(x=85, y=120)
btn1.bind('<Enter>', on_enter1)
btn1.bind('<Leave>', on_leave1)
# ct_button = tk.Button(root, text="Browse CT Image", command=browse_ct_image)
# ct_button.pack()

# Create an MRI button
btn2 = tk.Button(win, text='Browse MRI Image', bg='light slate blue', fg='black', width=15, height=2, font=('bold', 10),
                 borderwidth=2, relief="solid", command=browse_mri_image)
btn2.place(x=85, y=180)
btn2.bind('<Enter>', on_enter2)
btn2.bind('<Leave>', on_leave2)
# mri_button = tk.Button(root, text="Browse MRI Image", command=browse_mri_image)
# mri_button.pack()

# Create a Fusion button

btn3 = tk.Button(win, text='Fused Image', bg='light slate blue', fg='black', width=15, height=2, font=('bold', 10),
                 borderwidth=2, relief="solid", command=generate_fused_image)
btn3.place(x=85, y=250)

btn3.bind('<Enter>', on_enter3)
btn3.bind('<Leave>', on_leave3)
# fusion_button = tk.Button(root, text="Fuse Images", command=generate_fused_image)
# fusion_button.pack()


# btn4 = tk.Button(win, text='Segmentation', bg='light slate blue', fg='black', width=15, height=2, font=('bold', 10),
# borderwidth=2, relief="solid" )
# btn4.place(x=85, y=300)

# btn4.bind('<Enter>', on_enter4)
# btn4.bind('<Leave>', on_leave4)

btn5 = tk.Button(win, text='Classify', bg='light slate blue', fg='black', width=15, height=2, font=('bold', 10),
                 borderwidth=2, relief="solid", command=classify)
btn5.place(x=85, y=350)

btn5.bind('<Enter>', on_enter5)
btn5.bind('<Leave>', on_leave5)

btn6 = tk.Button(win, text='Analysis', bg='light slate blue', fg='black', width=15, height=2, font=('bold', 10),
                 borderwidth=2, relief="solid", command=Analysis)
btn6.place(x=85, y=420)

btn6.bind('<Enter>', on_enter6)
btn6.bind('<Leave>', on_leave6)

btn7 = tk.Button(win, text='Clear All', bg='light slate blue', fg='black', width=15, height=2, font=('bold', 10),
                 borderwidth=2, relief="solid",command=clear_all )
btn7.place(x=85, y=480)
btn7.bind('<Enter>', on_enter7)
btn7.bind('<Leave>', on_leave7)

btn8 = tk.Button(win, text='Exit', bg='light slate blue', fg='black', width=15, height=2, font=('bold', 10),
                 borderwidth=2,
                 relief="solid",command=Exit)
btn8.place(x=85, y=540)
btn8.bind('<Enter>', on_enter8)
btn8.bind('<Leave>', on_leave8)

# Run the tkinter event loop
win.mainloop()

