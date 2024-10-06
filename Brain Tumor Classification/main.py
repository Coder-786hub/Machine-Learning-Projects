import os
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import joblib

# Load the model
model = joblib.load("model.joblib")
file_path = None

# Prediction function
def predict():
    global file_path
    if file_path:
        # Open and process the image
        img = Image.open(file_path).convert("L")
        img_resized = img.resize((200, 200))
        img_array = np.array(img_resized)
        img_flat = img_array.flatten() / 255.0  # Normalize the image data
        img_flat = img_flat.reshape(1, -1)
        
        # Get model prediction
        output_array = model.predict(img_flat)
        output = output_array[0]

        # Set result label based on prediction
        if output == 0:
            predicted_value = "No Tumor"
        elif output == 1:
            predicted_value = "Positive Tumor"
        else:
            predicted_value = "Unknown"

        result_label.config(text=predicted_value)

        # Display the same image in a new label below the result label
        img_display = Image.open(file_path)
        img_resized_display = img_display.resize((300, 200))
        photo_display = ImageTk.PhotoImage(img_resized_display)
        image_display_label.config(image=photo_display)
        image_display_label.image = photo_display  # Keep a reference to avoid garbage collection

        # Reset the button text to "Upload MRI" for another upload
        btn.config(text="Upload MRI", command=showimage)

# Function to upload image
def showimage():
    global file_path
    file = filedialog.askopenfilename()
    if file:
        file_path = file
        img = Image.open(file)
        img_resized = img.resize((210, 210))
        photo = ImageTk.PhotoImage(img_resized)
        lbl2.config(image=photo)
        lbl2.image = photo  # Keep a reference to avoid garbage collection
        btn.config(text="Check Result", command=predict)

# GUI Setup
app = Tk()
app.geometry("800x800+0+0")
app.title("Brain Tumor Classifier")
app.config(bg="green")

# Title label
lbl = Label(app, text="Brain Tumor Classifier", font=("roboto", 30, "bold"), bg="black", fg="white", bd=7)
lbl.pack(fill="x", ipady=20)

# Frame for image display
frame1 = Frame(app, width=220, height=220, bd=3, bg="black", relief="groove")
frame1.pack(pady=10)

# Label for displaying the MRI image
lbl2 = Label(frame1, bg="black")
lbl2.place(x=0, y=0)

# Button to upload image
btn = Button(app, text="Upload MRI", font=("roboto", 20, "bold"), bd=2, bg="black", fg="white", command=showimage)
btn.pack(pady=30)

# Label for displaying the prediction result
result_label = Label(app, font=("roboto", 20, "bold"), bg="green")
result_label.pack(pady=10)

# New label to display the same image used for prediction (below result_label)
image_display_label = Label(app)
image_display_label.pack(ipady=5)

app.mainloop()
