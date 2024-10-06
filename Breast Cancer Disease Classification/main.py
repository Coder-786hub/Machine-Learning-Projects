from tkinter import *
from PIL import Image, ImageTk
import numpy as np
import joblib

# Loading model
model = joblib.load("model.joblib")

def predict():
    # Get input from entry field
    input_text = entryfield.get()  # Retrieve the text entered by the user

    # Convert the input to a numpy array
    input_features = np.fromstring(input_text, sep=',')  # Assuming input is comma-separated numbers
    
    # Predict the result using the model
    try:
        prediction = model.predict(input_features.reshape(1, -1))  # Reshape to match model input
        if prediction[0] == 1:
            output_label.config(text="Cancerous", fg="red")
            display_image("img1.jpeg")  # Display the cancerous image
        else:
            output_label.config(text="Not Cancerous", fg="green")
            display_image("img2.jpeg")  # Display the non-cancerous image
    except ValueError as e:
        output_label.config(text=f"Error: {e}", fg="yellow")

def display_image(image_path):
    """Display the image in the label"""
    img = Image.open(image_path)
    img = img.resize((300, 300), Image.Resampling.LANCZOS)  # Resize the image to fit in the GUI
    img = ImageTk.PhotoImage(img)
    image_label.config(image=img)
    image_label.image = img  # Keep a reference to avoid garbage collection


# Tkinter app setup
app = Tk()
app.geometry("1270x700+20+40")
app.title("Breast Cancer Predictor")
app.config(bg="black")

# Heading
heading = Label(app, text="Breast Cancer Prediction Model", font=("Roboto", 34, "bold"), fg="blue", bg="black")
heading.pack(pady=30)

# Input label and entry field
lbl = Label(app, text="Input Breast Cancer Features (comma-separated)", font=("Roboto", 15, "bold"), fg="white", bg="black")
lbl.place(x=100, y=120)

entryfield = Entry(app, width=100, font=(18))
entryfield.place(x=100, y=170)

# Predict button
predict_btn = Button(app, text="Predict", font=("Roboto", 25, "bold"), fg="white", bg="blue", command=predict)
predict_btn.place(x=100, y=220)

# Output label
output_label = Label(app, font=("Roboto", 25),bg="black")
output_label.place(x=100, y=290)

# Image label
image_label = Label(app, bg="black",pady = 20)
image_label.place(x = 500, y = 360)

# Main loop
app.mainloop()
