from tkinter import *
import numpy as np
import pandas as pd
import joblib

# load model
model = joblib.load("model.joblib")
vectrorizer = joblib.load("vectorizer.joblib")



# function 

def check():
    email = email_entry.get("1.0", END).strip()
    if email:
        message_vectorized = vectrorizer.transform([email])
        prediction = model.predict(message_vectorized)[0]
        result =  "Spam" if prediction == 1 else "Non-Spam"
        output_label.config(text = f"This email is a : {result}")

    else:
        output_label.config(text = "Please enter an email")   


app = Tk()
app.geometry("900x600+60+40")
app.title("Email Spam Checker")
app.config(bg = "green")

heading = Label(app, text = "Email Spam Detecter", font = ("Robot", 30, "bold"), bg = "green", fg = "white")
heading.pack(fill = "x", pady = 30)

email_label = Label(app, text = "Enter Email:", font = ("Robot", 17, "bold"), fg = "black", bg = "green")
email_label.place(x = 120, y = 115)


email_entry = Text(app, font = ("Robot", 17, "bold"), width = 50,height = 8, bd = 6)
email_entry.pack(pady = 40)

email_btn = Button(app, text = "Check", font = ("Robot", 20, "bold"),bd = 5, justify = "center", width = 46, bg = "red", fg = "white", cursor = "hand2", command =check)
email_btn.place(x = 50, y = 410)


# output_label 

output_label = Label(app, font = ("Robot", 17, "bold"), bg = "green", fg = "white")
output_label.place(x = 300, y = 490)

app.mainloop()


