from tkinter import *
import joblib 

# Load model
model = joblib.load("news_model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

# Function

def check():
    news = enter_text.get("1.0",END).strip()
    if news:
        out_put_news = vectorizer.transform([news])
        output = model.predict(out_put_news)[0]
        result = "Fake" if output == 0 else "No Fake"
        output_result.config(text = f"This is the {result} news")

    else:
        output_result.config(text = "Please enter any news")    

app = Tk()
app.geometry("1000x700+100+20")
app.title("Fake News Detecter")
app.config(bg = "#43F0E8")

heading = Label(app, text = "FAKE NEWS DETECTER", font = ("ROBOT", 35, "bold"), bg = "#43F0E8", fg = "blue")
heading.pack(fill = "x", pady = 20)

input_label = Label(app, text = "Enter News Details", font = ("ROBOT", 15, "bold"), bg = "#43F0E8", fg = "green")
input_label.place(x = 150, y = 115)

enter_text = Text(app, font = "arial 15 bold", width = 60, height = 10)
enter_text.place(x = 150, y = 150)


predict_btn = Button(app, text = "CHECK", font = "ROBOT 25 bold", bg = "#F043D2", fg = "white", bd = 4, command = check)
predict_btn.place(x = 380,y = 420)

# Output Label

output_result = Label(app, font = "arial 20 bold", bg = "#43F0E8", fg = "#E90E10")
output_result.place(x = 290, y = 530)


app.mainloop()