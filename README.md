# 🧠 Next Word Predictor using LSTM  
A deep learning project that predicts the **next word** in a sentence using an **LSTM-based Neural Network**.  
This project demonstrates text preprocessing, sequence generation, model training, and real-time prediction.

---

## 🚀 Features
- Trains an LSTM model on a custom text dataset  
- Uses Tokenizer for text preprocessing  
- Creates input-output sequences for next-word prediction  
- Embedding + LSTM + Dense architecture  
- Predicts the next word based on user input  
- Includes a Jupyter Notebook for clarity and experimentation  

---


## 📂 Project Structure

Next-Word-Predictor/
│
├── Next_Word_Predictor.ipynb # Main notebook
├── data.txt # Training dataset
├── README.md # Documentation
└── models/ # (Optional) saved models

---

## 🛠️ Technologies Used
- Python  
- TensorFlow / Keras  
- NumPy  
- Pandas  
- Jupyter Notebook  

---

## 📥 Installation

### 1️⃣ Clone the repository
bash
git clone https://github.com/KunjanMinama/Next-Word-Predictor.git
cd Next-Word-Predictor
2️⃣ Install dependencies
bash
Copy code
pip install tensorflow numpy pandas
📊 How It Works
1. Load Dataset
Reads text from data.txt

Converts text to lowercase

Removes unwanted characters

2. Tokenization
python
Copy code
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
3. Create sequences
Example:

kotlin
Copy code
deep learning is fun
becomes training sequences like:

csharp
Copy code
deep learning
deep learning is
4. Pad sequences
Ensures equal input length.

5. Build LSTM Model
python
Copy code
model = Sequential()
model.add(Embedding(vocab_size, 128, input_length=max_len))
model.add(LSTM(150, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(vocab_size, activation='softmax'))
6. Train the model
python
Copy code
model.fit(X, y, epochs=50, batch_size=64)
7. Predict next word
Input:

arduino
Copy code
"deep learning is"
Output:

arduino
Copy code
"powerful"
🧪 Example Prediction Code
python
Copy code
def predict_next_word(model, tokenizer, text, max_len):
    for _ in range(1):
        token_list = tokenizer.texts_to_sequences([text])[0]
        token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        next_index = np.argmax(predicted)
        for word, index in tokenizer.word_index.items():
            if index == next_index:
                return word
📈 Results
LSTM learns sentence patterns well

Predicts meaningful next words

Works better with larger datasets

💡 Future Improvements
Add Bidirectional LSTM

Replace LSTM with Transformer

Add GUI using Streamlit/Gradio

Train on large text corpora

🤝 Contributing
Pull requests are welcome.
If you find any issues, feel free to open an issue.

⭐ Support
If you like this project, please give it a star ⭐ on GitHub.
It motivates further development!

🧑‍💻 Author
Kunjan Minama
AI/ML Developer | Deep Learning | NLP

yaml
Copy code

---

# Want a **professional GitHub banner**, project logo, or badges (build/accuracy/stars)?  
I can generate and design them for you.






