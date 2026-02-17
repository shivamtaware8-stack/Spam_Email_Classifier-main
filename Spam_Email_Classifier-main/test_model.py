import pickle
from preprocessing import clean_text

# LOAD saved model
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

while True:
    email = input("Enter email text (type 'exit' to stop): ")

    if email.lower() == "exit":
        break

    email_cleaned = clean_text(email)
    email_vector = vectorizer.transform([email_cleaned])

    result = model.predict(email_vector)

    if result[0] == 1:
        print("🚫 Spam Email")
    else:
        print("✅ Not Spam (Ham)")
