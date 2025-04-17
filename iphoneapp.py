import streamlit as st
from PIL import Image, ImageStat
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="Mobile Data and Picture Analyzer", layout="centered")

st.markdown("""
    <style>
        .main {
            max-width: 700px;
            margin: auto;
        }
    </style>
    """, unsafe_allow_html=True)

st.title("Mobile Data and Picture Analyzer")

# --- Picture Analyzer Section ---
st.header("iPhone Photo Analyzer")

uploaded_files = st.file_uploader("Upload iPhone Photos", accept_multiple_files=True, type=["jpg", "jpeg", "png", "heic"])

if uploaded_files:
    photo_data = []
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        stat = ImageStat.Stat(image)
        info = {
            "Filename": uploaded_file.name,
            "Format": image.format,
            "Mode": image.mode,
            "Size": image.size,
            "Mean Brightness": round(sum(stat.mean) / len(stat.mean), 2),
        }
        photo_data.append(info)
        st.image(image, caption=uploaded_file.name, width=250)
        st.write(f"Resolution: {image.size[0]} x {image.size[1]}")
        st.write(f"Color Mode: {image.mode}")
        st.write("---")

    df_photos = pd.DataFrame(photo_data)
    st.subheader("Image Metadata Summary")
    st.dataframe(df_photos)

    st.subheader("Image Width vs Height")
    fig_photo, ax_photo = plt.subplots(figsize=(5, 4))
    for _, row in df_photos.iterrows():
        ax_photo.scatter(row["Size"][0], row["Size"][1], label=row["Filename"])
    ax_photo.set_xlabel("Width")
    ax_photo.set_ylabel("Height")
    ax_photo.set_title("Resolution Comparison")
    st.pyplot(fig_photo)

# --- Mobile Dataset Analysis Section ---
st.header("Mobile Dataset Analysis")

# Updated dataset path
df = pd.read_csv(r"C:\Users\lucky bohra\OneDrive\Desktop\Datasets\mobile_dataset.csv", encoding="ISO-8859-1")

st.subheader("Dataset Preview")
st.write(df.head())

label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

st.subheader("Target Column Selection")
target_column = st.selectbox("Select Target Column:", df.columns)
features = df.drop(columns=[target_column])
target = df[target_column]

st.subheader("Target Class Distribution")
target_counts = target.value_counts()
fig1, ax1 = plt.subplots(figsize=(5, 4))
sns.barplot(x=target_counts.index, y=target_counts.values, ax=ax1)
ax1.set_xlabel("Class")
ax1.set_ylabel("Count")
ax1.set_title(f"Target Column: {target_column}")
st.pyplot(fig1)

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

st.subheader("Feature Correlation Heatmap")
corr = df.corr()
fig2, ax2 = plt.subplots(figsize=(8, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax2)
ax2.set_title("Feature Correlation")
st.pyplot(fig2)

st.subheader("Model Training and Evaluation")
model_option = st.radio("Choose a model:", ("KNN", "Logistic Regression", "Decision Tree"))

if model_option == "KNN":
    k = st.slider("Select K for KNN:", 1, 15, 3)
    model = KNeighborsClassifier(n_neighbors=k)
elif model_option == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)
else:
    model = DecisionTreeClassifier()

model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

st.write(f"Accuracy: {acc:.2f}")
st.write("Classification Report")
st.dataframe(pd.DataFrame(report).transpose())

st.write("Confusion Matrix")
fig3, ax3 = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
ax3.set_xlabel("Predicted")
ax3.set_ylabel("Actual")
st.pyplot(fig3)

st.markdown("""
### Learnings:
- PIL was used for analyzing photo resolution and brightness.
- Applied machine learning models on a mobile dataset.
- Evaluated models using accuracy, confusion matrix, and classification report.
- Integrated everything in a single Streamlit interface.
""")
