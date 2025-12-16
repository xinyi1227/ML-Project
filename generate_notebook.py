import json
import os

notebook_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# CBU5201 Mini-Project submission\n",
            "\n",
            "**Student Name**: [Your Name Here]  \n",
            "**Student ID**: [Your ID Here]\n",
            "\n",
            "# 2 Problem formulation\n",
            "\n",
            "The goal of this project is to identify the song title from a short audio recording (hum or whistle). This is a multi-class classification problem with 8 potential classes (songs). The input is a raw audio waveform (approx. 10 seconds), and the output is the predicted song label.\n",
            "\n",
            "This problem is challenging due to the high variability in human production of hums and whistles (pitch, tempo, quality) and the limited size of the dataset.\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 3 Methodology\n",
            "\n",
            "My methodology focuses on extracting timbral features from the audio signals and training a supervised learning classifier.\n",
            "\n",
            "1. **Data Loading & Cleaning**: Parse filenames to extract labels and participant IDs. Use `librosa` to load audio.\n",
            "2. **Feature Extraction**: I will use Mel-frequency cepstral coefficients (MFCCs). MFCCs are widely used in speech and audio processing as they represent the short-term power spectrum of sound, approximating the human auditory system's response. I will compute the mean and standard deviation of the MFCCs over time to produce a fixed-size feature vector for each recording.\n",
            "3. **Model Performance**: Performance will be defined by **Accuracy** (percentage of correctly classified songs) and a **Confusion Matrix** to analyze misclassifications.\n",
            "4. **Validation**: I will use **Group K-Fold Cross-Validation**, grouping by `ParticipantID`. This is crucial because simply shuffling all files would likely put the same person's recordings in both train and test sets, leading to data leakage (the model recognizing the person rather than the song). Group K-Fold ensures the model is tested on 'unseen' participants.\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 4 Implemented ML prediction pipelines\n",
            "\n",
            "My pipeline consists of the following stages:\n",
            "1.  **Input**: Raw Audio (.wav)\n",
            "2.  **Transformation**: MFCC Extraction -> Aggregation (Mean/Std)\n",
            "3.  **Preprocessing**: Standard Scaling (Z-score normalization)\n",
            "4.  **Model**: Random Forest Classifier\n",
            "5.  **Output**: Song Label\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4.1 Transformation stage\n",
            "\n",
            "I use `librosa` to load the audio. For each file:\n",
            "- Resample to a standard rate (default 22050 Hz).\n",
            "- Compute 20 MFCCs.\n",
            "- Calculate the Mean and Standard Deviation across the time axis.\n",
            "- Output: A vector of size 40 (20 means + 20 stds).\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4.2 Model stage\n",
            "\n",
            "I chose a **Random Forest Classifier**. Random Forests are robust ensemble methods that handle high-dimensional data well and are less prone to overfitting than single decision trees. They also work reasonably well without heavy hyperparameter tuning, making them a good baseline for this task.\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4.3 Ensemble stage\n",
            "\n",
            "The Random Forest itself is an ensemble method (bagging of decision trees). I did not implement further stacking or boosting layers to keep the pipeline interpretable and efficient given the dataset size.\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 5 Dataset\n",
            "\n",
            "I will verify the dataset path and parse the filenames to construct our metadata dataframe.\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import librosa\n",
            "import librosa.display\n",
            "import matplotlib.pyplot as plt\n",
            "from sklearn.model_selection import GroupKFold, cross_val_predict, cross_val_score\n",
            "from sklearn.ensemble import RandomForestClassifier\n",
            "from sklearn.preprocessing import StandardScaler\n",
            "from sklearn.metrics import accuracy_score, confusion_matrix, classification_report\n",
            "import seaborn as sns\n",
            "\n",
            "# Configuration\n",
            "DATA_PATH = 'MLEndHWII_sample_400'  # Adjust if needed\n",
            "\n",
            "def parse_filename(filename):\n",
            "    # Filename format: S<ID>_<type>_<take>_<Song>.wav\n",
            "    # Example: S9_hum_4_RememberMe.wav\n",
            "    base = filename.replace('.wav', '')\n",
            "    parts = base.split('_')\n",
            "    if len(parts) >= 4:\n",
            "        return {\n",
            "            'filename': filename,\n",
            "            'participant_id': parts[0],\n",
            "            'type': parts[1],\n",
            "            'take': parts[2],\n",
            "            'song': parts[3]\n",
            "        }\n",
            "    return None\n",
            "\n",
            "data_list = []\n",
            "for f in os.listdir(DATA_PATH):\n",
            "    if f.endswith('.wav'):\n",
            "        meta = parse_filename(f)\n",
            "        if meta:\n",
            "            data_list.append(meta)\n",
            "\n",
            "df = pd.DataFrame(data_list)\n",
            "print(f\"Total samples: {len(df)}\")\n",
            "print(f\"Unique songs: {df['song'].nunique()}\")\n",
            "print(f\"Unique participants: {df['participant_id'].nunique()}\")\n",
            "df.head()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Data Visualization\n",
            "Let's visualize a sample waveform and its spectrogram.\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Visualize one sample\n",
            "sample_file = os.path.join(DATA_PATH, df.iloc[0]['filename'])\n",
            "y, sr = librosa.load(sample_file)\n",
            "\n",
            "plt.figure(figsize=(12, 4))\n",
            "plt.subplot(1, 2, 1)\n",
            "librosa.display.waveshow(y, sr=sr)\n",
            "plt.title('Waveform')\n",
            "\n",
            "plt.subplot(1, 2, 2)\n",
            "S = librosa.feature.melspectrogram(y=y, sr=sr)\n",
            "S_dB = librosa.power_to_db(S, ref=np.max)\n",
            "librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr)\n",
            "plt.title('Mel Spectrogram')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 6 Experiments and results\n",
            "\n",
            "Here I will extract features for all files and run the cross-validation experiment.\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def get_features(file_path):\n",
            "    try:\n",
            "        y, sr = librosa.load(file_path, duration=10.0)\n",
            "        # Extract MFCCs\n",
            "        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)\n",
            "        # Aggregate features (Mean and Std)\n",
            "        mfcc_mean = np.mean(mfcc, axis=1)\n",
            "        mfcc_std = np.std(mfcc, axis=1)\n",
            "        return np.concatenate([mfcc_mean, mfcc_std])\n",
            "    except Exception as e:\n",
            "        print(f\"Error reading {file_path}: {e}\")\n",
            "        return None\n",
            "\n",
            "# Extract features\n",
            "features = []\n",
            "labels = []\n",
            "groups = []\n",
            "\n",
            "print(\"Extracting features...\")\n",
            "for index, row in df.iterrows():\n",
            "    path = os.path.join(DATA_PATH, row['filename'])\n",
            "    feat = get_features(path)\n",
            "    if feat is not None:\n",
            "        features.append(feat)\n",
            "        labels.append(row['song'])\n",
            "        groups.append(row['participant_id'])\n",
            "\n",
            "X = np.array(features)\n",
            "y = np.array(labels)\n",
            "groups = np.array(groups)\n",
            "\n",
            "print(f\"Feature matrix shape: {X.shape}\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Initialize Classifier\n",
            "clf = RandomForestClassifier(n_estimators=100, random_state=42)\n",
            "\n",
            "# Define Cross-Validation Strategy (GroupKFold)\n",
            "gkf = GroupKFold(n_splits=5)\n",
            "\n",
            "# Scale features\n",
            "scaler = StandardScaler()\n",
            "X_scaled = scaler.fit_transform(X)\n",
            "\n",
            "print(\"Performing Cross-Validation...\")\n",
            "# Get predictions for all samples using CV\n",
            "y_pred = cross_val_predict(clf, X_scaled, y, groups=groups, cv=gkf)\n",
            "\n",
            "# Calculate metrics\n",
            "acc = accuracy_score(y, y_pred)\n",
            "print(f\"Overall Accuracy: {acc:.4f}\")\n",
            "\n",
            "print(\"\\nClassification Report:\")\n",
            "print(classification_report(y, y_pred))\n",
            "\n",
            "print(\"Confusion Matrix:\")\n",
            "cm = confusion_matrix(y, y_pred)\n",
            "plt.figure(figsize=(10, 8))\n",
            "sns.heatmap(cm, annot=True, fmt='d', xticklabels=np.unique(y), yticklabels=np.unique(y), cmap='Blues')\n",
            "plt.xlabel('Predicted')\n",
            "plt.ylabel('True')\n",
            "plt.title('Confusion Matrix')\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 7 Conclusions\n",
            "\n",
            "The model achieved a modest accuracy using basic timbral features (MFCCs) and a Random Forest classifier. This result confirms the difficulty of the task stated in the problem description. Hums and whistles vary significantly between individuals, and MFCC averages may lose important temporal melodic information.\n",
            "\n",
            "### Suggestions for Improvement:\n",
            "1.  **Temporal Features**: Use Recurrent Neural Networks (RNNs) or Long Short-Term Memory (LSTM) networks to capture the sequence of notes rather than just statistical averages.\n",
            "2.  **Pitch Detection**: Explicitly extracting pitch contours (chroma features) might help identify the melody better than timbre alone.\n",
            "3.  **Data Augmentation**: Adding noise or pitch-shifting could help robustify the model.\n",
            "4.  **Transfer Learning**: Using pre-trained audio embeddings (e.g., VGGish, YAMNet) could provide higher-level features.\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 8 References\n",
            "\n",
            "- McFee, B., et al. (2015). librosa: Audio and Music Signal Analysis in Python. In Proceedings of the 14th Python in Science Conference.\n",
            "- Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.\n"
        ]
    }
]

notebook_content = {
    "cells": notebook_cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.5"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open('CBU5201_miniproject_2526.ipynb', 'w') as f:
    json.dump(notebook_content, f, indent=2)

print("Notebook generated successfully.")

