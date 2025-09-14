import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Iris Dataset Analysis",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #A23B72;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🌸 Iris Dataset Analysis & ML Classification</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Choose a page", 
                           ["Dataset Overview", "Data Visualization", "Machine Learning Models", "Model Comparison"])

# Load data
@st.cache_data
def load_data():
    # For demo purposes, create sample Iris data if CSV not available
    try:
        df = pd.read_csv('Iris.csv')
        if 'Id' in df.columns:
            df = df.drop(columns=['Id'])
    except FileNotFoundError:
        # Create sample Iris data
        from sklearn.datasets import load_iris
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])
        df['Species'] = iris.target_names[iris.target]
    return df

df = load_data()

# Dataset Overview Page
if page == "Dataset Overview":
    st.markdown('<h2 class="sub-header">📊 Dataset Overview</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Info")
        st.write(f"**Shape:** {df.shape}")
        st.write(f"**Features:** {list(df.columns[:-1])}")
        st.write(f"**Target:** Species")
        
        st.subheader("First 5 Rows")
        st.dataframe(df.head())
        
    with col2:
        st.subheader("Statistical Summary")
        st.dataframe(df.describe())
        
        st.subheader("Species Distribution")
        species_counts = df['Species'].value_counts()
        st.bar_chart(species_counts)
    
    # Missing values check
    st.subheader("Data Quality Check")
    missing_values = df.isnull().sum()
    if missing_values.sum() == 0:
        st.success("✅ No missing values found in the dataset!")
    else:
        st.warning("⚠️ Missing values detected:")
        st.write(missing_values)
    
    # Correlation matrix
    st.subheader("Feature Correlation Matrix")
    numeric_df = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0, ax=ax)
    plt.title("Feature Correlation Heatmap")
    st.pyplot(fig)

# Data Visualization Page
elif page == "Data Visualization":
    st.markdown('<h2 class="sub-header">📈 Data Visualization</h2>', unsafe_allow_html=True)
    
    # Feature distributions
    st.subheader("Feature Distributions")
    features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    
    col1, col2 = st.columns(2)
    with col1:
        for i in range(0, len(features), 2):
            fig, ax = plt.subplots(figsize=(8, 5))
            df[features[i]].hist(bins=20, alpha=0.7, ax=ax)
            ax.set_title(f'{features[i]} Distribution')
            ax.set_xlabel(features[i])
            ax.set_ylabel('Frequency')
            st.pyplot(fig)
    
    with col2:
        for i in range(1, len(features), 2):
            if i < len(features):
                fig, ax = plt.subplots(figsize=(8, 5))
                df[features[i]].hist(bins=20, alpha=0.7, ax=ax)
                ax.set_title(f'{features[i]} Distribution')
                ax.set_xlabel(features[i])
                ax.set_ylabel('Frequency')
                st.pyplot(fig)
    
    # Scatter plots
    st.subheader("Species Comparison - Scatter Plots")
    
    colors = ['red', 'orange', 'blue']
    species = df['Species'].unique()
    
    scatter_options = [
        ('SepalLengthCm', 'SepalWidthCm', 'Sepal Length vs Sepal Width'),
        ('PetalLengthCm', 'PetalWidthCm', 'Petal Length vs Petal Width'),
        ('SepalLengthCm', 'PetalLengthCm', 'Sepal Length vs Petal Length'),
        ('SepalWidthCm', 'PetalWidthCm', 'Sepal Width vs Petal Width')
    ]
    
    for x_col, y_col, title in scatter_options:
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, sp in enumerate(species):
            species_data = df[df['Species'] == sp]
            ax.scatter(species_data[x_col], species_data[y_col], 
                      c=colors[i % len(colors)], label=sp, alpha=0.7, s=50)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

# Machine Learning Models Page
elif page == "Machine Learning Models":
    st.markdown('<h2 class="sub-header">🤖 Machine Learning Models</h2>', unsafe_allow_html=True)
    
    # Prepare data
    df_ml = df.copy()
    le = LabelEncoder()
    df_ml['Species'] = le.fit_transform(df_ml['Species'])
    
    X = df_ml.drop(columns=['Species'])
    y = df_ml['Species']
    
    # Model selection
    st.subheader("Model Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider("Test Size (%)", 10, 50, 30, 5) / 100
        random_state = st.number_input("Random State", 0, 100, 42)
        
    with col2:
        selected_model = st.selectbox("Choose Model", 
                                    ["Logistic Regression", "Decision Tree", "Random Forest", "SVM"])
    
    # Split data
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Train selected model
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            if selected_model == "Logistic Regression":
                model = LogisticRegression(random_state=random_state)
            elif selected_model == "Decision Tree":
                model = DecisionTreeClassifier(random_state=random_state)
            elif selected_model == "Random Forest":
                model = RandomForestClassifier(random_state=random_state)
            else:  # SVM
                model = SVC(random_state=random_state)
            
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test) * 100
            
            # Display results
            st.success(f"Model trained successfully!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Accuracy</h3>
                    <h2>{accuracy:.2f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Training Size</h3>
                    <h2>{len(x_train)}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Test Size</h3>
                    <h2>{len(x_test)}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Predictions and detailed metrics
            y_pred = model.predict(x_test)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
                ax.set_title('Confusion Matrix')
                ax.set_ylabel('True Label')
                ax.set_xlabel('Predicted Label')
                st.pyplot(fig)
            
            with col2:
                st.subheader("Classification Report")
                report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)

# Model Comparison Page
elif page == "Model Comparison":
    st.markdown('<h2 class="sub-header">⚖️ Model Comparison</h2>', unsafe_allow_html=True)
    
    # Prepare data
    df_ml = df.copy()
    le = LabelEncoder()
    df_ml['Species'] = le.fit_transform(df_ml['Species'])
    
    X = df_ml.drop(columns=['Species'])
    y = df_ml['Species']
    
    test_size = st.slider("Test Size for Comparison (%)", 10, 50, 30, 5) / 100
    random_state = st.number_input("Random State for Comparison", 0, 100, 42)
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    if st.button("Compare All Models"):
        with st.spinner("Training all models..."):
            models = {
                "Logistic Regression": LogisticRegression(random_state=random_state),
                "Decision Tree": DecisionTreeClassifier(random_state=random_state),
                "Random Forest": RandomForestClassifier(random_state=random_state),
                "SVM": SVC(random_state=random_state)
            }
            
            results = {}
            
            for name, model in models.items():
                model.fit(x_train, y_train)
                accuracy = model.score(x_test, y_test) * 100
                results[name] = accuracy
            
            # Display comparison
            st.subheader("Model Accuracy Comparison")
            
            # Create comparison dataframe
            comparison_df = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy'])
            comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
            
            # Display as table
            st.dataframe(comparison_df.style.format({'Accuracy': '{:.2f}%'}))
            
            # Bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(comparison_df['Model'], comparison_df['Accuracy'], 
                         color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
            ax.set_ylabel('Accuracy (%)')
            ax.set_title('Model Accuracy Comparison')
            ax.set_ylim(0, 105)
            
            # Add value labels on bars
            for bar, accuracy in zip(bars, comparison_df['Accuracy']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{accuracy:.1f}%', ha='center', va='bottom')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Best model highlight
            best_model = comparison_df.iloc[0]
            st.success(f"🏆 Best Model: **{best_model['Model']}** with {best_model['Accuracy']:.2f}% accuracy")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Iris Dataset Analysis & Classification")