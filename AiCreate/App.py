import streamlit as st
import machinelearning
import Detail
import Web

# Set page config
st.set_page_config(page_title="AI-Powered Insights", layout="wide")

# Custom CSS for dark theme and styling with fixed box height
st.markdown("""
    <style>
        body {
            background-color: #0e0e0e;
            color: white;
        }
        .header {
            font-size: 24px;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 1.5px;
        }
        .navbar {
            display: flex;
            justify-content: space-between;
            border-bottom: 1px solid #333;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        .nav-links a {
            color: #aaa;
            text-decoration: none;
            margin: 0 15px;
            font-size: 18px;
        }
        .nav-links a:hover {
            color: cyan;
        }
        .title-container {
            display: flex;
            justify-content: flex-start;
            align-items: center;
            height: 50vh;
            padding-left: 50px;
        }
        .title {
            font-size: 60px;
            font-weight: bold;
            text-align: left;
        }
        .highlight {
            color: cyan;
        }
        .button-container {
            margin-top: 10px;
        }
        .button {
            background: cyan;
            padding: 8px 15px;
            color: black;
            border-radius: 5px;
            text-decoration: none;
            font-weight: bold;
            display: inline-block;
        }

        /* Styling for the boxed text with fixed height */
        .box {
            background: linear-gradient(45deg, #4CAF50, #2196F3); /* Gradient from green to blue */
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            height: 300px; /* Fixed height */
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }

        /* Ensure the columns have equal height */
        .row-equal-height {
            display: flex;
            justify-content: space-between;
            height: 100%;
        }

        .column-equal-height {
            display: flex;
            flex-direction: column;
            flex-grow: 1;
        }

        /* Add some padding to prevent overlap of content */
        .content-padding {
            padding: 15px;
        }
    </style>
""", unsafe_allow_html=True)

# Navbar Navigation
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<div class="header">FINAL PROJECT IS</div>', unsafe_allow_html=True)
with col2:
    page = st.selectbox("เลือกหน้า", ["Home", "Development Approach", "Machine Learning", "Neural Network"], label_visibility="collapsed")

st.markdown('<hr style="border: 1px solid #333;">', unsafe_allow_html=True)

# Main Section
if page == "Home":
    # Use columns to align text and image together
    col1, col2 = st.columns([2, 1])  # Ratio 2:1 to give more space to text
    
    with col1:
        st.markdown('<div class="title-container"><p class="title"><h1>AI-Powered <span class="highlight">Insights</span><br> Simplified</h1></p></div>', unsafe_allow_html=True)

    with col2:
        st.image("/Users/xobazjr/Documents/GitHub/web-application-is/assets/images/robot-ai.png", width=550) # Update path to correct location of image
    
    # Data Preparation Section
    st.markdown("## Data Preparation")

    st.markdown('<hr style="border: 1px solid #333;">', unsafe_allow_html=True)
    
    # Create two columns for the content boxes to be displayed side by side
    col1, col2 = st.columns(2)

    # Using the 'row-equal-height' class to ensure equal heights
    with col1:
        st.markdown('<div class="column-equal-height">', unsafe_allow_html=True)
        st.markdown("""
            <div class="box">
                <h3>Dataset Selection & Collection</h3>
                <p>- Use synthetic data or real data from rental records.</p>
                <p>- Possible data sources.</p>
                <p>- Open datasets from Kaggle, UCI Machine Learning Repository, etc.</p>
                <p>- Possible data sources: Generated using ChatGPT or Python (Pandas, NumPy).</p>
            </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="column-equal-height">', unsafe_allow_html=True)
        st.markdown("""
            <div class="box">
                <h3>Data Cleaning & Preprocessing</h3>
                <p>- Handling Missing Values: Use mean/median for numerical values and mode for categorical data.</p>
                <p>- Encoding Categorical Data: Apply One-Hot Encoding or Label Encoding to convert text data into numerical form.</p>
                <p>- Removing Outliers: Use IQR (Interquartile Range) or Z-score to detect and remove anomalies.</p>
            </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

elif page == "Development Approach":
    Detail.show()

elif page == "Machine Learning":
    machinelearning.show()

elif page == "Neural Network":
    Web.show()
