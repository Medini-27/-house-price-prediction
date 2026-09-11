import os
import joblib
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = "models/house_price_model.pkl"
METRICS_PATH = "models/metrics.pkl"


if not os.path.exists(MODEL_PATH):

    st.error(
        "Model not found. Please run train_model.py first."
    )

    st.stop()


model = joblib.load(MODEL_PATH)


if os.path.exists(METRICS_PATH):
    metrics = joblib.load(METRICS_PATH)
else:
    metrics = None


# ============================================================
# HEADER
# ============================================================

st.title("🏠 House Price Prediction System")

st.markdown(
    """
    ### Machine Learning Based Property Price Prediction

    Enter the property details in the sidebar and the
    trained machine-learning model will estimate the
    property price.
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🏡 Property Details")


city = st.sidebar.selectbox(
    "City",
    [
        "Mumbai",
        "Hyderabad",
        "Chennai",
        "Pune",
        "Delhi NCR",
        "Bangalore"
    ]
)


# ============================================================
# LOCALITIES
# ============================================================

localities = {

    "Mumbai": [
        "Bandra",
        "Andheri West"
    ],

    "Hyderabad": [
        "Gachibowli",
        "Hitech City"
    ],

    "Chennai": [
        "Velachery"
    ],

    "Pune": [
        "Kharadi"
    ],

    "Delhi NCR": [
        "Gurgaon Sec 56",
        "Noida Sec 62"
    ],

    "Bangalore": [
        "Whitefield",
        "Koramangala"
    ]
}


locality = st.sidebar.selectbox(
    "Locality",
    localities[city]
)


# ============================================================
# PROPERTY INPUTS
# ============================================================

bhk = st.sidebar.slider(
    "BHK",
    min_value=1,
    max_value=5,
    value=2
)


area = st.sidebar.number_input(
    "Area (Sq Ft)",
    min_value=300,
    max_value=5000,
    value=1000,
    step=50
)


bathrooms = st.sidebar.slider(
    "Bathrooms",
    min_value=1,
    max_value=5,
    value=2
)


balconies = st.sidebar.slider(
    "Balconies",
    min_value=0,
    max_value=5,
    value=2
)


furnishing = st.sidebar.selectbox(
    "Furnishing",
    [
        "Unfurnished",
        "Semi-Furnished",
        "Fully Furnished"
    ]
)


property_age = st.sidebar.slider(
    "Property Age (Years)",
    min_value=0,
    max_value=30,
    value=5
)


# ============================================================
# MAIN LAYOUT
# ============================================================

col1, col2 = st.columns(2)


# ============================================================
# PROPERTY SUMMARY
# ============================================================

with col1:

    st.subheader("📋 Property Summary")

    summary = pd.DataFrame({

        "Feature": [
            "City",
            "Locality",
            "BHK",
            "Area",
            "Bathrooms",
            "Balconies",
            "Furnishing",
            "Property Age"
        ],

        "Value": [
            city,
            locality,
            bhk,
            f"{area:,} Sq Ft",
            bathrooms,
            balconies,
            furnishing,
            f"{property_age} Years"
        ]
    })

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PRICE PREDICTION
# ============================================================

with col2:

    st.subheader("💰 Price Prediction")

    st.write(
        "Click the button to estimate the property price."
    )

    predict = st.button(
        "🔮 Predict House Price",
        type="primary",
        use_container_width=True
    )


    if predict:

        input_data = pd.DataFrame({

            "City": [city],

            "Locality": [locality],

            "BHK": [bhk],

            "Area_SqFt": [area],

            "Bathrooms": [bathrooms],

            "Balconies": [balconies],

            "Furnishing": [furnishing],

            "Property_Age_Yrs": [property_age]
        })


        prediction = model.predict(
            input_data
        )[0]


        # Lakhs → Crores

        price_crore = prediction / 100


        # Price per square foot

        price_per_sqft = (
            prediction * 100000
        ) / area


        st.success(
            "Prediction completed successfully!"
        )


        st.metric(
            "Estimated Property Price",
            f"₹{prediction:.2f} Lakhs"
        )


        c1, c2 = st.columns(2)


        with c1:

            st.metric(
                "Price in Crores",
                f"₹{price_crore:.2f} Cr"
            )


        with c2:

            st.metric(
                "Price / Sq Ft",
                f"₹{price_per_sqft:,.0f}"
            )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.divider()

st.subheader("📊 Model Performance")


if metrics:

    c1, c2, c3 = st.columns(3)


    with c1:

        st.metric(
            "MAE",
            f"₹{metrics['MAE']:.2f} Lakhs"
        )


    with c2:

        st.metric(
            "RMSE",
            f"₹{metrics['RMSE']:.2f} Lakhs"
        )


    with c3:

        st.metric(
            "R² Score",
            f"{metrics['R2']:.4f}"
        )


    st.info(
        f"""
        The model achieved an R² score of
        **{metrics['R2']:.4f}**.

        This means the model explains approximately
        **{metrics['R2']:.2%}** of the variation in
        property prices in the test dataset.
        """
    )


# ============================================================
# MACHINE LEARNING INFORMATION
# ============================================================

st.divider()

st.subheader("🤖 Machine Learning Information")

st.write(
    """
    **Algorithm:** Random Forest Regression

    **Dataset:** 250 residential properties

    **Input Features:**

    • City

    • Locality

    • BHK

    • Area in Sq Ft

    • Bathrooms

    • Balconies

    • Furnishing

    • Property Age

    **Target:** Property Price in Lakhs

    **Categorical Encoding:** One-Hot Encoding

    **Missing Value Handling:** Imputation
    """
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "House Price Prediction System | Machine Learning Project"
)
