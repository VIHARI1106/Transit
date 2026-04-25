import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import re
import numpy as np

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Transit Insight Dashboard",
    layout="wide"
)

# ======================================================
# LOAD MODEL
# ======================================================
@st.cache_resource
def load_fake_model():
    model = joblib.load("models/fake_model.pkl")
    scaler = joblib.load("models/fake_scaler.pkl")
    return model, scaler


fake_model, fake_scaler = load_fake_model()

# ======================================================
# LOAD DATA
# ======================================================
sentiment_df = pd.read_csv("data/processed/sentiment_results.csv")
risk_df = pd.read_csv("data/processed/risk_results.csv")
peak_hours = pd.read_csv("data/processed/peak_hours.csv")
location_risk = pd.read_csv("data/processed/location_risk.csv")
driver_scores = pd.read_csv("data/processed/driver_scores.csv")
fake_df = pd.read_csv("data/processed/fake_detection_results.csv")
trend = pd.read_csv("data/processed/trend.csv")
category = pd.read_csv("data/processed/category.csv")

locations = sorted(fake_df["location"].dropna().unique())
drivers = sorted(fake_df["driver_id"].dropna().unique())

# ======================================================
# ADMIN SESSION
# ======================================================
if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False

# ======================================================
# SIDEBAR : USER + ADMIN
# ======================================================
main_tab = st.sidebar.radio(
    "Portal Access",
    ["User", "Admin"]
)

# ======================================================
# USER TAB
# ======================================================
if main_tab == "User":
    section = "Submit Review"

# ======================================================
# ADMIN TAB
# ======================================================
else:
    if not st.session_state.admin_logged_in:

        st.sidebar.subheader("🔐 Admin Login")

        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")

        if st.sidebar.button("Login"):
            if username == "admin" and password == "admin123":
                st.session_state.admin_logged_in = True
                st.success("Admin Login Successful")
                st.rerun()
            else:
                st.error("Invalid Credentials")

        st.title("🚖 Transit Review Portal")
        st.info(
            "Users can submit reviews from the USER tab.\n"
            "Admins must login to access analytics."
        )
        st.stop()

    else:
        st.sidebar.success("Logged in as Admin")

        if st.sidebar.button("Logout"):
            st.session_state.admin_logged_in = False
            st.rerun()

        section = st.sidebar.radio(
            "Navigation",
            [
                "Overview",
                "Model Comparison",
                "Analytics",
                "Location Analysis",
                "Driver Analysis",
                "Fake Detection"
            ]
        )

# ======================================================
# GLOBAL TITLE (ONLY ONCE)
# ======================================================
if section == "Submit Review":
    st.title("📝 Submit Review")
else:
    st.title("🚀 Transit Insight Dashboard")

# ======================================================
# OVERVIEW
# ======================================================
if section == "Overview":

    st.subheader("📊 Key Metrics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Reviews", len(fake_df))

    fake_counts = fake_df["fake_prediction"].value_counts()
    col2.metric("Fake Reviews", fake_counts.get("Fake", 0))

    col3.metric("Drivers", fake_df["driver_id"].nunique())
    col4.metric("Avg Rating", round(fake_df["rating"].mean(), 2))

    st.markdown("### 🙂 Sentiment Distribution")

    colA, colB = st.columns([1, 2])

    with colA:
        fig, ax = plt.subplots(figsize=(4, 4))
        fake_df["sentiment"].value_counts().plot.pie(
            autopct="%1.1f%%",
            ax=ax
        )
        ax.set_ylabel("")
        st.pyplot(fig)

    with colB:
        st.info("""
• Positive → Good service  
• Negative → Complaints  
• Neutral → Mixed feedback
        """)

    peak = peak_hours.sort_values(
        "review_count",
        ascending=False
    ).iloc[0]

    st.markdown("### 📊 Insights")
    st.write(
        f"• Peak hour: **{peak['hour']}** with "
        f"{peak['review_count']} reviews"
    )
    st.write(
        f"• Most risky location: "
        f"**{location_risk.iloc[0]['location']}**"
    )

    st.markdown("### 💡 Recommendations")
    st.write("""
1. Focus on high-risk locations  
2. Retrain high-risk drivers  
3. Improve service quality  
4. Monitor fake reviews
    """)

# ======================================================
# MODEL COMPARISON
# ======================================================
elif section == "Model Comparison":

    combined = pd.concat([
        sentiment_df.assign(Task="Sentiment"),
        risk_df.assign(Task="Risk")
    ])

    st.subheader("📊 Model Performance Table")
    st.dataframe(
        combined,
        use_container_width=True
    )

    st.subheader("📈 Model Comparison")

    # Row 1
    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots(figsize=(7, 5))
        sns.barplot(
            x="Model",
            y="Accuracy",
            hue="Task",
            data=combined,
            ax=ax1
        )
        ax1.set_title("Accuracy")
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        sns.barplot(
            x="Model",
            y="Precision",
            hue="Task",
            data=combined,
            ax=ax2
        )
        ax2.set_title("Precision")
        st.pyplot(fig2)

    # Row 2
    col3, col4 = st.columns(2)

    with col3:
        fig3, ax3 = plt.subplots(figsize=(7, 5))
        sns.barplot(
            x="Model",
            y="Recall",
            hue="Task",
            data=combined,
            ax=ax3
        )
        ax3.set_title("Recall")
        st.pyplot(fig3)

    with col4:
        fig4, ax4 = plt.subplots(figsize=(7, 5))
        sns.barplot(
            x="Model",
            y="F1 Score",
            hue="Task",
            data=combined,
            ax=ax4
        )
        ax4.set_title("F1 Score")
        st.pyplot(fig4)

# ======================================================
# ANALYTICS
# ======================================================
elif section == "Analytics":

    st.subheader("📍 Top 10 High Risk Locations")

    fig1, ax1 = plt.subplots(figsize=(12, 6))
    sns.barplot(
        x="high_risk_count",
        y="location",
        data=location_risk,
        ax=ax1
    )
    st.pyplot(fig1)

    st.caption(
        "Higher bar = more high-risk incidents in that location"
    )

    st.markdown("---")

    st.subheader("⏱️ Peak Hour Activity")

    fig2, ax2 = plt.subplots(figsize=(12, 7))
    sns.barplot(
        x="hour",
        y="review_count",
        data=peak_hours,
        ax=ax2
    )
    st.pyplot(fig2)

    st.caption(
        "Hour (0–23) vs number of reviews — shows busiest times"
    )

    st.markdown("---")

    st.subheader("📈 Trend Over Time")

    fig3, ax3 = plt.subplots(figsize=(12, 6))
    sns.lineplot(
        x="time",
        y="reviews",
        data=trend,
        ax=ax3
    )
    plt.xticks(rotation=45)
    st.pyplot(fig3)

    st.caption(
        "Shows how number of reviews changes over time"
    )

    st.markdown("---")

    st.subheader("📊 Complaint Categories")

    col1, col2 = st.columns([2, 1])

    with col1:
        fig4, ax4 = plt.subplots(figsize=(10, 5))
        sns.barplot(
            x="count",
            y="category",
            data=category,
            ax=ax4
        )
        st.pyplot(fig4)

    with col2:
        st.info("""
• Delay → Late service

• Pricing → Fare issues

• Driver Behavior → Driver issues

• Safety → Risk incidents

• Vehicle Condition → Vehicle problems
        """)

# ======================================================
# LOCATION ANALYSIS
# ======================================================
elif section == "Location Analysis":

    selected_location = st.selectbox(
        "Select Location",
        locations
    )

    df_loc = fake_df[
        fake_df["location"] == selected_location
    ]

    st.subheader(f"📍 {selected_location}")

    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots(figsize=(7, 6))
        df_loc["sentiment"].value_counts().plot.pie(
            autopct="%1.1f%%",
            ax=ax1
        )
        ax1.set_ylabel("")
        st.pyplot(fig1)
        st.caption("Sentiment distribution")

    with col2:
        fig2, ax2 = plt.subplots(figsize=(7, 6))
        df_loc["risk"].value_counts().plot(
            kind="bar",
            ax=ax2
        )
        st.pyplot(fig2)
        st.caption(
            "Risk levels (0=Low, 1=Medium, 2=High)"
        )

    st.markdown("---")

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("⏱️ Peak Hours")

        loc_peak = df_loc.groupby("hour").size().reset_index(
            name="count"
        )

        fig3, ax3 = plt.subplots(figsize=(7, 5))
        sns.barplot(
            x="hour",
            y="count",
            data=loc_peak,
            ax=ax3
        )
        st.pyplot(fig3)
        st.caption(
            "Hour vs number of reviews in this location"
        )

    with col4:
        st.subheader("📊 Categories")

        loc_cat = df_loc["category"].value_counts().reset_index()
        loc_cat.columns = ["category", "count"]

        fig4, ax4 = plt.subplots(figsize=(7, 5))
        sns.barplot(
            x="count",
            y="category",
            data=loc_cat,
            ax=ax4
        )
        st.pyplot(fig4)
        st.caption(
            "Complaint types in this location"
        )

    st.markdown("---")

    st.write("### 📌 Summary")
    st.write(f"**Total Reviews:** {len(df_loc)}")
    st.write(
        f"**Avg Rating:** "
        f"{round(df_loc['rating'].mean(), 2)}"
    )

# ======================================================
# DRIVER ANALYSIS
# ======================================================
elif section == "Driver Analysis":

    st.subheader("🚖 Driver Performance Table")

    st.dataframe(
        driver_scores.head(10),
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("🔥 Top High-Risk Drivers")

    top_drivers = driver_scores.sort_values(
        "risk",
        ascending=False
    ).head(5)

    fig1, ax1 = plt.subplots(figsize=(12, 6))
    sns.barplot(
        x="risk",
        y="driver_id",
        data=top_drivers,
        ax=ax1
    )
    st.pyplot(fig1)

    st.markdown("---")

    selected_driver = st.selectbox(
        "Select Driver",
        drivers
    )

    df_driver = fake_df[
        fake_df["driver_id"] == selected_driver
    ]

    col1, col2 = st.columns(2)

    with col1:
        fig2, ax2 = plt.subplots(figsize=(7, 6))
        df_driver["sentiment"].value_counts().plot.pie(
            autopct="%1.1f%%",
            ax=ax2
        )
        ax2.set_ylabel("")
        st.pyplot(fig2)

    with col2:
        fig3, ax3 = plt.subplots(figsize=(7, 6))
        df_driver["risk"].value_counts().plot(
            kind="bar",
            ax=ax3
        )
        st.pyplot(fig3)

# ======================================================
# FAKE DETECTION
# ======================================================
elif section == "Fake Detection":

    col1, col2 = st.columns([1, 2])

    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        fake_df["fake_prediction"].value_counts().plot.pie(
            autopct="%1.1f%%",
            ax=ax1
        )
        ax1.set_ylabel("")
        st.pyplot(fig1)

    with col2:
        st.info("""
### 🔍 Fake Detection Criteria

• Very short reviews

• Repetitive or duplicate wording

• Unusual writing patterns

• Excess punctuation / uppercase / numbers

### 🧠 Prediction Classes

• Genuine → Real user feedback

• Fake → Suspicious / spam-like review

• Ambiguous → Uncertain case (borderline)
        """)

    st.markdown("---")

    st.subheader("📋 Flagged Reviews")

    st.dataframe(
        fake_df.head(20),
        use_container_width=True
    )

# ======================================================
# SUBMIT REVIEW
# ======================================================
elif section == "Submit Review":

    review = st.text_area("Your Review")
    rating = st.slider("Rating", 1, 5)
    location = st.selectbox("Location", locations)
    driver_id = st.selectbox("Driver", drivers)

    if st.button("Submit"):

        if review.strip() == "":
            st.warning("Please enter a review")

        else:
            st.success("Review submitted successfully!")

            review_len = len(review.split())
            char_len = len(review)
            unique_words = len(set(review.split()))

            avg_word_len = np.mean(
                [len(w) for w in review.split()]
            ) if review.split() else 0

            repetition_ratio = review_len / (unique_words + 1)

            uppercase_ratio = sum(
                1 for c in review if c.isupper()
            ) / (len(review) + 1)

            punctuation_count = len(
                re.findall(r"[!?.]", review)
            )

            digit_ratio = sum(
                c.isdigit() for c in review
            ) / (len(review) + 1)

            features = [[
                review_len,
                char_len,
                unique_words,
                avg_word_len,
                repetition_ratio,
                uppercase_ratio,
                punctuation_count,
                digit_ratio
            ]]

            features_scaled = fake_scaler.transform(features)
            score = fake_model.decision_function(
                features_scaled
            )[0]

            if repetition_ratio > 1.8:
                fake_label = "Fake"
            elif review_len <= 2:
                fake_label = "Fake"
            elif punctuation_count > 4:
                fake_label = "Fake"
            elif score < -0.15:
                fake_label = "Fake"
            elif score < 0.05:
                fake_label = "Ambiguous"
            else:
                fake_label = "Genuine"

            st.subheader("🤖 Prediction")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Authenticity",
                    fake_label
                )

            with col2:
                st.metric(
                    "Confidence Score",
                    round(abs(score), 3)
                )
