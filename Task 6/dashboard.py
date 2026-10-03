import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Netflix Content Analytics",
    layout="wide"
)


@st.cache_data
def load_dashboard_data():
    data = pd.read_csv("data/Dataset.csv")
    return data


@st.cache_resource
def load_model():
    model = joblib.load("outputs/content_type_model.pkl")
    return model


data = load_dashboard_data()
model = load_model()


st.sidebar.title("Analysis Controls")

st.sidebar.write(
    "Use the controls below to filter the dashboard."
)

st.sidebar.divider()

st.sidebar.subheader("Content Filters")

min_year = int(data["release_year"].min())
max_year = int(data["release_year"].max())

selected_years = st.sidebar.slider(
    "Select Release Year Range",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year)
)

content_filter = st.sidebar.selectbox(
    "Select Content Type",
    ["All", "Movie", "TV Show"]
)

st.sidebar.divider()

st.sidebar.subheader("Dataset Information")

st.sidebar.write(
    f"Total records: {len(data):,}"
)

st.sidebar.write(
    f"Year range: {min_year} - {max_year}"
)

filtered_data = data[
    data["release_year"].between(
        selected_years[0],
        selected_years[1]
    )
]

if content_filter != "All":
    filtered_data = filtered_data[
        filtered_data["type"] == content_filter
    ]


st.title("Netflix Content Analytics")

st.write(
    "Interactive data analysis and machine learning dashboard "
    "for exploring Netflix content, trends and business insights."
)

st.caption(
    f"Analysis period: {selected_years[0]} - {selected_years[1]} "
    f"| Content filter: {content_filter}"
)

st.divider()


total_titles = len(filtered_data)

movie_count = (
    filtered_data["type"] == "Movie"
).sum()

tv_count = (
    filtered_data["type"] == "TV Show"
).sum()


if total_titles > 0:

    movie_percentage = (
        movie_count / total_titles
    ) * 100

    tv_percentage = (
        tv_count / total_titles
    ) * 100

    top_rating = (
        filtered_data["rating"]
        .value_counts()
        .index[0]
    )

else:

    movie_percentage = 0
    tv_percentage = 0
    top_rating = "N/A"


overview_tab, explorer_tab, ml_tab, business_tab = st.tabs(
    [
        "Overview",
        "Dataset Explorer",
        "Machine Learning",
        "Business Insights"
    ]
)


with overview_tab:

    st.header("Overview")

    st.write(
    "A high-level view of the Netflix content catalogue "
    "based on the selected filters."
)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Titles",
        f"{total_titles:,}"
    )

    col2.metric(
        "Movies",
        f"{movie_percentage:.2f}%"
    )

    col3.metric(
        "TV Shows",
        f"{tv_percentage:.2f}%"
    )

    col4.metric(
        "Top Rating",
        top_rating
    )

    st.header("Content Analysis")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Content Type Distribution")

        content_types = (
            filtered_data["type"]
            .value_counts()
        )

        st.bar_chart(content_types)

    with col2:

        st.subheader("Content Ratings")

        ratings = (
            filtered_data["rating"]
            .value_counts()
        )

        st.bar_chart(ratings)

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Top 10 Countries")

        top_countries = (
            filtered_data["country"]
            .value_counts()
            .head(10)
        )

        st.bar_chart(top_countries)

    with col2:

        st.subheader("Top 10 Genres")

        genres = (
            filtered_data["listed_in"]
            .dropna()
            .str.split(", ")
        )

        all_genres = genres.explode()

        top_genres = (
            all_genres
            .value_counts()
            .head(10)
        )

        st.bar_chart(top_genres)

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Content by Release Year")

        release_years = (
            filtered_data["release_year"]
            .value_counts()
            .sort_index()
        )

        st.line_chart(release_years)

    with col2:

        st.subheader("Movie Duration Analysis")

        movies = filtered_data[
            filtered_data["type"] == "Movie"
        ].copy()

        if len(movies) > 0:

            movies["duration_minutes"] = (
                movies["duration"]
                .str.extract(r"(\d+)")[0]
                .astype(float)
            )

            average_duration = (
                movies["duration_minutes"]
                .mean()
            )

            median_duration = (
                movies["duration_minutes"]
                .median()
            )

            duration_col1, duration_col2 = st.columns(2)

            duration_col1.metric(
                "Average",
                f"{average_duration:.2f} min"
            )

            duration_col2.metric(
                "Median",
                f"{median_duration:.0f} min"
            )

        else:

            st.info(
                "No movies found in the selected period."
            )


with explorer_tab:

    st.header("Dataset Explorer")

    explorer_data = filtered_data.copy()

    col1, col2, col3 = st.columns(3)

    with col1:

        countries = sorted(
            explorer_data["country"]
            .dropna()
            .unique()
            .tolist()
        )

        selected_country = st.selectbox(
            "Filter by Country",
            ["All"] + countries
        )

    with col2:

        ratings = sorted(
            explorer_data["rating"]
            .dropna()
            .unique()
            .tolist()
        )

        selected_rating = st.selectbox(
            "Filter by Rating",
            ["All"] + ratings
        )

    with col3:

        title_search = st.text_input(
            "Search Title"
        )

    if selected_country != "All":

        explorer_data = explorer_data[
            explorer_data["country"].str.contains(
                selected_country,
                case=False,
                na=False
            )
        ]

    if selected_rating != "All":

        explorer_data = explorer_data[
            explorer_data["rating"] == selected_rating
        ]

    if title_search:

        explorer_data = explorer_data[
            explorer_data["title"].str.contains(
                title_search,
                case=False,
                na=False
            )
        ]

    st.write(
        f"Showing {len(explorer_data):,} matching records."
    )

    st.dataframe(
        explorer_data.head(100),
        use_container_width=True
    )

    csv_data = explorer_data.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Filtered Dataset",
        data=csv_data,
        file_name="filtered_netflix_dataset.csv",
        mime="text/csv"
    )


    st.header("Machine Learning")

    st.write(
    "A Random Forest classification model is used to "
    "predict whether a Netflix title is a Movie or TV Show."
)

    st.subheader("Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
    "Accuracy",
    "96.99%"
)

    col2.metric(
    "Precision",
    "96.66%"
)

    col3.metric(
    "Recall",
    "99.10%"
)

    col4.metric(
    "F1 Score",
    "97.86%"
)

    st.caption(
    "Evaluation results from the trained Random Forest model."
)

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Confusion Matrix")

        confusion_data = pd.DataFrame(
            [
                [1214, 11],
                [42, 491]
            ],
            columns=[
                "Predicted Movie",
                "Predicted TV Show"
            ],
            index=[
                "Actual Movie",
                "Actual TV Show"
            ]
        )

        fig, ax = plt.subplots()

        ax.imshow(confusion_data.values)

        ax.set_xticks(
            range(len(confusion_data.columns))
        )

        ax.set_xticklabels(
            confusion_data.columns
        )

        ax.set_yticks(
            range(len(confusion_data.index))
        )

        ax.set_yticklabels(
            confusion_data.index
        )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        for i in range(2):
            for j in range(2):

                ax.text(
                    j,
                    i,
                    confusion_data.iloc[i, j],
                    ha="center",
                    va="center"
                )

        st.pyplot(fig)

with col2:

    st.subheader("Model Information")

    st.write(
        "Algorithm: Random Forest Classifier"
    )

    st.write(
        "Prediction target: Content Type"
    )

    st.write(
        "Classes: Movie and TV Show"
    )

    st.write(
        "Training features:"
    )

    st.write(
        "• Release year"
    )

    st.write(
        "• Rating"
    )

    st.write(
        "• Country"
    )

    st.write(
        "• Genre information"
    )

    st.info(
        "The model uses metadata to classify a title "
        "as either a Movie or TV Show."
    )

    st.header("Predict Content Type")

    st.write(
        "Enter information about a title and let the trained "
        "Random Forest model predict its content type."
    )

    col1, col2 = st.columns(2)

    with col1:

        release_year = st.number_input(
            "Release Year",
            min_value=1900,
            max_value=2030,
            value=2020
        )

        rating = st.text_input(
            "Rating",
            value="TV-MA"
        )

    with col2:

        country = st.text_input(
            "Country",
            value="United States"
        )

        listed_in = st.text_input(
            "Genres",
            value="Dramas, International Movies"
        )

    if st.button(
        "Predict Content Type",
        use_container_width=True
    ):

        input_data = pd.DataFrame(
            {
                "release_year": [release_year],
                "rating": [rating],
                "country": [country],
                "listed_in": [listed_in]
            }
        )

        prediction = model.predict(
            input_data
        )[0]

        probabilities = model.predict_proba(
            input_data
        )[0]

        classes = model.classes_

        prediction_probability = probabilities[
            list(classes).index(prediction)
        ]

        st.success(
            f"Predicted Content Type: {prediction}"
        )

        st.metric(
            "Prediction Confidence",
            f"{prediction_probability * 100:.2f}%"
        )

        probability_data = pd.DataFrame(
            {
                "Content Type": classes,
                "Probability": probabilities * 100
            }
        )

        st.subheader(
            "Prediction Probabilities"
        )

        st.bar_chart(
            probability_data.set_index(
                "Content Type"
            )
        )

        st.info(
            "The prediction is generated by the trained "
            "Random Forest model using release year, rating, "
            "country and genre information."
        )


with business_tab:

    st.header("Business Insights")

    if total_titles > 0:

        movie_percentage = (
            (filtered_data["type"] == "Movie").sum()
            / len(filtered_data)
        ) * 100

        tv_percentage = (
            (filtered_data["type"] == "TV Show").sum()
            / len(filtered_data)
        ) * 100

        top_country = (
            filtered_data["country"]
            .value_counts()
            .index[0]
        )

        top_rating = (
            filtered_data["rating"]
            .value_counts()
            .index[0]
        )

        genres = (
            filtered_data["listed_in"]
            .dropna()
            .str.split(", ")
        )

        all_genres = genres.explode()

        top_genre = (
            all_genres
            .value_counts()
            .index[0]
        )

        movies = filtered_data[
            filtered_data["type"] == "Movie"
        ].copy()

        if len(movies) > 0:

            movies["duration_minutes"] = (
                movies["duration"]
                .str.extract(r"(\d+)")[0]
                .astype(float)
            )

            average_movie_duration = (
                movies["duration_minutes"]
                .mean()
            )

        else:

            average_movie_duration = 0

        tv_shows = filtered_data[
            filtered_data["type"] == "TV Show"
        ].copy()

        if len(tv_shows) > 0:

            tv_shows["seasons"] = (
                tv_shows["duration"]
                .str.extract(r"(\d+)")[0]
                .astype(float)
            )

            median_tv_seasons = (
                tv_shows["seasons"]
                .median()
            )

        else:

            median_tv_seasons = 0

        col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Movie Share",
    f"{movie_percentage:.2f}%"
)

col2.metric(
    "TV Show Share",
    f"{tv_percentage:.2f}%"
)

col3.metric(
    "Top Country",
    top_country
)

col4.metric(
    "Top Genre",
    top_genre
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Content Insights")

    st.write(
        f"Movies represent **{movie_percentage:.2f}%** "
        "of the selected content library."
    )

    st.write(
        f"TV Shows represent **{tv_percentage:.2f}%** "
        "of the selected content library."
    )

    st.write(
        f"**{top_country}** has the highest number "
        "of titles in the selected dataset."
    )

    st.write(
        f"**{top_rating}** is the most common "
        "content rating."
    )

with col2:
    st.subheader("Content Characteristics")

    st.write(
        f"**{top_genre}** is the most common genre."
    )

    st.write(
        f"The average movie duration is "
        f"**{average_movie_duration:.2f} minutes**."
    )

    st.write(
        f"The median TV Show has "
        f"**{median_tv_seasons:.0f} season(s)**."
    )


    st.header("Business Recommendations")

    recommendations = [

        "Maintain strong movie content availability while continuing to develop the TV Show library.",

        "Prioritize internationally diverse content because international content represents a major part of the catalogue.",

        "Consider content strategies for mature audiences because TV-MA is the most common rating.",

        "Use historical content-addition trends to support future content acquisition planning.",

        "Consider both short-form and longer-running TV series when expanding the TV Show catalogue."

    ]

    for recommendation in recommendations:

        st.write(
            f"• {recommendation}"
        )

    st.divider()

    st.header("Executive Summary")

    st.write(
        "This dashboard analyzes Netflix content using "
        "exploratory data analysis, visualizations and "
        "machine learning. The analysis examines content "
        "types, ratings, countries, genres, release trends "
        "and movie durations."
    )

    st.write(
        "A Random Forest classification model is also used "
        "to predict whether content is a Movie or TV Show "
        "based on available metadata."
    )

    st.success(
        "The analysis combines descriptive analytics, "
        "predictive analytics and business insights in "
        "a single interactive dashboard."
    )

    st.divider()

st.header("Download Executive Report")

report_text = f"""
NETFLIX CONTENT ANALYTICS REPORT

ANALYSIS PERIOD
{selected_years[0]} - {selected_years[1]}

CONTENT FILTER
{content_filter}

DATASET SUMMARY
Total Titles: {total_titles:,}
Movie Share: {movie_percentage:.2f}%
TV Show Share: {tv_percentage:.2f}%

KEY INSIGHTS
Top Country: {top_country}
Top Rating: {top_rating}
Top Genre: {top_genre}
Average Movie Duration: {average_movie_duration:.2f} minutes
Median TV Show Seasons: {median_tv_seasons:.0f}

MACHINE LEARNING
Algorithm: Random Forest Classifier
Accuracy: 96.99%
Precision: 96.66%
Recall: 99.10%
F1 Score: 97.86%

BUSINESS RECOMMENDATIONS
"""

for recommendation in recommendations:
    report_text += f"- {recommendation}\n"

report_text += f"""

EXECUTIVE SUMMARY

This analysis examines Netflix content using exploratory
data analysis, visualization and machine learning.

The dashboard evaluates content types, ratings, countries,
genres, release trends and content characteristics.

A Random Forest classification model was used to predict
whether a title is a Movie or TV Show based on metadata.

The analysis combines descriptive analytics, predictive
analytics and business insights in an interactive dashboard.
"""

st.download_button(
    label="Download Executive Report",
    data=report_text,
    file_name="netflix_executive_report.txt",
    mime="text/plain",
    use_container_width=True
)