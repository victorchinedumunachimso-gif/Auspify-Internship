import pandas as pd

def generate_business_insights(data):
    insights = []

    movie_count = (data['type'] == 'Movie').sum()
    tv_count = (data['type'] == 'TV Show').sum()

    total = len(data)

    movie_percentage = (movie_count / total) * 100
    tv_percentage = (tv_count / total) * 100

    insights.append(
        f"Movies account for {movie_percentage:.2f}% of the content library."
    )

    insights.append(
        f"TV Shows account for {tv_percentage:.2f}% of the content library."
    )

    top_country = data['country'].value_counts().index[0]

    insights.append(
        f"{top_country} is the country with the highest number of titles."
    )

    top_rating = data['rating'].value_counts().index[0]

    insights.append(
        f"{top_rating} is the most common content rating."
    )

    return insights

def generate_duration_insight(data):
    movies = data[data['type'] == 'Movie'].copy()

    movies['duration_minutes'] = (
        movies['duration']
        .str.extract(r'(\d+)')[0]
        .astype(float)
    )

    average_duration = movies['duration_minutes'].mean()

    return (
        f"The average movie duration is "
        f"{average_duration:.2f} minutes."
    )

def generate_top_genre_insight(data):
    genres = data['listed_in'].dropna().str.split(', ')
    all_genres = genres.explode()

    top_genre = all_genres.value_counts().index[0]
    count = all_genres.value_counts().iloc[0]

    return (
        f"{top_genre} is the most common genre, "
        f"appearing in {count} titles."
    )

def generate_tv_season_insight(data):
    tv_shows = data[data['type'] == 'TV Show'].copy()

    tv_shows['seasons'] = (
        tv_shows['duration']
        .str.extract(r'(\d+)')[0]
        .astype(float)
    )

    median_seasons = tv_shows['seasons'].median()

    return (
        f"The median TV Show has {median_seasons:.0f} season(s)."
    )

def generate_growth_insight(data):
    data = data.copy()

    data['date_added'] = pd.to_datetime(
        data['date_added'],
        errors='coerce'
    )

    yearly_content = data['date_added'].dt.year.value_counts()

    peak_year = yearly_content.idxmax()
    peak_count = yearly_content.max()

    return (
        f"{peak_year} had the highest number of titles added, "
        f"with {peak_count} titles."
    )