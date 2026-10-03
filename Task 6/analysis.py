import pandas as pd
import matplotlib.pyplot as plt


def load_data(file_path):
    data = pd.read_csv(file_path)
    return data



def show_basic_information(data):
    print(data.shape)
    print(data.columns)
    print(data.dtypes)
    print(data.head())

def show_columns(data):
    print(data.columns)

def show_missing_values(data):
    print(data.isnull().sum())


def show_duplicates(data):
    print(data.duplicated())
    print(data.duplicated().sum())


def remove_duplicates(data):
    data = data.drop_duplicates()
    return data

def missing_value_report(data):

    missing_values = data.isnull().sum()

    missing_percentage = (missing_values / len(data)) * 100

    report = pd.DataFrame({
        "Column": data.columns,
        "Missing Values": missing_values,
        "Missing %": missing_percentage
    })

    return report


def fill_text_missing_values(data):
    for column in ['director', 'listed_in', 'country']:
        data[column] = data[column].fillna('Unknown')

    return data

def fill_numeric_missing_values(data):
    numeric_columns = data.select_dtypes(include='number').columns

    for column in numeric_columns:
        data[column] = data[column].fillna(data[column].median())

    return data

def analyze_content_types(data):
    content_types = data['type'].value_counts()
    return content_types

def content_type_percentage(data):
    percentages = data['type'].value_counts(normalize = True) * 100
    return percentages

def plot_content_types(data):
    content_types = data['type'].value_counts()

    content_types.plot(kind='bar')

    plt.title('Movies vs TV Shows')
    plt.xlabel('Content Type')
    plt.ylabel('Number of Titles')

    plt.show()


def analyze_release_years(data):
    release_years = data['release_year'].value_counts().sort_index()
    return release_years

def plot_release_year_trend(data):
    release_years = data['release_year'].value_counts().sort_index()

    release_years.plot(kind='line')

    plt.title('Content by Release Year')
    plt.xlabel('Release Year')
    plt.ylabel('Number of Titles')

    plt.show()


def analyze_ratings(data):
    ratings = data['rating'].value_counts()

    return ratings


def plot_ratings(data):
    ratings = data['rating'].value_counts()

    ratings.plot(kind='bar')

    plt.title('Content Ratings Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Number of Titles')

    plt.show()

def analyze_duration(data):
    duration = data['duration'].value_counts()
    return duration

def analyze_movie_duration(data):
    movies = data[data['type'] == 'Movie']

    duration = movies['duration'].value_counts()

    return duration

def analyze_tv_seasons(data):
    tv_shows = data[data['type'] == 'TV Show']

    seasons = tv_shows['duration'].value_counts()

    return seasons

def analyze_countries(data):
    countries = data['country'].value_counts()
    return countries

def analyze_top_countries(data, n=10):
    countries = data['country'].value_counts()
    top_countries = countries.head(n)

    return top_countries

def analyze_genres(data):
    genres = data['listed_in'].value_counts()
    return genres

def analyze_individual_genres(data):
    genres = data['listed_in'].dropna().str.split(', ')

    all_genres = genres.explode()

    genre_counts = all_genres.value_counts()

    return genre_counts

def analyze_content_by_year(data):
    content_by_year = (
        data.groupby(['release_year', 'type'])
        .size()
        .unstack(fill_value=0)
    )

    return content_by_year

def analyze_rating_by_type(data):
    rating_by_type = pd.crosstab(data['rating'], data['type'])
    return rating_by_type

def analyze_content_added_over_time(data):
    data = data.copy()

    data['date_added'] = pd.to_datetime(
        data['date_added'],
        errors='coerce'
    )

    content_added = (
        data['date_added']
        .dt.to_period('M')
        .value_counts()
        .sort_index()
    )

    return content_added

def analyze_movie_duration_stats(data):
    movies = data[data['type'] == 'Movie'].copy()

    movies['duration_minutes'] = (
        movies['duration']
        .str.extract(r'(\d+)')[0]
        .astype(float)
    )

    average_duration = movies['duration_minutes'].mean()
    median_duration = movies['duration_minutes'].median()
    minimum_duration = movies['duration_minutes'].min()
    maximum_duration = movies['duration_minutes'].max()

    return {
        'average': average_duration,
        'median': median_duration,
        'minimum': minimum_duration,
        'maximum': maximum_duration
    }

def analyze_tv_season_stats(data):
    tv_shows = data[data['type'] == 'TV Show'].copy()

    tv_shows['seasons'] = (
        tv_shows['duration']
        .str.extract(r'(\d+)')[0]
        .astype(float)
    )

    average_seasons = tv_shows['seasons'].mean()
    median_seasons = tv_shows['seasons'].median()
    minimum_seasons = tv_shows['seasons'].min()
    maximum_seasons = tv_shows['seasons'].max()

    return {
        'average': average_seasons,
        'median': median_seasons,
        'minimum': minimum_seasons,
        'maximum': maximum_seasons
    }

def analyze_top_directors(data, n=10):
    directors = data['director'].dropna()

    directors = directors[directors != 'Unknown']

    top_directors = directors.value_counts().head(n)

    return top_directors