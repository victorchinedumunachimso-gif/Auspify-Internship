import matplotlib.pyplot as plt
import pandas as pd

def plot_top_countries(data, n=10):
    countries = data['country'].value_counts().head(n)

    countries.plot(kind='bar')

    plt.title('Top Countries by Number of Titles')
    plt.xlabel('Country')
    plt.ylabel('Number of Titles')

    plt.tight_layout()
    plt.show()


def plot_top_genres(data, n=10):
    genres = data['listed_in'].dropna().str.split(', ')
    all_genres = genres.explode()
    genre_counts = all_genres.value_counts().head(n)

    genre_counts.plot(kind='bar')

    plt.title('Top Genres')
    plt.xlabel('Genre')
    plt.ylabel('Number of Titles')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_content_by_year(data):
    content_by_year = (
        data.groupby(['release_year', 'type'])
        .size()
        .unstack(fill_value=0)
    )

    content_by_year.plot(kind='line')

    plt.title('Movies vs TV Shows by Release Year')
    plt.xlabel('Release Year')
    plt.ylabel('Number of Titles')

    plt.tight_layout()
    plt.show()

def plot_rating_by_type(data):
    rating_by_type = pd.crosstab(data['rating'], data['type'])

    rating_by_type.plot(kind='bar')

    plt.title('Ratings by Content Type')
    plt.xlabel('Rating')
    plt.ylabel('Number of Titles')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_content_added_over_time(data):
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

    content_added.index = content_added.index.astype(str)

    content_added.plot(kind='line')

    plt.title('Content Added Over Time')
    plt.xlabel('Date Added')
    plt.ylabel('Number of Titles')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_top_directors(data, n=10):
    directors = data['director'].dropna()

    directors = directors[directors != 'Unknown']

    top_directors = directors.value_counts().head(n)

    top_directors.plot(kind='bar')

    plt.title('Top Directors by Number of Titles')
    plt.xlabel('Director')
    plt.ylabel('Number of Titles')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()





def plot_confusion_matrix(matrix):
    plt.imshow(matrix)

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    plt.xticks(
        [0, 1],
        ['Movie', 'TV Show']
    )

    plt.yticks(
        [0, 1],
        ['Movie', 'TV Show']
    )

    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                matrix[i, j],
                ha='center',
                va='center'
            )

    plt.tight_layout()
    plt.show()