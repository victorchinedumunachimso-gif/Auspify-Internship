from analysis import load_data, show_basic_information, show_missing_values, show_duplicates, remove_duplicates, missing_value_report, fill_text_missing_values, fill_numeric_missing_values, show_columns, analyze_content_types, content_type_percentage, plot_content_types, analyze_release_years, plot_release_year_trend, analyze_ratings, plot_ratings, analyze_duration, analyze_movie_duration, analyze_tv_seasons, analyze_countries, analyze_top_countries, analyze_genres, analyze_individual_genres, analyze_content_by_year, analyze_rating_by_type, analyze_content_added_over_time, analyze_movie_duration_stats, analyze_tv_season_stats, analyze_top_directors
from visualizations import plot_top_genres, plot_top_countries, plot_content_by_year, plot_rating_by_type, plot_content_added_over_time, plot_top_directors, plot_confusion_matrix

from models import prepare_ml_data, split_features_target,create_train_test_data, create_model_pipeline, train_model, make_predictions, evaluate_model, create_confusion_matrix, generate_classification_report, save_model

from business_insights.insights import generate_duration_insight, generate_business_insights, generate_top_genre_insight, generate_tv_season_insight, generate_growth_insight
from business_insights.recommendations import generate_recommendations, save_business_report
import os



data = load_data('Task 6/data/Dataset.csv')

show_basic_information(data)

show_missing_values(data)

show_duplicates(data)

data = remove_duplicates(data)

report = missing_value_report(data)

print(report)

data = fill_text_missing_values(data)

print(data)

show_columns(data)

data = fill_numeric_missing_values(data)


content_types = analyze_content_types(data)
print(content_types)

percentages = content_type_percentage(data)
print(percentages)
plot_content_types(data)

release_years = analyze_release_years(data)
print(release_years)

plot_release_year_trend(data)

ratings = analyze_ratings(data)
print(ratings)

plot_ratings(data)

duration = analyze_duration(data)
print(duration)

movie_duration = analyze_movie_duration(data)
print(movie_duration)

tv_seasons = analyze_tv_seasons(data)
print(tv_seasons)

countries = analyze_countries(data)
print(countries)

top_countries = analyze_top_countries(data)
print(top_countries)

plot_top_countries(data)

genres = analyze_genres(data)
print(genres.head(10))

genre_counts = analyze_individual_genres(data)

print(genre_counts.head(10))

plot_top_genres(data)

content_by_year = analyze_content_by_year(data)
print(content_by_year)

plot_content_by_year(data)

rating_by_type = analyze_rating_by_type(data)

print(rating_by_type)

plot_rating_by_type(data)

content_added = analyze_content_added_over_time(data)

print(content_added)

plot_content_added_over_time(data)

duration_stats = analyze_movie_duration_stats(data)

print(duration_stats)

season_stats = analyze_tv_season_stats(data)
print(season_stats)

top_directors = analyze_top_directors(data)

print(top_directors)

plot_top_directors(data)

ml_data = prepare_ml_data(data)
print(ml_data.head())
print(ml_data.shape)


X, y = split_features_target(ml_data)

print("Features:")
print(X.head())

print("\nTarget:")
print(y.head())

X_train, X_test, y_train, y_test = create_train_test_data(X, y)

print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)

print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)


pipeline = create_model_pipeline()

pipeline = train_model(
    pipeline,
    X_train,
    y_train
)

print("Model training completed!")

predictions = make_predictions(
    pipeline,
    X_test
)

print(predictions[:10])

metrics = evaluate_model(y_test, predictions)

print(metrics)

matrix = create_confusion_matrix(
    y_test,
    predictions
)

print(matrix)

report = generate_classification_report(
    y_test,
    predictions
)

print(report)

plot_confusion_matrix(matrix)

save_model(
    pipeline,
    'Task 6/outputs/content_type_model.pkl'
)


duration_insight = generate_duration_insight(data)

print(duration_insight)

insights = generate_business_insights(data)

for insight in insights:
    print(insight)


genre_insight = generate_top_genre_insight(data)
print(genre_insight)

season_insight = generate_tv_season_insight(data)
print(season_insight)

growth_insight = generate_growth_insight(data)
print(growth_insight)

recommendations = generate_recommendations(data)

print("\nBUSINESS RECOMMENDATIONS")

for recommendation in recommendations:
    print("-", recommendation)
    
print(data)



print("Current folder:", os.getcwd())
print("Outputs folder exists:", os.path.exists("outputs"))
print("Model exists:", os.path.exists("outputs/content_type_model.pkl"))

save_business_report(
    insights,
    recommendations,
    'Task 6/outputs/business_report.txt'
)
