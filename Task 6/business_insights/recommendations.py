def generate_recommendations(data):

    recommendations = []

    movie_percentage = (
        (data['type'] == 'Movie').sum() / len(data)
    ) * 100

    tv_percentage = (
        (data['type'] == 'TV Show').sum() / len(data)
    ) * 100

    if movie_percentage > tv_percentage:
        recommendations.append(
            "Maintain strong movie content availability while "
            "continuing to develop the TV Show library."
        )

    recommendations.append(
        "Prioritize internationally diverse content because "
        "International Movies is the most common genre."
    )

    recommendations.append(
        "Consider content strategies around mature audiences "
        "because TV-MA is the most common rating."
    )

    recommendations.append(
        "Use historical content-addition trends to plan future "
        "content acquisition and release strategies."
    )

    recommendations.append(
        "Consider both short-form TV series and longer-running "
        "series when expanding the TV Show catalogue."
    )

    return recommendations

def save_business_report(insights, recommendations, file_path):

    with open(file_path, 'w', encoding='utf-8') as file:

        file.write("NETFLIX DATA ANALYSIS REPORT\n")
        file.write("=" * 40 + "\n\n")

        file.write("BUSINESS INSIGHTS\n")
        file.write("-" * 40 + "\n")

        for insight in insights:
            file.write(f"- {insight}\n")

        file.write("\nBUSINESS RECOMMENDATIONS\n")
        file.write("-" * 40 + "\n")

        for recommendation in recommendations:
            file.write(f"- {recommendation}\n")