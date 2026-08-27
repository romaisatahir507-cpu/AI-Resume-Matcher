def calculate_final_score(
    tfidf_score: float,
    semantic_score: float
) -> float:
    
    # Weight configuration 
    tfidf_weight = 0.40
    semantic_weight = 0.60

    # Calculate weighted score
    final_score = (
        tfidf_score * tfidf_weight 
        + semantic_score * semantic_weight
    )

    # Round to 2 decimal places

    return round(final_score, 2)



def get_recommendation(score: float) -> str:

    if score >= 80:
        return "Excellent Match"

    elif score >= 65:
        return "Strong Match"

    elif score >= 50:
        return "Moderate Match"

    elif score >= 35:
        return "Weak Match"

    else:
        return "Poor Match"