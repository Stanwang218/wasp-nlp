"""
Label semantics for CARTE classification datasets.

Each entry maps:
  - target_col: original column name in raw data
  - threshold: binarization threshold
  - label_1: human-readable meaning of label=1
  - label_0: human-readable meaning of label=0

Extracted from: https://github.com/soda-inria/carte/blob/main/carte_ai/scripts/preprocess_raw.py
"""

LABEL_SEMANTICS = {
    "chocolate_bar_ratings": {
        "target_col": "Rating",
        "threshold": 3.25,
        "label_1": "Good quality (Rating ≥ 3.25)",
        "label_0": "Below average (Rating < 3.25)",
        "answer_1": "Good",
        "answer_0": "Average",
    },
    "coffee_ratings": {
        "target_col": "rating",
        "threshold": 93,
        "label_1": "Excellent (rating > 93)",
        "label_0": "Average or below (rating ≤ 93)",
        "answer_1": "Excellent",
        "answer_0": "Average",
    },
    "michelin": {
        "target_col": "Award",
        "threshold": None,  # special: "MICHELIN" → 1, "Bib Gourmand" → 0
        "label_1": "Awarded MICHELIN star",
        "label_0": "Awarded Bib Gourmand only",
        "answer_1": "Starred",
        "answer_0": "Bib",
    },
    "nba_draft": {
        "target_col": "value_over_replacement",
        "threshold": 0,
        "label_1": "Positive value (VORP > 0)",
        "label_0": "No positive value (VORP ≤ 0)",
        "answer_1": "Valuable",
        "answer_0": "Replaceable",
    },
    "ramen_ratings": {
        "target_col": "Stars",
        "threshold": 4.0,
        "label_1": "Highly rated (Stars ≥ 4)",
        "label_0": "Low rated (Stars < 4)",
        "answer_1": "Good",
        "answer_0": "Average",
    },
    "roger_ebert": {
        "target_col": "critic_rating",
        "threshold": 3.5,
        "label_1": "Good review (rating ≥ 3.5)",
        "label_0": "Average review (rating < 3.5)",
        "answer_1": "Good",
        "answer_0": "Average",
    },
    "spotify": {
        "target_col": "popularity",
        "threshold": None,  # no binarization in preprocessing — keep as-is or median split
        "label_1": "Popular (above median)",
        "label_0": "Unpopular (below median)",
        "answer_1": "Popular",
        "answer_0": "Unpopular",
    },
    "whisky": {
        "target_col": "Meta_Critic",
        "threshold": 8.6,
        "label_1": "Exceptional (Meta_Critic > 8.6)",
        "label_0": "Average (Meta_Critic ≤ 8.6)",
        "answer_1": "Exceptional",
        "answer_0": "Average",
    },
    "yelp": {
        "target_col": "stars",
        "threshold": 3.5,
        "label_1": "Popular (stars > 3.5)",
        "label_0": "Less popular (stars ≤ 3.5)",
        "answer_1": "Popular",
        "answer_0": "Unpopular",
    },
    "zomato": {
        "target_col": "rating",
        "threshold": 4.0,
        "label_1": "Good restaurant (rating ≥ 4)",
        "label_0": "Average restaurant (rating < 4)",
        "answer_1": "Good",
        "answer_0": "Average",
    },
}

# Datasets from CARTE that are regression (not classification) — skip or use median split
REGRESSION_DATASETS = {
    "anime_planet", "babies_r_us", "beer_ratings", "bikedekho", "bikewale",
    "buy_buy_baby", "cardekho", "clear_corpus", "company_employees",
    "employee_remuneration", "employee_salaries", "fifa22_players",
    "filmtv_movies", "journal_jcr", "journal_sjr", "jp_anime", "k_drama",
    "mlds_salaries", "movies", "museums", "mydramalist",
    "prescription_drugs", "rotten_tomatoes",
    "us_accidents_counts", "us_presidential",
    "used_cars_24", "used_cars_benz_italy", "used_cars_dot_com",
    "used_cars_pakistan", "used_cars_saudi_arabia",
    "videogame_sales", "wikiliq_beer", "wikiliq_spirit", "wina_pl",
    "wine_dot_com_prices", "wine_dot_com_ratings",
    "wine_enthusiasts_prices", "wine_enthusiasts_ratings",
    "wine_vivino_price", "wine_vivino_rating",
}
