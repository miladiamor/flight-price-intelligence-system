# services/model3_service.py
from __future__ import annotations

from datetime import datetime
from typing import Dict, Any

from services.recommender_db import (
    init_db,
    get_searches,
    get_feedback_summary,
    get_popularity,
    log_search,
)
from services.recommender_algo import recommend_hybrid

# Ensure DB exists
init_db()


def recommend(
    user_id: str,
    origin: str,
    date_str: str,
    price: float,
    k: int = 5,
    exclude_destination: str | None = None,
) -> Dict[str, Any]:
    """
    Returns:
      {
        "query": {...},
        "recommendations": [...]
      }
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d").date()
    month = int(dt.month)

    # pull user history + feedback + popularity
    searches = get_searches(limit=2000)
    feedback = get_feedback_summary(user_id)
    popularity = get_popularity()

    recs = recommend_hybrid(
        user_id=str(user_id),
        origin=origin,
        month=month,
        price=float(price),
        k=int(k),
        searches=searches,
        feedback=feedback,
        popularity=popularity,
        exclude_destination=exclude_destination,
        reference_destination=exclude_destination,  # IMPORTANT for relative price proxy
    )

    # Optional: log top-1 result as a "search" so popularity grows for demo purposes
    if recs:
        log_search(
            user_id=str(user_id),
            origin=origin.upper(),
            destination=recs[0]["destination"],
            month=month,
            price=float(recs[0].get("predicted_price") or price),
        )

    return {
        "query": {
            "user_id": str(user_id),
            "origin": origin.upper(),
            "date": str(dt),
            "month": month,
            "price": float(price),
            "excluded_destination": exclude_destination.upper() if exclude_destination else None,
        },
        "recommendations": recs,
    }
