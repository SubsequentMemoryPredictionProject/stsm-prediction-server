from DB.db_access import get_results


def validate_user_results(results,db):
    predicted_values = get_results()
    return