from werkzeug.security import check_password_hash
from .entities.User import User

class ModelUser:

    @staticmethod
    def authenticate(name: str, password: str) -> User:
        user_from_db = User.query_user_by_name(name)
        if user_from_db and user_from_db.check_password(password):
            return user_from_db
        return None
