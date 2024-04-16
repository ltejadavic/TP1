class Config:
    SECRET_KEY = 'B!1weNAt1T^%kvhUI*S^'

class DevelopmentConfig():
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///retainai.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

config = {
    'development': DevelopmentConfig
}    