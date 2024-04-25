class Config:
    SECRET_KEY = 'B!1weNAt1T^%kvhUI*S^'

class DevelopmentConfig():
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///retainai.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

class ProductionConfig(Config):
    DEBUG = False
    SQLALCHEMY_ECHO = False
    # Asegúrate de establecer la URI de la base de datos para producción
    SQLALCHEMY_DATABASE_URI = 'sqlite:///retainai.db'  # Cambia esta línea según sea necesario  

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig
}    