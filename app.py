from flask import Flask
from flask_cors import CORS
from asgiref.wsgi import WsgiToAsgi
from routes import chatbot, transactions, health, financial_score, alerts, statistics
from routes.chatbot import bp as chatbot_bp
from routes.daily_spend import bp as daily_spend_bp
from routes.spending_by_category import bp as spending_by_category_bp


# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Register blueprints
app.register_blueprint(chatbot_bp)
app.register_blueprint(transactions.bp)
app.register_blueprint(health.bp)
app.register_blueprint(financial_score.bp)
app.register_blueprint(statistics.bp)
app.register_blueprint(alerts.bp)
app.register_blueprint(daily_spend_bp)
app.register_blueprint(spending_by_category_bp)

# ASGI wrapper for Uvicorn
asgi_app = WsgiToAsgi(app)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(asgi_app, host="0.0.0.0", port=5000, log_level="info")
