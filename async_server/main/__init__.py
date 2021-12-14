from flask import Flask, jsonify, request
from celery import Celery
from .config import Config

app = Flask(__name__)
app.config.from_object(Config)

CELERY_TASK_LIST = [
    "main.tasks"
]

def make_celery():
    celery = Celery(
        broker=app.config["CELERY_BROKER_URL"],
        backend=app.config["CELERY_RESULT_BACKEND"],
        include=CELERY_TASK_LIST
    )
    
    celery.conf.task_routes = {
        "NN.predict": {'queue': "predict"}
    }
    
    TaskBase = celery.Task
    
    class ContextTask(TaskBase):
        abstract = True
        
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return TaskBase.__call__(self, *args, **kwargs)
    
    celery.Task = ContextTask
    
    return celery

from .tasks import predict

@app.route("/pred", methods=["POST"])
def pred():
    data = request.data
    images = data.decode("utf-8")
    
    res = predict.apply_async(args=[images], ignore_result=False)
    
    return jsonify(res.get())