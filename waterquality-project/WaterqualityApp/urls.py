from django.urls import path
from . import views

urlpatterns = [
    path("index.html", views.index, name="index"),
    path("Signup.html", views.Signup, name="Signup"),
    path("SignupAction", views.SignupAction, name="SignupAction"),
    path("UserLogin.html", views.UserLogin, name="UserLogin"),
    path("UserLoginAction", views.UserLoginAction, name="UserLoginAction"),
    path("ProcessData", views.ProcessData, name="ProcessData"),
    path("TrainRF", views.TrainRF, name="TrainRF"),
    path("TrainLSTM", views.TrainLSTM, name="TrainLSTM"),
    path("TrainLR", views.TrainLR, name="TrainLR"),
    path("TrainDTC", views.TrainDTC, name="TrainDTC"),
    path("TrainKNN", views.TrainKNN, name="TrainKNN"),
    path("TrainGBC", views.TrainGBC, name="TrainGBC"),
    path("TrainSVM", views.TrainSVM, name="TrainSVM"),
    path("Predict", views.Predict, name="Predict"),
    path("PredictAction", views.PredictAction, name="PredictAction"),
    path("Forecast", views.Forecast, name="Forecast"),
]
