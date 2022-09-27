from django.urls import URLPattern, path
from . import views

urlpatterns = [
    path("", views.startPage, name="startPage"),
    path("labeling_web", views.labeling_web, name="labeling_web"),
    path("support", views.support, name="support"),
    path("home", views.indexPage, name="home"),
    path("addtomodel", views.addtomodel, name="addtomodel"),
    # path("detection", views.detection, name="detection"),
    path("image_list", views.image_list, name="image_list"),
    path("image_list/<int:id>/", views.delete_photoac, name="delete_photoac"),
    path("final_report", views.final_report, name="final_report"),
    path("final_report/<int:id>/", views.delete_photo, name="delete_photo"),
    path("change_label/<int:id>/", views.change_label, name="change_label"),
    # path("formweb", views.formweb, name="formweb"),
    path("image_label", views.image_label, name="image_label"),
    path("add_label/<int:id>/", views.add_label, name="add_label"),
    path(
        "delete_unlabelphoto/<int:id>/",
        views.delete_unlabelphoto,
        name="delete_unlabelphoto",
    ),
    path(
        "remove_labelphoto/<int:id>/", views.remove_labelphoto, name="remove_labelphoto"
    ),
    path("final_report", views.final_report, name="final_report"),
    path("activ_learning", views.activ_learning, name="activ_learning"),
    path("model_introduce", views.model_introduce, name="model_introduce"),
]
