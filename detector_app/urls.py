from django.urls import path
from .views import dispatch_report_view


from .views import (
    home_view,
    login_view,
    signup_view,
    logout_view,
    dashboard_view,
    profile_view,
    analyze_road_view,
    download_pdf_report,
)

urlpatterns = [
    path('', home_view, name='home'),
    path('login/', login_view, name='login'),
    path('signup/', signup_view, name='signup'),
    path('logout/', logout_view, name='logout'),
    path('dashboard/', dashboard_view, name='dashboard'),
    path('profile/', profile_view, name='profile'),
    path('analyze-road/', analyze_road_view, name='analyze_road'),
    path('download-report/<int:report_id>/', download_pdf_report, name='download_pdf_report'),
    path('dispatch-report/<int:report_id>/', dispatch_report_view, name='dispatch_report'),
]