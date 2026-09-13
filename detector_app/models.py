from django.db import models
from django.contrib.auth.models import User


class ScanReport(models.Model):
    ticket_id = models.CharField(max_length=64, blank=True, null=True, unique=True)
    dispatch_status = models.CharField(max_length=30, default='Not Dispatched')
    dispatched_at = models.DateTimeField(blank=True, null=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE)

    video_name = models.CharField(max_length=255)
    video_file = models.FileField(upload_to='input_videos/')
    output_video = models.FileField(
        upload_to='output_videos/',
        null=True,
        blank=True
    )

    unique_potholes = models.IntegerField(default=0)
    minor_potholes = models.IntegerField(default=0)
    moderate_potholes = models.IntegerField(default=0)
    severe_potholes = models.IntegerField(default=0)

    psi = models.FloatField(default=0.0)
    worst_segment = models.CharField(max_length=50, default="-")

    latitude = models.CharField(
        max_length=100,
        null=True,
        blank=True
    )

    longitude = models.CharField(
        max_length=100,
        null=True,
        blank=True
    )

    # Road location entered manually by the user
    road_location = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        default="Unspecified Location"
    )

    created_at = models.DateTimeField(auto_now_add=True)

    # --- New Dispatch Fields ---
    ticket_id = models.CharField(
        max_length=64, 
        blank=True, 
        null=True, 
        unique=True
    )
    
    dispatch_status = models.CharField(
        max_length=30,
        default='Not Dispatched',
        choices=[
            ('Not Dispatched', 'Not Dispatched'),
            ('Pending Review', 'Pending Review'),
            ('Dispatched to Ward PWD', 'Dispatched to Ward PWD'),
            ('Under Maintenance', 'Under Maintenance'),
            ('Resolved', 'Resolved')
        ]
    )
    
    dispatched_at = models.DateTimeField(blank=True, null=True)

    def __str__(self):
        return f"{self.user.username} - {self.video_name}"