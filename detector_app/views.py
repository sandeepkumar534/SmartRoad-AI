import os
import sys
import subprocess
import shutil
import io
import random
import pandas as pd

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.conf import settings
from django.http import FileResponse
from django.utils import timezone

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

from .models import ScanReport


def home_view(request):
    return render(request, 'home.html')


def login_view(request):
    if request.method == 'POST':
        u = request.POST.get('username')
        p = request.POST.get('password')
        user = authenticate(request, username=u, password=p)
        if user:
            login(request, user)
            return redirect('dashboard')
        messages.error(request, 'Invalid username or password.')
    return render(request, 'login.html')


def signup_view(request):
    if request.method == 'POST':
        u = request.POST.get('username')
        e = request.POST.get('email')
        p = request.POST.get('password')
        cp = request.POST.get('confirm_password')
        if p != cp:
            messages.error(request, 'Passwords do not match.')
        elif User.objects.filter(username=u).exists():
            messages.error(request, 'Username already exists.')
        else:
            User.objects.create_user(username=u, email=e, password=p)
            messages.success(request, 'Account created! Please log in.')
            return redirect('login')
    return render(request, 'signup.html')


def logout_view(request):
    logout(request)
    return redirect('home')


@login_required(login_url='login')
def profile_view(request):
    return render(request, 'profile.html')


@login_required(login_url='login')
def dashboard_view(request):
    history = ScanReport.objects.filter(user=request.user).order_by('-created_at')
    total_scans = history.count()
    total_potholes = sum(h.unique_potholes for h in history)
    context = {
        'history': history,
        'total_scans': total_scans,
        'total_potholes': total_potholes,
    }
    return render(request, 'dashboard.html', context)


@login_required(login_url='login')
def analyze_road_view(request):
    if request.method == 'POST' and request.FILES.get('road_video'):
        video_file = request.FILES['road_video']

        # =========================================================
        # LOCATION INPUTS: MANUAL TEXT + GPS
        # =========================================================
        road_location = request.POST.get('road_location', '').strip()
        if not road_location:
            road_location = "Bengaluru Urban Road"

        lat = request.POST.get('latitude', '12.9716')
        lon = request.POST.get('longitude', '77.5946')

        # =========================================================
        # CREATE SCAN REPORT
        # =========================================================
        report = ScanReport.objects.create(
            user=request.user,
            video_name=video_file.name,
            video_file=video_file,
            road_location=road_location,
            latitude=lat,
            longitude=lon
        )

        input_video_abs = os.path.abspath(report.video_file.path)
        video_stem = os.path.splitext(video_file.name)[0]

        run_folder_name = f"{request.user.username}_{video_stem}"
        run_output_dir = os.path.join(
            settings.MEDIA_ROOT,
            'outputs',
            run_folder_name
        )
        os.makedirs(run_output_dir, exist_ok=True)

        python_bin = sys.executable

        cmd = [
            python_bin,
            "-m",
            "src.main",
            "--model",
            "yolov8",
            "--track",
            "--input",
            input_video_abs,
            "--output",
            run_output_dir
        ]

        print(f"\n[INFO] Starting Pipeline Execution: {' '.join(cmd)}\n")
        proc = subprocess.run(cmd, cwd=str(settings.BASE_DIR))
        print(f"\n[INFO] Pipeline Completed with Return Code: {proc.returncode}\n")

        # =========================================================
        # 1. CSV PARSING
        # =========================================================
        unique_p = 0
        minor_p = 0
        mod_p = 0
        sev_p = 0
        sev_pct = 0.0
        avg_score = 0.0
        worst_seg = "2"
        psi_val = 16.18

        csv_path = os.path.join(run_output_dir, "benchmark_results.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if not df.empty:
                    row = df.iloc[0]
                    unique_p = int(row.get('total_unique_potholes', 0))
                    minor_p = int(row.get('total_minor', 0))
                    mod_p = int(row.get('total_moderate', 0))
                    sev_p = int(row.get('total_severe', 0))
                    sev_pct = round(float(row.get('severe_percent', 0.0)), 1)
                    avg_score = round(float(row.get('average_severity_score', 0.0)), 2)

                    if 'relative_psi' in row:
                        psi_val = round(float(row.get('relative_psi', 16.18)), 2)
                    elif 'Relative PSI' in row:
                        psi_val = round(float(row.get('Relative PSI', 16.18)), 2)

                    if 'worst_segment' in row:
                        worst_seg = str(row.get('worst_segment', '2'))
                    elif 'Worst Segment' in row:
                        worst_seg = str(row.get('Worst Segment', '2'))
            except Exception as e:
                print(f"[ERROR] CSV parse error: {e}")

        # =========================================================
        # 2. BROWSER VIDEO CONVERSION (H.264)
        # =========================================================
        raw_video_path = os.path.join(run_output_dir, "yolov8_result.mp4")
        web_video_path = os.path.join(run_output_dir, "yolov8_web.mp4")
        video_to_serve = "yolov8_result.mp4"

        if os.path.exists(raw_video_path):
            ffmpeg_exe = shutil.which("ffmpeg")
            if not ffmpeg_exe:
                try:
                    import imageio_ffmpeg
                    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                except ImportError:
                    ffmpeg_exe = None

            if ffmpeg_exe:
                convert_cmd = [
                    ffmpeg_exe,
                    "-y",
                    "-i",
                    raw_video_path,
                    "-vcodec",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    "-crf",
                    "23",
                    web_video_path
                ]
                try:
                    print("[INFO] Converting output video to browser-compatible H.264 format...")
                    conv_res = subprocess.run(
                        convert_cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    if conv_res.returncode == 0 and os.path.exists(web_video_path):
                        video_to_serve = "yolov8_web.mp4"
                        print("[INFO] Video conversion successful! Serving yolov8_web.mp4")
                except Exception as e:
                    print(f"[WARN] Video conversion skipped: {e}")

        # =========================================================
        # 3. SAVE RESULT TO DATABASE
        # =========================================================
        rel_video_path = f"outputs/{run_folder_name}/{video_to_serve}"
        report.output_video.name = rel_video_path
        report.unique_potholes = unique_p
        report.minor_potholes = minor_p
        report.moderate_potholes = mod_p
        report.severe_potholes = sev_p
        report.psi = psi_val
        report.worst_segment = worst_seg
        report.save()

        # =========================================================
        # 4. DYNAMIC ROAD DAMAGE & SERVICEABILITY ASSESSMENT
        # =========================================================
        if sev_pct >= 40 or unique_p >= 30:
            avg_sev_text = "Severe Damage"
            road_status_title = "Critical / Poor Condition"
            road_status_desc = "High concentration of severe potholes detected. Immediate resurfacing required."
            road_status_color = "#dc2626"
        elif sev_pct >= 20 or unique_p >= 15:
            avg_sev_text = "Moderate Damage"
            road_status_title = "Moderate Wear & Tear"
            road_status_desc = "Surface exhibits structural degradation. Patchwork recommended within 14 days."
            road_status_color = "#ea580c"
        elif unique_p > 0:
            avg_sev_text = "Minor Damage"
            road_status_title = "Minor Defects Detected"
            road_status_desc = "Surface has small defects. Routine periodic maintenance recommended."
            road_status_color = "#0284c7"
        else:
            avg_sev_text = "Good / Minimal Damage"
            road_status_title = "Good Condition"
            road_status_desc = "Surface is within operational tolerances. Standard periodic audits recommended."
            road_status_color = "#16a34a"

        # =========================================================
        # 5. FINAL CONTEXT
        # =========================================================
        context = {
            'video_url': f"{settings.MEDIA_URL}{rel_video_path}",
            'report_id': report.id,
            'road_location': road_location,
            'total_unique_potholes': unique_p,
            'total_minor': minor_p,
            'total_moderate': mod_p,
            'total_severe': sev_p,
            'severe_percent': sev_pct,
            'average_severity': avg_sev_text,
            'road_status_title': road_status_title,
            'road_status_desc': road_status_desc,
            'road_status_color': road_status_color,
            'worst_segment': worst_seg,
            'psi': psi_val,
            'latitude': lat,
            'longitude': lon,
            'ticket_id': getattr(report, 'ticket_id', None),
            'dispatch_status': getattr(report, 'dispatch_status', 'Not Dispatched'),
        }

        return render(request, 'analyze_road.html', context)

    return redirect('dashboard')


# =========================================================
# 6. OFFICIAL AUDIT PDF DOWNLOAD VIEW
# =========================================================
@login_required(login_url='login')
def download_pdf_report(request, report_id):
    report = get_object_or_404(ScanReport, id=report_id, user=request.user)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    elements = []
    styles = getSampleStyleSheet()

    # Title & Incident Metadata
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=6
    )
    elements.append(Paragraph("SmartRoad AI — Road Maintenance Audit Report", title_style))
    
    # If dispatched, add Ticket Reference in subheader
    ticket_badge = f" | Ticket Ref: #{report.ticket_id}" if getattr(report, 'ticket_id', None) else ""
    elements.append(Paragraph(
        f"Audit Date: {report.created_at.strftime('%d-%m-%Y %H:%M')} | Incident ID: #SR-2026-{report.id}{ticket_badge}",
        styles['Normal']
    ))
    elements.append(Spacer(1, 12))

    # Location & General Info Table
    disp_status = getattr(report, 'dispatch_status', 'Not Dispatched')
    info_data = [
        ["Inspected Road:", str(report.road_location), "Inspector:", str(report.user.username)],
        ["Geographic Coordinates:", f"{report.latitude}, {report.longitude}", "Serviceability Index (PSI):", f"{report.psi}"],
        ["Worst Affected Segment:", f"Segment #{report.worst_segment}", "Authority Dispatch Status:", str(disp_status)]
    ]
    t_info = Table(info_data, colWidths=[150, 160, 150, 80])
    t_info.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor("#334155")),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(t_info)
    elements.append(Spacer(1, 14))

    # Metrics Table
    metrics_data = [
        ["Defect Classification", "Detected Count", "Action Recommended"],
        ["Unique Potholes Detected", str(report.unique_potholes), "Official Surface Audit Logged"],
        ["Minor Damage", str(report.minor_potholes), "Routine Surface Sealing"],
        ["Moderate Damage", str(report.moderate_potholes), "Patchwork Recommended within 14 Days"],
        ["Severe / Critical Damage", str(report.severe_potholes), "Immediate Asphalt Milling & Resurfacing"],
    ]
    t_metrics = Table(metrics_data, colWidths=[180, 110, 250])
    t_metrics.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2563eb")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    elements.append(t_metrics)
    elements.append(Spacer(1, 14))

    # Embed Segment Graph if available in outputs
    video_stem = os.path.splitext(report.video_name)[0]
    graph_path = os.path.join(settings.MEDIA_ROOT, 'outputs', f"{report.user.username}_{video_stem}", "segment_analysis.png")
    if os.path.exists(graph_path):
        elements.append(Paragraph("<b>Segment Severity Distribution Graph:</b>", styles['Normal']))
        elements.append(Spacer(1, 6))
        elements.append(RLImage(graph_path, width=480, height=180))
        elements.append(Spacer(1, 10))

    elements.append(Paragraph("Authorized by Civic Works Infrastructure Audit Module.", styles['Italic']))

    doc.build(elements)
    buffer.seek(0)
    return FileResponse(buffer, as_attachment=True, filename=f"SmartRoad_Audit_{report.id}.pdf")


# =========================================================
# 7. CIVIC AUTHORITY DISPATCH VIEW
# =========================================================
@login_required(login_url='login')
def dispatch_report_view(request, report_id):
    report = get_object_or_404(ScanReport, id=report_id, user=request.user)

    if not getattr(report, 'ticket_id', None):
        date_str = timezone.now().strftime("%Y%m%d")
        rand_suffix = random.randint(1000, 9999)
        report.ticket_id = f"PWD-{date_str}-{report.id}{rand_suffix}"
        report.dispatch_status = "Dispatched to Ward PWD"
        report.dispatched_at = timezone.now()
        report.save()
        messages.success(request, f"Complaint officially dispatched to Municipal Authority! Ticket ID: {report.ticket_id}")
    else:
        messages.info(request, f"This audit is already registered under Ticket ID: {report.ticket_id}")

    return redirect('dashboard')