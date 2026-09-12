import os
import subprocess
from pathlib import Path
from typing import Dict, Any

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QLineEdit,
    QPushButton, QGroupBox, QListWidget, QListWidgetItem, QInputDialog,
    QMessageBox, QGridLayout, QFrame
)
from PySide6.QtCore import Qt

from core.project_manager import ProjectManager
from core.usage_tracker import UsageTracker
from core.diagnostics.system_check import SystemDiagnostics
import config.paths as paths

class ProjectUsagePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(12)

        # LEFT COLUMN: Project Manager (480px width)
        left_col = QWidget()
        left_col.setFixedWidth(480)
        left_lay = QVBoxLayout(left_col)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(6)

        # Project Box
        proj_box = QGroupBox("PROJECT WORKSPACE MANAGER")
        p_lay = QVBoxLayout(proj_box)
        p_lay.setContentsMargins(8, 10, 8, 6)
        p_lay.setSpacing(6)

        btn_row = QHBoxLayout()
        self.new_proj_btn = QPushButton("+ Create New Project")
        self.new_proj_btn.setObjectName("btnAccent")
        self.open_proj_folder_btn = QPushButton("Open Projects Folder")
        btn_row.addWidget(self.new_proj_btn)
        btn_row.addWidget(self.open_proj_folder_btn)
        p_lay.addLayout(btn_row)

        self.proj_list = QListWidget()
        self.proj_list.setStyleSheet("background: #070D19; border: 1px solid #1E293B; border-radius: 6px;")
        p_lay.addWidget(self.proj_list)
        left_lay.addWidget(proj_box)

        # Diagnostics Run Box
        diag_box = QGroupBox("SYSTEM DIAGNOSTICS & TELEMETRY")
        d_lay = QVBoxLayout(diag_box)
        self.run_diag_btn = QPushButton("Run Full Diagnostic Self-Check")
        self.run_diag_btn.setObjectName("btnPrimary")
        d_lay.addWidget(self.run_diag_btn)

        self.diag_console = QTextEdit()
        self.diag_console.setReadOnly(True)
        self.diag_console.setStyleSheet("font-family: 'JetBrains Mono', monospace; font-size: 11px; background: #070D19; border: 1px solid #1E293B;")
        d_lay.addWidget(self.diag_console)
        left_lay.addWidget(diag_box)

        main_layout.addWidget(left_col, stretch=0)

        # RIGHT COLUMN: Usage Statistics Dashboard
        right_box = QGroupBox("IMAGESTUDIO USAGE & PRODUCTIVITY DASHBOARD")
        right_lay = QVBoxLayout(right_box)
        right_lay.setContentsMargins(12, 14, 12, 12)
        right_lay.setSpacing(12)

        # Today Grid
        today_lbl = QLabel("TODAY'S ACTIVITY")
        today_lbl.setStyleSheet("font-weight: bold; color: #38BDF8; font-size: 13px;")
        right_lay.addWidget(today_lbl)

        self.today_grid = QGridLayout()
        self.lbl_today_gen = self._create_stat_card("Images Generated", "0", 0, 0)
        self.lbl_today_edit = self._create_stat_card("Images Edited", "0", 0, 1)
        self.lbl_today_i2i = self._create_stat_card("Img2Img Transformations", "0", 0, 2)
        self.lbl_today_lora = self._create_stat_card("LoRA Generations", "0", 1, 0)
        self.lbl_today_up = self._create_stat_card("Upscaled Outputs", "0", 1, 1)
        self.lbl_today_vram = self._create_stat_card("VRAM Peak", "0.0 GB", 1, 2)
        self.lbl_today_avg = self._create_stat_card("Average Latency", "0.0 s", 2, 0)
        self.lbl_today_time = self._create_stat_card("Total Active Time", "0.0 s", 2, 1)
        right_lay.addLayout(self.today_grid)

        # Divider
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #1E293B;")
        right_lay.addWidget(line)

        # All-Time Grid
        all_lbl = QLabel("ALL-TIME METRICS")
        all_lbl.setStyleSheet("font-weight: bold; color: #10B981; font-size: 13px;")
        right_lay.addWidget(all_lbl)

        self.all_grid = QGridLayout()
        self.lbl_all_gen = self._create_stat_card("Total Generations", "0", 0, 0, self.all_grid)
        self.lbl_all_edit = self._create_stat_card("Total Edits", "0", 0, 1, self.all_grid)
        self.lbl_all_time = self._create_stat_card("Total Render Time", "0.0 s", 0, 2, self.all_grid)
        self.lbl_all_top_model = self._create_stat_card("Most Used Checkpoint", "None", 1, 0, self.all_grid)
        self.lbl_all_top_sampler = self._create_stat_card("Most Used Sampler", "None", 1, 1, self.all_grid)
        right_lay.addLayout(self.all_grid)

        right_lay.addStretch()
        main_layout.addWidget(right_box, stretch=1)

        # Connect events
        self.new_proj_btn.clicked.connect(self._create_project)
        self.open_proj_folder_btn.clicked.connect(self._open_projects_folder)
        self.run_diag_btn.clicked.connect(self._run_diagnostics)

        self.refresh_all()

    def _create_stat_card(self, title: str, default_val: str, row: int, col: int, target_grid=None) -> QLabel:
        grid = target_grid or self.today_grid
        card = QFrame()
        card.setStyleSheet("background: #0B1120; border: 1px solid #1E293B; border-radius: 6px; padding: 6px;")
        c_lay = QVBoxLayout(card)
        c_lay.setContentsMargins(6, 4, 6, 4)
        c_lay.setSpacing(2)

        t_lbl = QLabel(title)
        t_lbl.setStyleSheet("color: #94A3B8; font-size: 10px; text-transform: uppercase;")
        v_lbl = QLabel(default_val)
        v_lbl.setStyleSheet("color: #FFFFFF; font-size: 14px; font-weight: bold; font-family: 'JetBrains Mono', monospace;")

        c_lay.addWidget(t_lbl)
        c_lay.addWidget(v_lbl)
        grid.addWidget(card, row, col)
        return v_lbl

    def refresh_all(self):
        # Refresh Projects List
        self.proj_list.clear()
        for p in ProjectManager.list_projects():
            item = QListWidgetItem(p)
            self.proj_list.addItem(item)

        # Refresh Usage Stats
        summary = UsageTracker.get_summary()
        t = summary.get("today", {})
        a = summary.get("all_time", {})

        self.lbl_today_gen.setText(str(t.get("generated", 0)))
        self.lbl_today_edit.setText(str(t.get("edited", 0)))
        self.lbl_today_i2i.setText(str(t.get("img2img", 0)))
        self.lbl_today_lora.setText(str(t.get("lora_runs", 0)))
        self.lbl_today_up.setText(str(t.get("upscaled", 0)))
        self.lbl_today_vram.setText(f"{t.get('peak_vram_gb', 0.0):.1f} GB")
        self.lbl_today_avg.setText(f"{t.get('avg_time_sec', 0.0):.1f} s")
        self.lbl_today_time.setText(f"{t.get('total_time_sec', 0.0):.1f} s")

        self.lbl_all_gen.setText(str(a.get("total_generations", 0)))
        self.lbl_all_edit.setText(str(a.get("total_edits", 0)))
        self.lbl_all_time.setText(f"{a.get('total_time_sec', 0.0) / 60:.1f} min")

        checkpoints = a.get("checkpoints", {})
        top_ckpt = max(checkpoints, key=checkpoints.get) if checkpoints else "None"
        self.lbl_all_top_model.setText(top_ckpt)

        samplers = a.get("samplers", {})
        top_samp = max(samplers, key=samplers.get) if samplers else "None"
        self.lbl_all_top_sampler.setText(top_samp)

    def _create_project(self):
        name, ok = QInputDialog.getText(self, "Create New Project", "Project Name:")
        if ok and name.strip():
            ProjectManager.create_project(name.strip())
            self.refresh_all()

    def _open_projects_folder(self):
        folder = ProjectManager.PROJECTS_DIR
        if folder.exists():
            subprocess.Popen(f'explorer "{folder.resolve()}"')

    def _run_diagnostics(self):
        report = SystemDiagnostics.generate_report()
        self.diag_console.setText(report)
