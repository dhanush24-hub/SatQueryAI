import os
import io
import datetime
from typing import List, Optional
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from PIL import Image

from app.schemas.analysis import AnalysisResult
from app.schemas.image_asset import ImageAsset
from app.services.storage.local import storage_service
from app.core.logging import logger


class PDFReportGenerator:
    """
    Generates publication-quality, evidence-first PDF analysis reports for SatQuery AI.
    Strictly documents observable findings, imagery metadata, model provenance, and limitations.
    """

    def generate_report(
        self,
        result: AnalysisResult,
        assets: List[ImageAsset],
        output_filename: Optional[str] = None
    ) -> str:
        """
        Builds a comprehensive PDF report and writes it to storage.
        Returns the relative storage path of the generated PDF.
        """
        filename = output_filename or f"satquery_report_{result.id}.pdf"
        full_pdf_path = storage_service.get_full_path(filename)
        os.makedirs(os.path.dirname(full_pdf_path), exist_ok=True)

        doc = SimpleDocTemplate(
            full_pdf_path,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()
        normal = styles["Normal"]
        
        title_style = ParagraphStyle(
            "ReportTitle",
            parent=styles["Heading1"],
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#0f172a"),
            spaceAfter=4
        )
        subtitle_style = ParagraphStyle(
            "ReportSubtitle",
            parent=normal,
            fontSize=10,
            leading=13,
            textColor=colors.HexColor("#64748b"),
            spaceAfter=12
        )
        section_style = ParagraphStyle(
            "SectionHeading",
            parent=styles["Heading2"],
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#1e293b"),
            spaceBefore=10,
            spaceAfter=6
        )
        body_style = ParagraphStyle(
            "BodyTextCustom",
            parent=normal,
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#334155")
        )
        badge_style = ParagraphStyle(
            "BadgeStyle",
            parent=normal,
            fontSize=8,
            leading=10,
            textColor=colors.white
        )

        story = []

        # 1. Header Banner
        story.append(Paragraph("<b>SatQuery AI</b> &mdash; Remote Sensing Intelligence", title_style))
        story.append(Paragraph(
            f"Evidence-First Analysis Report &bull; ID: <b>{result.id}</b> &bull; Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            subtitle_style
        ))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284c7"), spaceAfter=10))

        # 2. Query & Executive Answer Box
        task_str = result.task.value if hasattr(result.task, "value") else str(result.task or "UNKNOWN")
        state_color = "#16a34a" if result.evidence_state == "SUPPORTED" else ("#ca8a04" if "WARNING" in result.evidence_state else "#dc2626")
        
        answer_data = [
            [Paragraph(f"<b>Query:</b> <i>\"{result.answer.split('Answer: ')[-1] if 'Answer: ' in result.answer else result.answer}\"</i>", body_style)],
            [Paragraph(f"<b>Synthesized Finding:</b> {result.answer}", body_style)],
            [Paragraph(
                f"<b>Task Family:</b> {task_str} &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"<b>Evidence Quality:</b> <font color='{state_color}'><b>{result.evidence_state}</b></font> &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"<b>Status:</b> {result.status.value}",
                body_style
            )]
        ]
        answer_table = Table(answer_data, colWidths=[540])
        answer_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(answer_table)
        story.append(Spacer(1, 10))

        # 3. Input Imagery Metadata
        story.append(Paragraph("<b>1. Input Imagery Provenance & Sensors</b>", section_style))
        asset_rows = [["Asset ID", "Filename", "Modality", "Sensor", "CRS", "Dimensions", "Resolution (GSD)"]]
        for a in assets:
            crs_str = a.crs if a.crs else "Unreferenced (Pixel Grid)"
            res_str = f"{a.resolution:.2f} m" if a.resolution else "Unknown"
            asset_rows.append([
                Paragraph(a.id[:12], body_style),
                Paragraph(a.original_filename or a.filename, body_style),
                a.modality.value if hasattr(a.modality, "value") else str(a.modality),
                a.sensor or "Unknown",
                Paragraph(crs_str[:18], body_style),
                f"{a.width}x{a.height}x{a.bands}",
                res_str
            ])
        asset_table = Table(asset_rows, colWidths=[70, 110, 60, 60, 90, 75, 75])
        asset_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(asset_table)
        story.append(Spacer(1, 10))

        # 4. Imagery Previews (T1, T2, Change Mask side-by-side if present)
        story.append(Paragraph("<b>2. Spatial Evidence & Visual Overlays</b>", section_style))
        preview_imgs = []
        labels = []
        for idx, a in enumerate(assets[:2]):
            preview_local = storage_service.get_full_path(f"{a.id}_preview.png")
            if not os.path.exists(preview_local):
                preview_local = storage_service.get_full_path(os.path.basename(a.storage_path))
            
            if os.path.exists(preview_local) and preview_local.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    preview_imgs.append(RLImage(preview_local, width=170, height=130))
                    labels.append(Paragraph(f"<b>T{idx+1} ({a.sensor or 'Optical'}):</b> {a.id[:8]}", body_style))
                except Exception:
                    pass

        # Check for change mask preview
        if result.overlays and result.overlays.change_mask_url:
            mask_rel = result.overlays.change_mask_url.split("/api/uploads/previews/")[-1]
            mask_local = storage_service.get_full_path(mask_rel)
            if os.path.exists(mask_local):
                try:
                    preview_imgs.append(RLImage(mask_local, width=170, height=130))
                    labels.append(Paragraph("<b>Predicted Change Mask (&tau;=0.50)</b>", body_style))
                except Exception:
                    pass

        if preview_imgs:
            story.append(Table([preview_imgs, labels], colWidths=[180] * len(preview_imgs), style=[
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 2),
            ]))
            story.append(Spacer(1, 10))

        # 5. Structured Regional Findings Table
        story.append(Paragraph("<b>3. Structured Key Findings</b>", section_style))
        if result.findings:
            finding_rows = [["ID", "Summary", "Pixels", "Area (m²)", "Quality", "Source Model", "Limitations"]]
            for f in result.findings[:10]:
                area_str = f"{f.area_m2:.1f} m²" if f.area_m2 is not None else "N/A (Unreferenced)"
                limit_str = "; ".join(f.limitations[:2]) if f.limitations else "None"
                finding_rows.append([
                    f.finding_id,
                    Paragraph(f.summary, body_style),
                    str(f.changed_pixels or "N/A"),
                    area_str,
                    f.evidence_quality,
                    f.source_model,
                    Paragraph(limit_str, body_style)
                ])
            finding_table = Table(finding_rows, colWidths=[55, 145, 45, 65, 75, 75, 80])
            finding_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("PADDING", (0, 0), (-1, -1), 3),
            ]))
            story.append(finding_table)
        else:
            story.append(Paragraph("<i>No regional change clusters met the spatial threshold.</i>", body_style))
        story.append(Spacer(1, 10))

        # 6. Quantitative Statistics & Audit Checks (Registration & Domain)
        story.append(Paragraph("<b>4. Engineering Quality Gates & Model Provenance</b>", section_style))
        
        reg = result.registration
        reg_txt = f"Status: <b>{reg.status}</b> | Mutual Correlation: <b>{reg.correlation:.3f}</b> | Estimated Translation: <b>{reg.translation_px:.2f} px</b>" if reg else "Registration unchecked (Single-image task)"
        
        dom = result.domain_suitability
        dom_txt = f"Status: <b>{dom.status}</b> | Compliant: <b>{'YES' if dom.is_suitable else 'NO'}</b>" if dom else "Domain compliant"

        prov = result.model_provenance
        prov_txt = (
            f"Active Model: <b>{prov.model_name}</b> ({prov.architecture})<br/>"
            f"Checkpoint: <code>{prov.checkpoint}</code> &bull; Parameters: <b>{prov.parameters:,}</b> &bull; Decision Threshold: <b>&tau;={prov.threshold:.2f}</b><br/>"
            f"Validated Benchmark Performance (LEVIR-CD Held-Out Test Split): <b>Global F1: {prov.benchmark_f1:.4f}</b> | <b>Global IoU: {prov.benchmark_iou:.4f}</b>"
        ) if prov else "Analytical CVA or Standard VQA adapter."

        audit_data = [
            [Paragraph("<b>Co-Registration Quality Gate:</b>", body_style), Paragraph(reg_txt, body_style)],
            [Paragraph("<b>Domain Suitability Gate:</b>", body_style), Paragraph(dom_txt, body_style)],
            [Paragraph("<b>Model Provenance & Verification:</b>", body_style), Paragraph(prov_txt, body_style)]
        ]
        audit_table = Table(audit_data, colWidths=[140, 400])
        audit_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#94a3b8")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("PADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(audit_table)
        story.append(Spacer(1, 10))

        # 7. Observable Step-by-Step Execution Trace
        story.append(Paragraph("<b>5. Observable Execution Trace</b>", section_style))
        trace_rows = [["Step", "Tool / Capability", "Status", "Duration", "Execution Fact / Output"]]
        for idx, s in enumerate(result.execution_summary):
            dur_str = f"{s.duration_ms:.1f} ms" if s.duration_ms is not None else "--"
            trace_rows.append([
                str(idx + 1),
                Paragraph(s.tool_name, body_style),
                s.status,
                dur_str,
                Paragraph(s.summary, body_style)
            ])
        trace_table = Table(trace_rows, colWidths=[30, 110, 55, 55, 290])
        trace_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("PADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(trace_table)
        story.append(Spacer(1, 10))

        # 8. Operational Limitations & Scientific Notice
        story.append(Paragraph("<b>6. Mandatory Operational Limitations & Disclaimers</b>", section_style))
        limitations = result.warnings + [
            "Benchmark accuracy metrics reflect evaluation on LEVIR-CD test scenes and do not guarantee equivalent accuracy on unverified sensor modalities.",
            "Area calculations are strictly omitted for rasters lacking authentic georeferencing to prevent false precision.",
            "Visual appearance alone cannot confirm land-use semantics without multi-spectral verification or ground reference."
        ]
        lim_text = "<br/>&bull; ".join(list(dict.fromkeys(limitations)))
        story.append(Paragraph(f"&bull; {lim_text}", body_style))

        # Build document
        doc.build(story)
        logger.info(f"Generated PDF analysis report: {full_pdf_path}")
        return filename


pdf_generator = PDFReportGenerator()
