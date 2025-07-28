"""
Export Functionality for Research Results (PDF, JSON, CSV)
Phase 5: Production-Ready Features
"""

import io
import json
import csv
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
from enum import Enum
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from pydantic import BaseModel, Field
import logging
from io import BytesIO
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Export Models ---


class ExportFormat(str, Enum):
    """Supported export formats"""

    PDF = "pdf"
    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"
    MARKDOWN = "markdown"
    HTML = "html"


class ExportOptions(BaseModel):
    """Export configuration options"""

    format: ExportFormat
    include_sources: bool = True
    include_metadata: bool = True
    include_citations: bool = True
    include_summary: bool = True
    include_raw_data: bool = False
    page_size: str = "letter"  # letter or A4
    language: str = "en"
    custom_title: Optional[str] = None
    custom_footer: Optional[str] = None


class ExportResult(BaseModel):
    """Export operation result"""

    format: ExportFormat
    filename: str
    size_bytes: int
    content_type: str
    data: bytes
    metadata: Dict[str, Any] = Field(default_factory=dict)


# --- Base Exporter ---


class BaseExporter:
    """Base class for all exporters"""

    def __init__(self, options: ExportOptions):
        self.options = options

    async def export(self, research_data: Dict[str, Any]) -> ExportResult:
        """Export research data to specified format"""
        raise NotImplementedError


# --- PDF Exporter ---


class PDFExporter(BaseExporter):
    """Export research results to PDF"""

    def __init__(self, options: ExportOptions):
        super().__init__(options)
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Setup custom PDF styles"""
        # Title style
        self.styles.add(
            ParagraphStyle(
                name="CustomTitle",
                parent=self.styles["Title"],
                fontSize=24,
                textColor=colors.HexColor("#1a1a1a"),
                spaceAfter=30,
                alignment=TA_CENTER,
            )
        )

        # Paradigm styles
        paradigm_colors = {
            "dolores": "#DC143C",  # Crimson
            "teddy": "#4169E1",  # Royal Blue
            "bernard": "#000080",  # Navy
            "maeve": "#228B22",  # Forest Green
        }

        for paradigm, color in paradigm_colors.items():
            self.styles.add(
                ParagraphStyle(
                    name=f"Paradigm_{paradigm}",
                    parent=self.styles["Heading2"],
                    textColor=colors.HexColor(color),
                    fontSize=16,
                    spaceAfter=12,
                )
            )

        # Section styles
        self.styles.add(
            ParagraphStyle(
                name="SectionHeading",
                parent=self.styles["Heading2"],
                fontSize=14,
                textColor=colors.HexColor("#333333"),
                spaceAfter=10,
            )
        )

        # Citation style
        self.styles.add(
            ParagraphStyle(
                name="Citation",
                parent=self.styles["Normal"],
                fontSize=9,
                textColor=colors.HexColor("#666666"),
                leftIndent=20,
                rightIndent=20,
            )
        )

    async def export(self, research_data: Dict[str, Any]) -> ExportResult:
        """Export research data to PDF"""
        buffer = BytesIO()

        # Determine page size
        page_size = A4 if self.options.page_size == "A4" else letter

        # Create PDF document
        doc = SimpleDocTemplate(
            buffer,
            pagesize=page_size,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72,
        )

        # Build content
        story = []

        # Add title
        title = (
            self.options.custom_title
            or f"Research Report: {research_data.get('query', 'Unknown Query')}"
        )
        story.append(Paragraph(title, self.styles["CustomTitle"]))
        story.append(Spacer(1, 12))

        # Add metadata section
        if self.options.include_metadata:
            story.extend(self._build_metadata_section(research_data))

        # Add summary section
        if self.options.include_summary and "summary" in research_data:
            story.extend(self._build_summary_section(research_data["summary"]))

        # Add paradigm-specific content
        paradigm = research_data.get("paradigm", "unknown")
        if "answer" in research_data:
            story.extend(self._build_answer_section(research_data["answer"], paradigm))

        # Add sources section
        if self.options.include_sources and "sources" in research_data:
            story.extend(self._build_sources_section(research_data["sources"]))

        # Add citations
        if self.options.include_citations and "citations" in research_data:
            story.extend(self._build_citations_section(research_data["citations"]))

        # Build PDF
        doc.build(
            story,
            onFirstPage=self._add_header_footer,
            onLaterPages=self._add_header_footer,
        )

        # Get PDF data
        pdf_data = buffer.getvalue()
        buffer.close()

        # Create filename
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = (
            f"research_report_{research_data.get('id', 'unknown')}_{timestamp}.pdf"
        )

        return ExportResult(
            format=ExportFormat.PDF,
            filename=filename,
            size_bytes=len(pdf_data),
            content_type="application/pdf",
            data=pdf_data,
            metadata={
                "pages": doc.page,
                "paradigm": paradigm,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    def _build_metadata_section(self, research_data: Dict[str, Any]) -> List:
        """Build metadata section for PDF"""
        elements = []

        elements.append(Paragraph("Research Metadata", self.styles["SectionHeading"]))

        metadata_data = [
            ["Field", "Value"],
            ["Research ID", research_data.get("id", "N/A")],
            ["Query", research_data.get("query", "N/A")],
            ["Paradigm", research_data.get("paradigm", "N/A").title()],
            ["Depth", research_data.get("depth", "N/A")],
            ["Generated", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")],
            ["Sources Analyzed", str(research_data.get("sources_count", 0))],
            [
                "Confidence Score",
                f"{research_data.get('confidence_score', 0) * 100:.1f}%",
            ],
        ]

        metadata_table = Table(metadata_data, colWidths=[2 * inch, 4 * inch])
        metadata_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 12),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]
            )
        )

        elements.append(metadata_table)
        elements.append(Spacer(1, 20))

        return elements

    def _build_summary_section(self, summary: str) -> List:
        """Build summary section for PDF"""
        elements = []

        elements.append(Paragraph("Executive Summary", self.styles["SectionHeading"]))
        elements.append(Paragraph(summary, self.styles["BodyText"]))
        elements.append(Spacer(1, 20))

        return elements

    def _build_answer_section(self, answer: Dict[str, Any], paradigm: str) -> List:
        """Build answer section with paradigm-specific styling"""
        elements = []

        # Get paradigm style
        paradigm_style = self.styles.get(
            f"Paradigm_{paradigm}", self.styles["Heading2"]
        )

        elements.append(Paragraph(f"{paradigm.title()} Analysis", paradigm_style))

        # Add sections
        if "sections" in answer:
            for section in answer["sections"]:
                elements.append(
                    Paragraph(section.get("title", ""), self.styles["Heading3"])
                )
                elements.append(
                    Paragraph(section.get("content", ""), self.styles["BodyText"])
                )

                # Add bullet points if present
                if "points" in section:
                    bullet_items = []
                    for point in section["points"]:
                        bullet_items.append(
                            ListItem(Paragraph(point, self.styles["Normal"]))
                        )
                    elements.append(ListFlowable(bullet_items, bulletType="bullet"))

                elements.append(Spacer(1, 12))

        # Add key insights
        if "key_insights" in answer:
            elements.append(Paragraph("Key Insights", self.styles["Heading3"]))
            insights_items = []
            for insight in answer["key_insights"]:
                insights_items.append(
                    ListItem(Paragraph(insight, self.styles["Normal"]))
                )
            elements.append(ListFlowable(insights_items, bulletType="bullet"))
            elements.append(Spacer(1, 20))

        return elements

    def _build_sources_section(self, sources: List[Dict[str, Any]]) -> List:
        """Build sources section for PDF"""
        elements = []

        elements.append(PageBreak())
        elements.append(Paragraph("Sources", self.styles["SectionHeading"]))

        for i, source in enumerate(sources[:20], 1):  # Limit to 20 sources
            source_text = f"<b>[{i}]</b> {source.get('title', 'Untitled')} - "
            source_text += f"<i>{source.get('url', 'No URL')}</i>"

            if source.get("relevance_score"):
                source_text += f" (Relevance: {source['relevance_score']:.2f})"

            elements.append(Paragraph(source_text, self.styles["Normal"]))

            if source.get("summary"):
                elements.append(
                    Paragraph(source["summary"][:200] + "...", self.styles["Citation"])
                )

            elements.append(Spacer(1, 6))

        return elements

    def _build_citations_section(self, citations: List[Dict[str, Any]]) -> List:
        """Build citations section for PDF"""
        elements = []

        elements.append(Paragraph("Citations", self.styles["SectionHeading"]))

        for citation in citations:
            citation_text = f"{citation.get('text', '')} "
            citation_text += f"<super>[{citation.get('source_index', '?')}]</super>"
            elements.append(Paragraph(citation_text, self.styles["Citation"]))
            elements.append(Spacer(1, 4))

        return elements

    def _add_header_footer(self, canvas, doc):
        """Add header and footer to PDF pages"""
        canvas.saveState()

        # Header
        canvas.setFont("Helvetica", 9)
        canvas.drawString(
            inch, doc.height + doc.topMargin - 0.5 * inch, "Four Hosts Research Report"
        )
        canvas.drawRightString(
            doc.width + doc.leftMargin,
            doc.height + doc.topMargin - 0.5 * inch,
            datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        )

        # Footer
        if self.options.custom_footer:
            canvas.drawString(inch, 0.5 * inch, self.options.custom_footer)
        else:
            canvas.drawString(
                inch, 0.5 * inch, "Generated by Four Hosts Research System"
            )

        canvas.drawRightString(
            doc.width + doc.leftMargin, 0.5 * inch, f"Page {doc.page}"
        )

        canvas.restoreState()


# --- JSON Exporter ---


class JSONExporter(BaseExporter):
    """Export research results to JSON"""

    async def export(self, research_data: Dict[str, Any]) -> ExportResult:
        """Export research data to JSON"""
        # Prepare export data
        export_data = {
            "metadata": {
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "format_version": "1.0",
                "exporter": "four_hosts_research",
            },
            "research": {
                "id": research_data.get("id"),
                "query": research_data.get("query"),
                "paradigm": research_data.get("paradigm"),
                "depth": research_data.get("depth"),
                "timestamp": research_data.get("timestamp"),
            },
        }

        # Add optional sections
        if self.options.include_summary:
            export_data["summary"] = research_data.get("summary")

        if self.options.include_sources:
            export_data["sources"] = research_data.get("sources", [])

        if self.options.include_citations:
            export_data["citations"] = research_data.get("citations", [])

        if self.options.include_raw_data:
            export_data["raw_data"] = research_data

        # Add answer sections
        export_data["answer"] = research_data.get("answer", {})

        # Convert to JSON
        json_data = json.dumps(export_data, indent=2, ensure_ascii=False).encode(
            "utf-8"
        )

        # Create filename
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = (
            f"research_data_{research_data.get('id', 'unknown')}_{timestamp}.json"
        )

        return ExportResult(
            format=ExportFormat.JSON,
            filename=filename,
            size_bytes=len(json_data),
            content_type="application/json",
            data=json_data,
            metadata={"pretty_printed": True, "encoding": "utf-8"},
        )


# --- CSV Exporter ---


class CSVExporter(BaseExporter):
    """Export research results to CSV"""

    async def export(self, research_data: Dict[str, Any]) -> ExportResult:
        """Export research data to CSV"""
        output = io.StringIO()

        # Determine what to export based on content
        if "sources" in research_data and self.options.include_sources:
            # Export sources as CSV
            fieldnames = [
                "index",
                "title",
                "url",
                "relevance_score",
                "credibility_score",
                "summary",
            ]
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()

            for i, source in enumerate(research_data["sources"], 1):
                writer.writerow(
                    {
                        "index": i,
                        "title": source.get("title", ""),
                        "url": source.get("url", ""),
                        "relevance_score": source.get("relevance_score", 0),
                        "credibility_score": source.get("credibility_score", 0),
                        "summary": source.get("summary", "")[
                            :500
                        ],  # Truncate long summaries
                    }
                )

        else:
            # Export general research data
            writer = csv.writer(output)
            writer.writerow(["Field", "Value"])
            writer.writerow(["Research ID", research_data.get("id", "")])
            writer.writerow(["Query", research_data.get("query", "")])
            writer.writerow(["Paradigm", research_data.get("paradigm", "")])
            writer.writerow(["Depth", research_data.get("depth", "")])
            writer.writerow(
                ["Confidence Score", research_data.get("confidence_score", 0)]
            )
            writer.writerow(["Sources Analyzed", research_data.get("sources_count", 0)])

            if "summary" in research_data:
                writer.writerow(["Summary", research_data["summary"]])

            # Add key insights
            if "answer" in research_data and "key_insights" in research_data["answer"]:
                for i, insight in enumerate(research_data["answer"]["key_insights"], 1):
                    writer.writerow([f"Key Insight {i}", insight])

        # Get CSV data
        csv_data = output.getvalue().encode("utf-8")
        output.close()

        # Create filename
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = (
            f"research_export_{research_data.get('id', 'unknown')}_{timestamp}.csv"
        )

        return ExportResult(
            format=ExportFormat.CSV,
            filename=filename,
            size_bytes=len(csv_data),
            content_type="text/csv",
            data=csv_data,
            metadata={"encoding": "utf-8", "delimiter": ","},
        )


# --- Excel Exporter ---


class ExcelExporter(BaseExporter):
    """Export research results to Excel"""

    async def export(self, research_data: Dict[str, Any]) -> ExportResult:
        """Export research data to Excel"""
        output = BytesIO()

        # Create Excel writer
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            # Overview sheet
            overview_data = {
                "Field": [
                    "Research ID",
                    "Query",
                    "Paradigm",
                    "Depth",
                    "Confidence Score",
                    "Sources Analyzed",
                    "Generated At",
                ],
                "Value": [
                    research_data.get("id", ""),
                    research_data.get("query", ""),
                    research_data.get("paradigm", ""),
                    research_data.get("depth", ""),
                    f"{research_data.get('confidence_score', 0) * 100:.1f}%",
                    research_data.get("sources_count", 0),
                    datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                ],
            }
            overview_df = pd.DataFrame(overview_data)
            overview_df.to_excel(writer, sheet_name="Overview", index=False)

            # Sources sheet
            if "sources" in research_data and self.options.include_sources:
                sources_data = []
                for i, source in enumerate(research_data["sources"], 1):
                    sources_data.append(
                        {
                            "Index": i,
                            "Title": source.get("title", ""),
                            "URL": source.get("url", ""),
                            "Relevance Score": source.get("relevance_score", 0),
                            "Credibility Score": source.get("credibility_score", 0),
                            "Summary": source.get("summary", "")[:500],
                        }
                    )

                if sources_data:
                    sources_df = pd.DataFrame(sources_data)
                    sources_df.to_excel(writer, sheet_name="Sources", index=False)

            # Key Insights sheet
            if "answer" in research_data and "key_insights" in research_data["answer"]:
                insights_data = []
                for i, insight in enumerate(research_data["answer"]["key_insights"], 1):
                    insights_data.append({"Index": i, "Insight": insight})

                if insights_data:
                    insights_df = pd.DataFrame(insights_data)
                    insights_df.to_excel(writer, sheet_name="Key Insights", index=False)

            # Answer sections sheet
            if "answer" in research_data and "sections" in research_data["answer"]:
                sections_data = []
                for section in research_data["answer"]["sections"]:
                    sections_data.append(
                        {
                            "Section": section.get("title", ""),
                            "Content": section.get("content", ""),
                        }
                    )

                if sections_data:
                    sections_df = pd.DataFrame(sections_data)
                    sections_df.to_excel(writer, sheet_name="Analysis", index=False)

        # Get Excel data
        excel_data = output.getvalue()
        output.close()

        # Create filename
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = (
            f"research_analysis_{research_data.get('id', 'unknown')}_{timestamp}.xlsx"
        )

        return ExportResult(
            format=ExportFormat.EXCEL,
            filename=filename,
            size_bytes=len(excel_data),
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            data=excel_data,
            metadata={
                "sheets": ["Overview", "Sources", "Key Insights", "Analysis"],
                "engine": "openpyxl",
            },
        )


# --- Markdown Exporter ---


class MarkdownExporter(BaseExporter):
    """Export research results to Markdown"""

    async def export(self, research_data: Dict[str, Any]) -> ExportResult:
        """Export research data to Markdown"""
        lines = []

        # Title
        title = (
            self.options.custom_title
            or f"Research Report: {research_data.get('query', 'Unknown Query')}"
        )
        lines.append(f"# {title}")
        lines.append("")

        # Metadata
        if self.options.include_metadata:
            lines.append("## Metadata")
            lines.append("")
            lines.append(f"- **Research ID**: {research_data.get('id', 'N/A')}")
            lines.append(f"- **Query**: {research_data.get('query', 'N/A')}")
            lines.append(
                f"- **Paradigm**: {research_data.get('paradigm', 'N/A').title()}"
            )
            lines.append(f"- **Depth**: {research_data.get('depth', 'N/A')}")
            lines.append(
                f"- **Confidence Score**: {research_data.get('confidence_score', 0) * 100:.1f}%"
            )
            lines.append(
                f"- **Sources Analyzed**: {research_data.get('sources_count', 0)}"
            )
            lines.append(
                f"- **Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
            )
            lines.append("")

        # Summary
        if self.options.include_summary and "summary" in research_data:
            lines.append("## Executive Summary")
            lines.append("")
            lines.append(research_data["summary"])
            lines.append("")

        # Answer sections
        if "answer" in research_data:
            paradigm = research_data.get("paradigm", "Analysis")
            lines.append(f"## {paradigm.title()} Analysis")
            lines.append("")

            if "sections" in research_data["answer"]:
                for section in research_data["answer"]["sections"]:
                    lines.append(f"### {section.get('title', 'Section')}")
                    lines.append("")
                    lines.append(section.get("content", ""))
                    lines.append("")

                    if "points" in section:
                        for point in section["points"]:
                            lines.append(f"- {point}")
                        lines.append("")

            if "key_insights" in research_data["answer"]:
                lines.append("### Key Insights")
                lines.append("")
                for insight in research_data["answer"]["key_insights"]:
                    lines.append(f"- {insight}")
                lines.append("")

        # Sources
        if self.options.include_sources and "sources" in research_data:
            lines.append("## Sources")
            lines.append("")
            for i, source in enumerate(research_data["sources"][:20], 1):
                lines.append(f"{i}. **{source.get('title', 'Untitled')}**")
                lines.append(f"   - URL: {source.get('url', 'No URL')}")
                if source.get("relevance_score"):
                    lines.append(f"   - Relevance: {source['relevance_score']:.2f}")
                if source.get("summary"):
                    lines.append(f"   - Summary: {source['summary'][:200]}...")
                lines.append("")

        # Citations
        if self.options.include_citations and "citations" in research_data:
            lines.append("## Citations")
            lines.append("")
            for citation in research_data["citations"]:
                lines.append(
                    f"> {citation.get('text', '')} [{citation.get('source_index', '?')}]"
                )
                lines.append("")

        # Footer
        lines.append("---")
        lines.append("")
        if self.options.custom_footer:
            lines.append(self.options.custom_footer)
        else:
            lines.append("*Generated by Four Hosts Research System*")

        # Convert to bytes
        markdown_data = "\n".join(lines).encode("utf-8")

        # Create filename
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = (
            f"research_report_{research_data.get('id', 'unknown')}_{timestamp}.md"
        )

        return ExportResult(
            format=ExportFormat.MARKDOWN,
            filename=filename,
            size_bytes=len(markdown_data),
            content_type="text/markdown",
            data=markdown_data,
            metadata={"encoding": "utf-8", "line_count": len(lines)},
        )


# --- Export Service ---


class ExportService:
    """Main export service"""

    def __init__(self):
        self.exporters = {
            ExportFormat.PDF: PDFExporter,
            ExportFormat.JSON: JSONExporter,
            ExportFormat.CSV: CSVExporter,
            ExportFormat.EXCEL: ExcelExporter,
            ExportFormat.MARKDOWN: MarkdownExporter,
        }

    async def export_research(
        self, research_data: Dict[str, Any], options: ExportOptions
    ) -> ExportResult:
        """Export research data in specified format"""
        exporter_class = self.exporters.get(options.format)
        if not exporter_class:
            raise ValueError(f"Unsupported export format: {options.format}")

        exporter = exporter_class(options)
        result = await exporter.export(research_data)

        logger.info(
            f"Exported research {research_data.get('id')} as {options.format} "
            f"({result.size_bytes} bytes)"
        )

        return result

    def get_supported_formats(self) -> List[str]:
        """Get list of supported export formats"""
        return [format.value for format in ExportFormat]

    async def export_multiple(
        self,
        research_data: Dict[str, Any],
        formats: List[ExportFormat],
        base_options: Optional[ExportOptions] = None,
    ) -> Dict[ExportFormat, ExportResult]:
        """Export research data in multiple formats"""
        results = {}

        for format in formats:
            options = base_options or ExportOptions(format=format)
            options.format = format

            try:
                result = await self.export_research(research_data, options)
                results[format] = result
            except Exception as e:
                logger.error(f"Failed to export as {format}: {e}")

        return results


# --- FastAPI Integration ---

from fastapi import APIRouter, Depends, HTTPException, Response
from services.auth import get_current_user, TokenData


def create_export_router(export_service: ExportService) -> APIRouter:
    """Create FastAPI router for export endpoints"""
    router = APIRouter(prefix="/export", tags=["export"])

    @router.post("/research/{research_id}")
    async def export_research(
        research_id: str,
        options: ExportOptions,
        current_user: TokenData = Depends(get_current_user),
    ):
        """Export research results in specified format"""
        # TODO: Fetch research data from database
        research_data = {
            "id": research_id,
            "query": "Sample research query",
            "paradigm": "maeve",
            "depth": "standard",
            "confidence_score": 0.85,
            "sources_count": 127,
            "summary": "This is a sample research summary...",
            "answer": {
                "sections": [
                    {
                        "title": "Strategic Overview",
                        "content": "Sample strategic analysis content...",
                    }
                ],
                "key_insights": ["Key insight 1", "Key insight 2"],
            },
            "sources": [
                {
                    "title": "Sample Source",
                    "url": "https://example.com",
                    "relevance_score": 0.9,
                    "summary": "Source summary...",
                }
            ],
        }

        try:
            result = await export_service.export_research(research_data, options)

            return Response(
                content=result.data,
                media_type=result.content_type,
                headers={
                    "Content-Disposition": f'attachment; filename="{result.filename}"',
                    "X-Export-Format": result.format,
                    "X-Export-Size": str(result.size_bytes),
                },
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/formats")
    async def get_export_formats(current_user: TokenData = Depends(get_current_user)):
        """Get supported export formats"""
        return {
            "formats": export_service.get_supported_formats(),
            "default": ExportFormat.PDF,
        }

    return router
