"""
Report Generator — CSV export of full analysis results
"""
import io
import pandas as pd
import numpy as np
from fpdf import FPDF


def generate_report_csv(
    cleaned_df: pd.DataFrame,
    models: list | None,
    forecast_df: pd.DataFrame | None,
    feature_importance: pd.DataFrame | None,
) -> bytes:
    """
    Assemble a multi-section CSV report as bytes ready for st.download_button.
    """
    buf = io.StringIO()

    # ── Section 1: Summary stats ──────────────────────────────────────────────
    buf.write("=== REVENUE INTELLIGENCE REPORT ===\n\n")
    buf.write("--- SUMMARY STATISTICS ---\n")
    stats = {
        'Total Periods':     len(cleaned_df),
        'Total Revenue':     f"${cleaned_df['Revenue'].sum():,.2f}",
        'Average Revenue':   f"${cleaned_df['Revenue'].mean():,.2f}",
        'Max Revenue':       f"${cleaned_df['Revenue'].max():,.2f}",
        'Min Revenue':       f"${cleaned_df['Revenue'].min():,.2f}",
        'Std Dev':           f"${cleaned_df['Revenue'].std():,.2f}",
        'Start Date':        str(cleaned_df['Date'].min().date()),
        'End Date':          str(cleaned_df['Date'].max().date()),
    }
    for k, v in stats.items():
        buf.write(f"{k},{v}\n")
    buf.write("\n")

    # ── Section 2: Model evaluation ───────────────────────────────────────────
    if models:
        buf.write("--- MODEL EVALUATION ---\n")
        buf.write("Model,RMSE,MAE,R2\n")
        for m in models:
            buf.write(f"{m['name']},{m['rmse']:.2f},{m['mae']:.2f},{m['r2']:.4f}\n")
        buf.write("\n")

    # ── Section 3: Cleaned data ───────────────────────────────────────────────
    buf.write("--- CLEANED DATA ---\n")
    cleaned_df.to_csv(buf, index=False)
    buf.write("\n")

    # ── Section 4: Forecast ───────────────────────────────────────────────────
    if forecast_df is not None:
        buf.write("--- FORECAST ---\n")
        fdf = forecast_df.copy()
        fdf['Date'] = fdf['Date'].dt.strftime('%Y-%m-%d')
        fdf.to_csv(buf, index=False)
        buf.write("\n")

    # ── Section 5: Feature importance ────────────────────────────────────────
    if feature_importance is not None:
        buf.write("--- FEATURE IMPORTANCE ---\n")
        feature_importance.to_csv(buf, index=False)
        buf.write("\n")

    return buf.getvalue().encode()


# ✅ PDF REPORT FUNCTION
def generate_report_pdf(
    cleaned_df: pd.DataFrame,
    models: list | None,
    forecast_df: pd.DataFrame | None,
    feature_importance: pd.DataFrame | None,
    target_result: dict | None = None,
) -> bytes:
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.set_margins(left=20, top=20, right=20)
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    W = pdf.w - pdf.l_margin - pdf.r_margin

    def section_title(text: str):
        pdf.ln(4)
        pdf.set_font("Arial", "B", 13)
        pdf.set_fill_color(240, 242, 255)
        pdf.cell(W, 8, text, ln=True, fill=True)
        pdf.ln(2)

    def row_line(label: str, value: str, bold_label: bool = False):
        pdf.set_font("Arial", "B" if bold_label else "", 10)
        pdf.cell(70, 7, label, ln=False)
        pdf.set_font("Arial", "", 10)
        pdf.cell(W - 70, 7, value, ln=True)

    # ── Title ─────────────────────────────────────────────────────────────────
    pdf.set_font("Arial", "B", 18)
    pdf.cell(W, 12, "Revenue Intelligence Report", ln=True, align="C")
    pdf.set_font("Arial", "", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(W, 6, f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    # ── Summary Statistics ────────────────────────────────────────────────────
    section_title("Summary Statistics")
    for label, value in [
        ("Total Periods",   str(len(cleaned_df))),
        ("Total Revenue",   f"${cleaned_df['Revenue'].sum():,.2f}"),
        ("Average Revenue", f"${cleaned_df['Revenue'].mean():,.2f}"),
        ("Max Revenue",     f"${cleaned_df['Revenue'].max():,.2f}"),
        ("Min Revenue",     f"${cleaned_df['Revenue'].min():,.2f}"),
        ("Std Deviation",   f"${cleaned_df['Revenue'].std():,.2f}"),
        ("Start Date",      str(cleaned_df['Date'].min().date())),
        ("End Date",        str(cleaned_df['Date'].max().date())),
    ]:
        row_line(label, value)

    # ── Target Feasibility ────────────────────────────────────────────────────
    if target_result:
        section_title("Target Feasibility Result")
        tr = target_result

        # Strip ALL HTML tags and emojis — fpdf only supports latin characters
        import re
        def _clean(text: str) -> str:
            text = re.sub(r'<[^>]+>', '', str(text))          # remove HTML tags
            text = re.sub(r'[^\x00-\x7F]+', '', text)         # remove non-ASCII (emojis etc.)
            return text.strip()

        clean_label = _clean(tr["label"])
        clean_note  = _clean(tr["note"])
        clean_rec   = _clean(tr["recommendation"])

        # Verdict label as bold heading
        pdf.set_font("Arial", "B", 12)
        pdf.cell(W, 8, clean_label, ln=True)
        pdf.set_font("Arial", "", 10)
        pdf.ln(2)

        for label, value in [
            ("Target Revenue",    f"${tr['target']:,.0f}"),
            ("ML Forecast",       f"${tr['forecast']:,.0f}"),
            ("Forecast Upper CI", f"${tr['upper']:,.0f}"),
            ("Forecast Lower CI", f"${tr['lower']:,.0f}"),
            ("Growth Required",   f"{tr['req_growth']:+.1f}%"),
            ("Check Period",      _clean(tr["period"])),
        ]:
            row_line(label, value)

        pdf.ln(2)
        pdf.set_font("Arial", "I", 10)
        pdf.multi_cell(W, 6, clean_note)
        pdf.ln(2)
        pdf.set_font("Arial", "B", 10)
        pdf.cell(45, 7, "Recommendation:", ln=False)
        pdf.set_font("Arial", "", 10)
        pdf.multi_cell(W - 45, 7, clean_rec)

    # ── Model Evaluation ──────────────────────────────────────────────────────
    if models:
        section_title("Model Evaluation")
        best = min(models, key=lambda m: m['rmse'])
        pdf.set_font("Arial", "B", 9)
        col_w = [60, 30, 30, 25, 30]
        for h, w in zip(["Model", "RMSE ($)", "MAE ($)", "R²", "CV RMSE"], col_w):
            pdf.cell(w, 7, h, border=1, ln=False, align="C")
        pdf.ln()
        pdf.set_font("Arial", "", 9)
        for m in models:
            is_best = m['name'] == best['name']
            if is_best:
                pdf.set_fill_color(235, 255, 240)
            cv = f"${m.get('cv_rmse_mean', m['rmse']):,.0f}"
            for val, w in zip([m['name'], f"${m['rmse']:,.0f}", f"${m['mae']:,.0f}",
                                f"{m['r2']:.4f}", cv], col_w):
                pdf.cell(w, 7, val, border=1, ln=False, align="C", fill=is_best)
            pdf.ln()
            if is_best:
                pdf.set_fill_color(255, 255, 255)

    # ── Feature Importance ────────────────────────────────────────────────────
    if feature_importance is not None:
        section_title("Top Revenue Drivers (SHAP)")
        pdf.set_font("Arial", "B", 9)
        pdf.cell(100, 7, "Feature", border=1, ln=False)
        pdf.cell(W - 100, 7, "Importance (%)", border=1, ln=True, align="C")
        pdf.set_font("Arial", "", 9)
        for _, fi_row in feature_importance.head(8).iterrows():
            pdf.cell(100, 7, str(fi_row['Feature']), border=1, ln=False)
            pdf.cell(W - 100, 7, f"{fi_row['Importance']:.1f}%", border=1, ln=True, align="C")

    # ── Forecast Data ─────────────────────────────────────────────────────────
    if forecast_df is not None:
        section_title("Forecast Data")
        pdf.set_font("Arial", "B", 9)
        cw = [W / 4] * 4
        for h in ["Date", "Forecast ($)", "Lower CI ($)", "Upper CI ($)"]:
            pdf.cell(cw[0], 7, h, border=1, ln=False, align="C")
        pdf.ln()
        pdf.set_font("Arial", "", 9)
        for _, frow in forecast_df.iterrows():
            for val, w in zip([
                frow['Date'].strftime('%Y-%m-%d'),
                f"${frow['Forecast']:,.0f}",
                f"${frow['Lower']:,.0f}",
                f"${frow['Upper']:,.0f}",
            ], cw):
                pdf.cell(w, 7, val, border=1, ln=False, align="C")
            pdf.ln()

    return bytes(pdf.output(dest='S'))
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    # Explicit margins: left=20, top=20, right=20  →  usable width = 210-40 = 170mm
    pdf.set_margins(left=20, top=20, right=20)
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Usable width for cells/multi_cell
    W = pdf.w - pdf.l_margin - pdf.r_margin   # 170mm

    # ── Helper: section heading ───────────────────────────────────────────────
    def section_title(text: str):
        pdf.ln(4)
        pdf.set_font("Arial", "B", 13)
        pdf.set_fill_color(240, 242, 255)
        pdf.cell(W, 8, text, ln=True, fill=True)
        pdf.ln(2)

    def row_line(label: str, value: str, bold_label: bool = False):
        """Print a label+value pair on one line."""
        pdf.set_font("Arial", "B" if bold_label else "", 10)
        pdf.cell(70, 7, label, ln=False)
        pdf.set_font("Arial", "", 10)
        pdf.cell(W - 70, 7, value, ln=True)

    # ── Title ─────────────────────────────────────────────────────────────────
    pdf.set_font("Arial", "B", 18)
    pdf.cell(W, 12, "Revenue Intelligence Report", ln=True, align="C")
    pdf.set_font("Arial", "", 9)
    pdf.set_text_color(120, 120, 120)
    generated_on = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    pdf.cell(W, 6, f"Generated: {generated_on}", ln=True, align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    # ── Summary Statistics ────────────────────────────────────────────────────
    section_title("Summary Statistics")
    stats = [
        ("Total Periods",   str(len(cleaned_df))),
        ("Total Revenue",   f"${cleaned_df['Revenue'].sum():,.2f}"),
        ("Average Revenue", f"${cleaned_df['Revenue'].mean():,.2f}"),
        ("Max Revenue",     f"${cleaned_df['Revenue'].max():,.2f}"),
        ("Min Revenue",     f"${cleaned_df['Revenue'].min():,.2f}"),
        ("Std Deviation",   f"${cleaned_df['Revenue'].std():,.2f}"),
        ("Start Date",      str(cleaned_df['Date'].min().date())),
        ("End Date",        str(cleaned_df['Date'].max().date())),
    ]
    for label, value in stats:
        row_line(label, value)

    # ── Model Evaluation ──────────────────────────────────────────────────────
    if models:
        section_title("Model Evaluation")
        best = min(models, key=lambda m: m['rmse'])

        # Header row
        pdf.set_font("Arial", "B", 9)
        col_w = [70, 30, 30, 25, 20]
        headers = ["Model", "RMSE ($)", "MAE ($)", "R²", ""]
        for i, h in enumerate(headers):
            pdf.cell(col_w[i], 7, h, border=1, ln=False, align="C")
        pdf.ln()

        pdf.set_font("Arial", "", 9)
        for m in models:
            is_best = m['name'] == best['name']
            if is_best:
                pdf.set_fill_color(235, 255, 240)
            vals = [
                m['name'],
                f"${m['rmse']:,.0f}",
                f"${m['mae']:,.0f}",
                f"{m['r2']:.4f}",
                "BEST" if is_best else "",
            ]
            for i, v in enumerate(vals):
                pdf.cell(col_w[i], 7, v, border=1, ln=False, align="C", fill=is_best)
            pdf.ln()
            if is_best:
                pdf.set_fill_color(255, 255, 255)

    # ── Feature Importance ────────────────────────────────────────────────────
    if feature_importance is not None:
        section_title("Top Revenue Drivers")
        pdf.set_font("Arial", "B", 9)
        pdf.cell(100, 7, "Feature", border=1, ln=False)
        pdf.cell(W - 100, 7, "Importance (%)", border=1, ln=True, align="C")
        pdf.set_font("Arial", "", 9)
        for _, fi_row in feature_importance.head(8).iterrows():
            pdf.cell(100, 7, str(fi_row['Feature']), border=1, ln=False)
            pdf.cell(W - 100, 7, f"{fi_row['Importance']:.1f}%", border=1, ln=True, align="C")

    # ── Forecast Data ─────────────────────────────────────────────────────────
    if forecast_df is not None:
        section_title("Forecast Data")

        # Table header
        pdf.set_font("Arial", "B", 9)
        cw = [W / 4] * 4
        for h in ["Date", "Forecast ($)", "Lower CI ($)", "Upper CI ($)"]:
            pdf.cell(cw[0], 7, h, border=1, ln=False, align="C")
        pdf.ln()

        pdf.set_font("Arial", "", 9)
        for _, frow in forecast_df.iterrows():
            cells = [
                frow['Date'].strftime('%Y-%m-%d'),
                f"${frow['Forecast']:,.0f}",
                f"${frow['Lower']:,.0f}",
                f"${frow['Upper']:,.0f}",
            ]
            for i, c in enumerate(cells):
                pdf.cell(cw[i], 7, c, border=1, ln=False, align="C")
            pdf.ln()

    return bytes(pdf.output(dest='S'))
