"""
Generates a realistic multi-column medical case report PDF for testing and verification.
Includes title, authors, two-column body text, embedded clinical and radiographic figures,
captions, and references.
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import pymupdf


def create_test_image(text: str, bg_color: str, filename: str) -> str:
    """Creates a sample medical figure image."""
    img = Image.new("RGB", (400, 260), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Draw border
    draw.rectangle([5, 5, 395, 255], outline="#1e293b", width=2)
    draw.rectangle([10, 10, 390, 250], outline="#cbd5e1", width=1)

    # Draw label text
    draw.text((30, 120), text, fill="#0f172a")
    img.save(filename, "PNG")
    return filename


def generate_case_report_pdf(output_path: str = "sample_case_report.pdf") -> str:
    doc = pymupdf.open()

    # Create figure image assets
    fig1_file = create_test_image("Figure 1: Intraoral Clinical Photo (Crown Fracture)", "#e0f2fe", "temp_fig1.png")
    fig2_file = create_test_image("Figure 2: Periapical Radiograph (Root Canal)", "#f1f5f9", "temp_fig2.png")

    # Page 1: Title, Authors, Abstract (full width), then 2 columns
    page1 = doc.new_page(width=595, height=842)  # A4 size

    # Journal Header
    page1.insert_text((50, 40), "Case Reports in Dentistry / Hindawi Publishing Corporation", fontsize=8, color=(0.4, 0.4, 0.4))

    # Title (Large, Bold)
    title_rect = pymupdf.Rect(50, 60, 545, 110)
    page1.insert_textbox(
        title_rect,
        "Complicated Crown-Root Fracture Treated Using Reattachment Procedure: A Single Visit Technique",
        fontsize=16,
        fontname="helv",
        color=(0.1, 0.2, 0.5)
    )

    # Authors
    author_rect = pymupdf.Rect(50, 115, 545, 140)
    page1.insert_textbox(
        author_rect,
        "Department of Conservative Dentistry and Endodontics, Dental College, Clinical Hospital",
        fontsize=9,
        fontname="helv",
        color=(0.3, 0.3, 0.3)
    )

    # Abstract Header & Box
    page1.insert_text((50, 155), "Abstract", fontsize=11, fontname="helv", color=(0.1, 0.2, 0.5))
    abs_rect = pymupdf.Rect(50, 165, 545, 240)
    abstract_text = (
        "Trauma to the oral and maxillofacial region occurs frequently, and crown-root fractures comprise "
        "a challenging clinical condition. In this report, a single-visit reattachment procedure using a fiber "
        "post and composite resin is presented. The patient presented with a fractured maxillary central incisor. "
        "A follow-up periapical radiograph demonstrated excellent healing of the periodontal ligament with no "
        "radiolucent lesion."
    )
    page1.insert_textbox(abs_rect, abstract_text, fontsize=9.5, fontname="helv")

    # Divider line
    page1.draw_line((50, 250), (545, 250), color=(0.8, 0.8, 0.8), width=0.5)

    # 2 Column Section:
    # Column 1: x from 50 to 285
    # Column 2: x from 310 to 545

    # Column 1 - Introduction
    page1.insert_text((50, 270), "1. Introduction", fontsize=11, fontname="helv", color=(0.1, 0.2, 0.5))
    col1_p1_rect = pymupdf.Rect(50, 280, 285, 380)
    col1_p1_text = (
        "Crown-root fractures are dental injuries involving enamel, dentin, and cementum, usually extending apical "
        "to the gingival margin. Treatment options depend on the fracture line location, biological width invasion, "
        "and pulpal exposure."
    )
    page1.insert_textbox(col1_p1_rect, col1_p1_text, fontsize=9, fontname="helv")

    col1_p2_rect = pymupdf.Rect(50, 390, 285, 500)
    col1_p2_text = (
        "When the fractured fragment is intact and available, reattachment procedure is considered the most "
        "conservative approach. It restores the original tooth morphology, aesthetics, and incisal function while "
        "maintaining biological compatibility."
    )
    page1.insert_textbox(col1_p2_rect, col1_p2_text, fontsize=9, fontname="helv")

    # Column 2 - Case Report
    page1.insert_text((310, 270), "Case Report", fontsize=11, fontname="helv", color=(0.1, 0.2, 0.5))
    col2_p1_rect = pymupdf.Rect(310, 280, 545, 410)
    col2_p1_text = (
        "A 22-year-old male patient reported to the department following a bicycle fall. Clinical examination revealed "
        "a complicated crown-root fracture on the permanent maxillary central incisor with pulpal exposure. "
        "The fractured coronal fragment was retrieved in physiological saline."
    )
    page1.insert_textbox(col2_p1_rect, col2_p1_text, fontsize=9, fontname="helv")

    # Figure 1 inside Column 2!
    # Insert Figure 1 image
    fig1_rect = pymupdf.Rect(315, 420, 540, 560)
    page1.insert_image(fig1_rect, filename=fig1_file)

    # Caption 1 below Figure 1
    cap1_rect = pymupdf.Rect(310, 570, 545, 620)
    cap1_text = "Figure 1. Preoperative intraoral photograph showing the complicated crown-root fracture with pulp exposure."
    page1.insert_textbox(cap1_rect, cap1_text, fontsize=8.5, fontname="helv", color=(0.2, 0.2, 0.2))

    # Follow-up text in Column 2 after Figure 1
    col2_p2_rect = pymupdf.Rect(310, 630, 545, 760)
    col2_p2_text = (
        "After administration of local anesthesia and rubber dam isolation, single-visit endodontic treatment "
        "was executed. A glass fiber post was cemented into the canal, and the tooth fragment was reattached "
        "using dual-cure composite resin."
    )
    page1.insert_textbox(col2_p2_rect, col2_p2_text, fontsize=9, fontname="helv")

    # Page 2: Discussion, Figure 2, Conclusion, References
    page2 = doc.new_page(width=595, height=842)

    # Page 2 Column 1:
    page2.insert_text((50, 50), "Discussion", fontsize=11, fontname="helv", color=(0.1, 0.2, 0.5))
    p2_c1_p1_rect = pymupdf.Rect(50, 65, 285, 210)
    p2_c1_p1_text = (
        "Management of subgingival fractures often requires surgical crown lengthening, gingivectomy, or osteotomy. "
        "However, when minimal invasion occurs, conservative reattachment provides optimal psychological and functional "
        "outcomes for the patient. Histopathology studies demonstrate favorable periodontal response."
    )
    page2.insert_textbox(p2_c1_p1_rect, p2_c1_p1_text, fontsize=9, fontname="helv")

    # Figure 2 inside Page 2 Column 1
    fig2_rect = pymupdf.Rect(55, 220, 280, 360)
    page2.insert_image(fig2_rect, filename=fig2_file)

    # Caption 2 below Figure 2
    cap2_rect = pymupdf.Rect(50, 370, 285, 420)
    cap2_text = "Figure 2. Follow-up periapical radiograph at 12 months showing intact periodontal ligament without radiolucent lesion."
    page2.insert_textbox(cap2_rect, cap2_text, fontsize=8.5, fontname="helv", color=(0.2, 0.2, 0.2))

    p2_c1_p2_rect = pymupdf.Rect(50, 430, 285, 540)
    p2_c1_p2_text = (
        "The patient was recalled after 3, 6, and 12 months for clinical evaluation. The restored tooth remained "
        "asymptomatic and fully functional."
    )
    page2.insert_textbox(p2_c1_p2_rect, p2_c1_p2_text, fontsize=9, fontname="helv")

    # Page 2 Column 2: Conclusion & References
    page2.insert_text((310, 50), "Conclusion", fontsize=11, fontname="helv", color=(0.1, 0.2, 0.5))
    p2_c2_p1_rect = pymupdf.Rect(310, 65, 545, 160)
    p2_c2_p1_text = (
        "Reattachment of fractured tooth fragments is an effective, conservative treatment alternative that preserves "
        "natural dental tissue and ensures immediate aesthetic recovery."
    )
    page2.insert_textbox(p2_c2_p1_rect, p2_c2_p1_text, fontsize=9, fontname="helv")

    # References
    page2.insert_text((310, 180), "References", fontsize=11, fontname="helv", color=(0.1, 0.2, 0.5))
    ref_rect = pymupdf.Rect(310, 195, 545, 360)
    ref_text = (
        "[1] Andreasen JO, Andreasen FM. Essentials of Traumatic Dental Injuries. 2nd ed. Copenhagen: Munksgaard, 2000.\n"
        "[2] Baratieri LN, Ritter AV, Monteiro S Jr. Tooth fragment reattachment: an alternative for restoration of fractured anterior teeth. Pract Periodontics Aesthet Dent. 1998;10(1):115-125.\n"
        "[3] Simonsen RJ. Traumatic aspects of restorative dentistry. Quintessence Int. 2001;32(9):679-688."
    )
    page2.insert_textbox(ref_rect, ref_text, fontsize=8.5, fontname="helv")

    doc.save(output_path)
    doc.close()

    # Clean up temp image files
    Path(fig1_file).unlink(missing_ok=True)
    Path(fig2_file).unlink(missing_ok=True)

    print(f"Successfully generated realistic test PDF: {output_path}")
    return output_path


if __name__ == "__main__":
    generate_case_report_pdf()
