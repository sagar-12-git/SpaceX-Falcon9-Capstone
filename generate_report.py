
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, PageBreak
)

output_file = "SpaceX_Falcon9_Project_Report.pdf"

doc = SimpleDocTemplate(
    output_file,
    pagesize=A4,
    rightMargin=2 * cm,
    leftMargin=2 * cm,
    topMargin=1.8 * cm,
    bottomMargin=1.8 * cm
)

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="CoverTitle",
    parent=styles["Title"],
    fontSize=23,
    leading=30,
    alignment=TA_CENTER,
    spaceAfter=20
))
styles.add(ParagraphStyle(
    name="SectionHeading",
    parent=styles["Heading1"],
    fontSize=15,
    leading=19,
    spaceBefore=12,
    spaceAfter=8
))
styles.add(ParagraphStyle(
    name="BodyTextCustom",
    parent=styles["BodyText"],
    fontSize=10,
    leading=15,
    spaceAfter=8
))

story = []

def paragraph(text, style="BodyTextCustom"):
    story.append(Paragraph(text, styles[style]))

# Cover page
story.append(Spacer(1, 4 * cm))
paragraph("SPACE X FALCON 9", "CoverTitle")
paragraph("First-Stage Landing Prediction", "CoverTitle")
story.append(Spacer(1, 1 * cm))
paragraph("Machine Learning Capstone Project", "Heading2")
story.append(Spacer(1, 2 * cm))
paragraph("Student Name: __________________________")
paragraph("USN / Roll Number: ______________________")
paragraph("Department: Computer Science and Data Science")
paragraph("College: _________________________________")
paragraph("Academic Year: __________________________")
story.append(Spacer(1, 2 * cm))
paragraph("Project Report", "Heading2")
story.append(PageBreak())

# Abstract
paragraph("Abstract", "SectionHeading")
paragraph(
    "This project explores machine-learning techniques for predicting "
    "the success of SpaceX Falcon 9 first-stage landings. A dataset "
    "containing 90 launch records and 18 original columns was examined. "
    "Logistic Regression and Decision Tree classifiers were evaluated. "
    "Five-fold cross-validation on the training portion produced mean "
    "accuracies of 84.76% and 80.38%, respectively. The results are "
    "preliminary because the dataset is small and further validation "
    "is required."
)

# Introduction
paragraph("1. Introduction", "SectionHeading")
paragraph(
    "Reusable rocket boosters can reduce launch costs when they can be "
    "recovered and reused. Predicting landing outcomes is a useful "
    "machine-learning exercise because outcomes can be studied in "
    "relation to launch and booster characteristics."
)

paragraph("2. Problem Statement", "SectionHeading")
paragraph(
    "To investigate whether features associated with a Falcon 9 launch "
    "can be used to classify the first-stage landing outcome as successful "
    "or unsuccessful."
)

paragraph("3. Objectives", "SectionHeading")
for item in [
    "Explore and understand the launch dataset.",
    "Identify the target variable and prepare input features.",
    "Train Logistic Regression and Decision Tree classifiers.",
    "Compare model performance using cross-validation.",
    "Save a reusable trained model and demonstrate a prediction."
]:
    paragraph("• " + item)

paragraph("4. Tools and Technologies", "SectionHeading")
paragraph(
    "Python 3.13, Pandas, Scikit-learn, Matplotlib, ReportLab and Joblib."
)

paragraph("5. Dataset Description", "SectionHeading")
paragraph(
    "The dataset file dataset_part_2.csv contains 90 rows and 18 columns. "
    "The target column is Class: 1 represents a successful landing and "
    "0 represents an unsuccessful landing. Initial inspection found "
    "26 missing values in LandingPad and no duplicate rows. The observed "
    "class distribution was 60 successful and 30 unsuccessful records."
)

paragraph("6. Methodology", "SectionHeading")
paragraph(
    "The data was loaded using Pandas. The Date column and selected "
    "identifier or landing-related fields were excluded from the later "
    "model experiment. Numerical missing values were imputed using the "
    "median and scaled. Categorical missing values were imputed using "
    "the most frequent category and encoded using one-hot encoding. "
    "A stratified 80/20 train-test split was created. Five-fold "
    "stratified cross-validation was then applied to the training split."
)

paragraph("7. Models Used", "SectionHeading")
paragraph(
    "<b>Logistic Regression:</b> A classification algorithm that estimates "
    "the probability of a class using a logistic function."
)
paragraph(
    "<b>Decision Tree:</b> A model that learns a sequence of feature-based "
    "decisions to classify examples."
)

paragraph("8. Experimental Results", "SectionHeading")
data = [
    ["Model", "Mean CV Accuracy", "Standard Deviation"],
    ["Logistic Regression", "84.76%", "6.43%"],
    ["Decision Tree", "80.38%", "12.04%"]
]
table = Table(data, colWidths=[6 * cm, 4 * cm, 4.5 * cm])
table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE6F1")),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("PADDING", (0, 0), (-1, -1), 7),
]))
story.append(table)
story.append(Spacer(1, 10))
paragraph(
    "Logistic Regression achieved the higher mean cross-validation "
    "accuracy in this experiment. These values are not a guarantee of "
    "performance on future launches. The final saved demonstration model "
    "was fitted on all 90 records; therefore, its training predictions "
    "are not independent test results."
)

paragraph("9. Prediction Demonstration", "SectionHeading")
paragraph(
    "The prediction script successfully loaded the saved model and "
    "generated a class prediction for an example from the dataset. "
    "For the first row, the predicted class was 0, with an estimated "
    "successful-landing probability of 8.47%. Because this example was "
    "part of the training data, this output is a demonstration only."
)

paragraph("10. Limitations", "SectionHeading")
for item in [
    "Only 90 records were available for this experiment.",
    "Cross-validation scores can vary with small datasets.",
    "Feature selection and possible information leakage need further review.",
    "The prediction example was taken from the training dataset.",
    "The model is educational and is not suitable for operational launch decisions."
]:
    paragraph("• " + item)

paragraph("11. Conclusion", "SectionHeading")
paragraph(
    "The project demonstrated a complete introductory machine-learning "
    "workflow: data exploration, preprocessing, model training, "
    "cross-validation, model comparison and model saving. Logistic "
    "Regression had a mean cross-validation accuracy of 84.76%, compared "
    "with 80.38% for the Decision Tree. More data and stronger validation "
    "are needed before drawing reliable conclusions about real-world "
    "landing prediction."
)

paragraph("12. Future Scope", "SectionHeading")
for item in [
    "Collect and verify a larger, traceable dataset.",
    "Perform time-aware evaluation using earlier launches for training and later launches for testing.",
    "Review feature availability to prevent target leakage.",
    "Evaluate additional classifiers and calibration methods.",
    "Build a simple web interface for educational demonstrations."
]:
    paragraph("• " + item)

paragraph("References", "SectionHeading")
paragraph(
    "1. Scikit-learn documentation: https://scikit-learn.org/<br/>"
    "2. Pandas documentation: https://pandas.pydata.org/docs/<br/>"
    "3. SpaceX API: https://github.com/r-spacex/SpaceX-API"
)

doc.build(story)
print("Report generated successfully:", output_file)
