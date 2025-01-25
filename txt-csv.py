import csv

# Path to the extracted text file
extracted_text_file = r"C:\Users\kishore l\gdg\extracted_text.txt"

# Read the contents of the .txt file
with open(extracted_text_file, "r", encoding="utf-8") as file:
    extracted_text = file.readlines()

# Process the text into clauses (e.g., split by newline or chunked content)
clauses = [line.strip() for line in extracted_text if line.strip()]  # Remove empty lines and extra spaces

# Prepare data for the dataset
data = []
for i, clause in enumerate(clauses):
    # Assign a category and importance dynamically (you can adjust this logic based on your needs)
    category = "Consent" if "consent" in clause.lower() else "General"
    importance = 5 if "penalty" in clause.lower() or "fine" in clause.lower() else 3
    data.append({"Clause ID": i + 1, "Clause Text": clause, "Category": category, "Importance": importance})

# Save the data to a CSV file
csv_file = r"C:\Users\kishore l\gdg\dpdp_clauses.csv"
with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=["Clause ID", "Clause Text", "Category", "Importance"])
    writer.writeheader()
    writer.writerows(data)

print(f"Dataset saved to {csv_file}")
