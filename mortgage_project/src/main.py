import tkinter as tk
from tkinter import messagebox, ttk

import joblib
import pandas as pd

from src.decisions import advisory_decision, validate_application_fields, validate_score
from src.data import load_data, print_audit
from src.model import IMPROVED_FEATURES, MODEL_FILE, run_improved_model


class MortgageDecisionGUI:
    """Graphical interface for the mortgage decision-support prototype."""

    def __init__(self, master):
        self.master = master
        self.master.title("Mortgage Risk Advisor (Prototype)")
        self.model = None
        self.sample_data = load_data().dropna(subset=IMPROVED_FEATURES)
        if self.sample_data.empty:
            raise ValueError(
                "The supplied dataset has no rows with complete model features "
                "to use as a sample application."
            )

        try:
            self.model = joblib.load(MODEL_FILE)
        except FileNotFoundError:
            messagebox.showerror(
                "Trained model not found",
                f"No trained model was found at:\n{MODEL_FILE}\n\n"
                "Train the model first by running `python -m src.model` "
                "from the project folder.",
                parent=self.master,
            )
            self.master.after_idle(self.master.destroy)
            return

        self.setup_ui()

    def setup_ui(self):
        """Create the applicant input form and assessment output."""
        self.input_frame = ttk.LabelFrame(self.master, text="Applicant Data Input")
        self.input_frame.pack(padx=10, pady=10, fill="x")

        self.entries = {}
        for row, field_name in enumerate(IMPROVED_FEATURES):
            label = field_name.replace("_", " ").title()
            ttk.Label(self.input_frame, text=f"{label}:").grid(
                row=row, column=0, padx=5, pady=5, sticky="w"
            )
            entry = ttk.Entry(self.input_frame, width=30)
            entry.grid(row=row, column=1, padx=5, pady=5, sticky="ew")
            self.entries[field_name] = entry

        self.input_frame.columnconfigure(1, weight=1)
        button_frame = ttk.Frame(self.master)
        button_frame.pack(padx=10, pady=5)
        ttk.Button(
            button_frame,
            text="Randomize Sample",
            command=self.fill_random_sample,
        ).pack(side="left", padx=5)
        ttk.Button(
            button_frame, text="Run Assessment", command=self.run_assessment
        ).pack(side="left", padx=5)

        self.output_frame = ttk.LabelFrame(
            self.master, text="Decision Support Outcome"
        )
        self.output_frame.pack(padx=10, pady=10, fill="both", expand=True)
        self.result_label = ttk.Label(
            self.output_frame,
            text="Enter the applicant data and select Run Assessment.",
            justify="left",
            wraplength=460,
        )
        self.result_label.pack(padx=10, pady=10, anchor="w")
        self.fill_random_sample()

    def fill_random_sample(self):
        """Fill the form with one complete synthetic dataset example."""
        sample = self.sample_data.sample(n=1).iloc[0]
        for field_name, entry in self.entries.items():
            entry.delete(0, tk.END)
            entry.insert(0, str(sample[field_name]))

        self.result_label.configure(
            text=(
                "A random synthetic sample is loaded. "
                "Select Run Assessment to score it."
            )
        )

    def run_assessment(self):
        """Validate form input, score the application, and show advisory results."""
        application = {
            field_name: entry.get().strip()
            for field_name, entry in self.entries.items()
        }

        try:
            validate_application_fields(application, IMPROVED_FEATURES)
            application = {
                field_name: float(value)
                for field_name, value in application.items()
            }
            validate_application_fields(application, IMPROVED_FEATURES)
        except ValueError as error:
            messagebox.showerror("Invalid applicant data", str(error), parent=self.master)
            return

        row = pd.DataFrame([application], columns=IMPROVED_FEATURES)
        score = validate_score(float(self.model.predict_proba(row)[0, 1]))
        decision = advisory_decision(score, application)
        self.result_label.configure(
            text=(
                f"Repayment-difficulty score: {score:.3f}\n"
                f"Status: {decision['status']}\n"
                f"Risk level: {decision['risk_level']}\n"
                f"Recommendation: {decision['recommendation']}\n"
                f"Reason: {decision['reason']}\n\n"
                "Advisory only; a human reviewer must make any final decision."
            )
        )


def main():
    if not MODEL_FILE.is_file():
        data = load_data()
        print_audit(data)
        run_improved_model(data)

    root = tk.Tk()
    MortgageDecisionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
