import tkinter as tk
from tkinter import messagebox, ttk

import joblib
import pandas as pd

from src.decisions import advisory_decision, validate_application_fields, validate_score
from src.data import load_data, print_audit
from src.model import (
    CLASSIFICATION_THRESHOLD,
    IMPROVED_FEATURES,
    MODEL_FILE,
    evaluate_on_holdout,
    run_improved_model,
)


class MortgageDecisionGUI:
    """Graphical interface for the mortgage decision-support prototype."""

    def __init__(self, master):
        self.master = master
        self.master.title("Mortgage Risk Advisor (Prototype)")
        self.model = None
        self.last_raw_results = None
        self.classification_threshold = CLASSIFICATION_THRESHOLD
        self.data = load_data()
        self.sample_data = self.data.dropna(subset=IMPROVED_FEATURES)
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
            value = tk.StringVar(master=self.master)
            value.trace_add("write", self.clear_raw_results)
            entry = ttk.Entry(self.input_frame, width=30, textvariable=value)
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
        self.raw_results_button = ttk.Button(
            button_frame,
            text="Show Raw Results",
            command=self.show_raw_results,
            state="disabled",
        )
        self.raw_results_button.pack(side="left", padx=5)

        threshold_frame = ttk.LabelFrame(
            self.master, text="Class 1 Classification Threshold"
        )
        threshold_frame.pack(padx=10, pady=5, fill="x")
        ttk.Label(
            threshold_frame,
            text="Flag class 1 when the predicted probability is at least:",
        ).pack(side="left", padx=5, pady=5)
        self.threshold_value = tk.StringVar(
            master=self.master,
            value=f"{self.classification_threshold:.2f}",
        )
        ttk.Spinbox(
            threshold_frame,
            from_=0.0,
            to=1.0,
            increment=0.05,
            textvariable=self.threshold_value,
            width=6,
        ).pack(side="left", padx=5, pady=5)
        ttk.Button(
            threshold_frame,
            text="Apply Threshold",
            command=self.apply_threshold,
        ).pack(side="left", padx=5, pady=5)
        ttk.Button(
            threshold_frame,
            text="Show Model Metrics",
            command=self.show_model_metrics,
        ).pack(side="left", padx=5, pady=5)

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

    def apply_threshold(self):
        """Validate and apply the class-1 probability cutoff for GUI predictions."""
        try:
            threshold = validate_score(float(self.threshold_value.get()))
        except ValueError as error:
            messagebox.showerror(
                "Invalid threshold",
                f"Enter a number between 0 and 1.\n\n{error}",
                parent=self.master,
            )
            return

        self.classification_threshold = threshold
        self.clear_raw_results()
        self.result_label.configure(
            text=(
                f"Classification threshold set to {threshold:.2f}. "
                "Select Run Assessment to score the current sample."
            )
        )

    def show_model_metrics(self):
        """Display held-out model metrics using the currently applied threshold."""
        results = evaluate_on_holdout(
            self.data,
            self.model,
            threshold=self.classification_threshold,
        )
        window = tk.Toplevel(self.master)
        window.title("Held-out Model Metrics")
        window.resizable(False, False)
        ttk.Label(
            window,
            text=f"Classification threshold: {self.classification_threshold:.2f}",
        ).grid(row=0, column=0, columnspan=2, padx=12, pady=(12, 6), sticky="w")
        ttk.Label(window, text="Measure", font=("", 10, "bold")).grid(
            row=1, column=0, padx=12, pady=4, sticky="w"
        )
        ttk.Label(window, text="Result", font=("", 10, "bold")).grid(
            row=1, column=1, padx=12, pady=4, sticky="e"
        )

        measures = (
            ("Accuracy", "accuracy"),
            ("Precision for class 1", "precision_1"),
            ("Recall for class 1", "recall_1"),
            ("F1 for class 1", "f1_1"),
            ("ROC AUC", "roc_auc"),
        )
        for row, (label, key) in enumerate(measures, start=2):
            ttk.Label(window, text=label).grid(
                row=row, column=0, padx=12, pady=3, sticky="w"
            )
            ttk.Label(window, text=f"{results[key]:.3f}").grid(
                row=row, column=1, padx=12, pady=3, sticky="e"
            )

        ttk.Label(
            window,
            text="ROC AUC uses probability scores and is not changed by the threshold.",
            wraplength=360,
        ).grid(row=7, column=0, columnspan=2, padx=12, pady=(8, 12), sticky="w")

    def clear_raw_results(self, *_args):
        """Discard results whenever an applicant input changes."""
        self.last_raw_results = None
        self.raw_results_button.configure(state="disabled")

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
        class_probabilities = self.model.predict_proba(row)[0]
        score = validate_score(float(class_probabilities[1]))
        predicted_class = int(score >= self.classification_threshold)
        decision = advisory_decision(score, application)
        self.last_raw_results = (
            "Application inputs:\n"
            + "\n".join(
                f"{field_name}: {application[field_name]}"
                for field_name in IMPROVED_FEATURES
            )
            + f"\n\nPredicted class: {predicted_class}\n"
            + "Class probabilities:\n"
            + "\n".join(
                f"class {class_label}: {float(probability)!r}"
                for class_label, probability in zip(
                    self.model.classes_, class_probabilities
                )
            )
            + f"\n\nClassification threshold: {self.classification_threshold:.2f}"
        )
        self.raw_results_button.configure(state="normal")
        self.result_label.configure(
            text=(
                f"Repayment-difficulty score: {score:.3f}\n"
                f"Predicted class (threshold "
                f"{self.classification_threshold:.2f}): {predicted_class}\n"
                f"Status: {decision['status']}\n"
                f"Risk level: {decision['risk_level']}\n"
                f"Recommendation: {decision['recommendation']}\n"
                f"Reason: {decision['reason']}\n\n"
                "Advisory only; a human reviewer must make any final decision."
            )
        )

    def show_raw_results(self):
        """Display the most recent raw model output in a separate window."""
        if self.last_raw_results is None:
            return

        window = tk.Toplevel(self.master)
        window.title("Raw Assessment Results")
        window.geometry("520x400")
        output_frame = ttk.Frame(window, padding=10)
        output_frame.pack(fill="both", expand=True)
        scrollbar = ttk.Scrollbar(output_frame)
        scrollbar.pack(side="right", fill="y")
        output = tk.Text(
            output_frame,
            wrap="word",
            yscrollcommand=scrollbar.set,
        )
        output.pack(side="left", fill="both", expand=True)
        scrollbar.configure(command=output.yview)
        output.insert("1.0", self.last_raw_results)
        output.configure(state="disabled")

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
