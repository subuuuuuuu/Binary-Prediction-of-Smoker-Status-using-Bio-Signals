import matplotlib.pyplot as plt
import seaborn as sns

# Feature labels map (human-readable clinical names)
FEATURE_METADATA = {
    'age': {
        'label': 'Age (Years)',
        'min': 20.0, 'max': 85.0, 'median': 40.0,
        'help': 'Age of the individual in years.'
    },
    'height(cm)': {
        'label': 'Height (cm)',
        'min': 135.0, 'max': 190.0, 'median': 165.0,
        'help': 'Height in centimeters.'
    },
    'weight(kg)': {
        'label': 'Weight (kg)',
        'min': 30.0, 'max': 130.0, 'median': 65.0,
        'help': 'Weight in kilograms.'
    },
    'waist(cm)': {
        'label': 'Waist Circumference (cm)',
        'min': 51.0, 'max': 127.0, 'median': 83.0,
        'help': 'Waist circumference in centimeters.'
    },
    'eyesight(left)': {
        'label': 'Eyesight (Left)',
        'min': 0.1, 'max': 9.9, 'median': 1.0,
        'help': 'Visual acuity score for the left eye (normally 0.1 to 2.0+).'
    },
    'eyesight(right)': {
        'label': 'Eyesight (Right)',
        'min': 0.1, 'max': 9.9, 'median': 1.0,
        'help': 'Visual acuity score for the right eye (normally 0.1 to 2.0+).'
    },
    'hearing(left)': {
        'label': 'Hearing (Left)',
        'min': 1, 'max': 2, 'median': 1,
        'help': 'Hearing ability score for the left ear (1 = Normal, 2 = Impaired).'
    },
    'hearing(right)': {
        'label': 'Hearing (Right)',
        'min': 1, 'max': 2, 'median': 1,
        'help': 'Hearing ability score for the right ear (1 = Normal, 2 = Impaired).'
    },
    'systolic': {
        'label': 'Systolic Blood Pressure (mmHg)',
        'min': 77.0, 'max': 213.0, 'median': 121.0,
        'help': 'Peak arterial pressure during heart contraction.'
    },
    'relaxation': {
        'label': 'Diastolic / Relaxation Pressure (mmHg)',
        'min': 44.0, 'max': 133.0, 'median': 78.0,
        'help': 'Minimum arterial pressure during heart relaxation.'
    },
    'fasting blood sugar': {
        'label': 'Fasting Blood Sugar (mg/dL)',
        'min': 46.0, 'max': 375.0, 'median': 96.0,
        'help': 'Blood sugar levels measured after fasting.'
    },
    'Cholesterol': {
        'label': 'Total Cholesterol (mg/dL)',
        'min': 77.0, 'max': 393.0, 'median': 196.0,
        'help': 'Total concentration of cholesterol in blood.'
    },
    'triglyceride': {
        'label': 'Triglycerides (mg/dL)',
        'min': 8.0, 'max': 766.0, 'median': 115.0,
        'help': 'Type of fat found in blood.'
    },
    'HDL': {
        'label': 'HDL Cholesterol (mg/dL)',
        'min': 9.0, 'max': 136.0, 'median': 54.0,
        'help': 'High-Density Lipoprotein (good cholesterol).'
    },
    'LDL': {
        'label': 'LDL Cholesterol (mg/dL)',
        'min': 1.0, 'max': 1860.0, 'median': 114.0,
        'help': 'Low-Density Lipoprotein (bad cholesterol).'
    },
    'hemoglobin': {
        'label': 'Hemoglobin (g/dL)',
        'min': 4.9, 'max': 21.0, 'median': 15.0,
        'help': 'Protein in red blood cells that carries oxygen.'
    },
    'Urine protein': {
        'label': 'Urine Protein Score',
        'min': 1, 'max': 6, 'median': 1,
        'help': 'Protein content in urine test (normally 1, higher indicates potential kidney strain).'
    },
    'serum creatinine': {
        'label': 'Serum Creatinine (mg/dL)',
        'min': 0.1, 'max': 9.9, 'median': 0.9,
        'help': 'Waste product filtered by kidneys (high values suggest kidney strain).'
    },
    'AST': {
        'label': 'Aspartate Aminotransferase / AST (IU/L)',
        'min': 6.0, 'max': 778.0, 'median': 24.0,
        'help': 'Liver enzyme levels (elevated levels may suggest liver strain).'
    },
    'ALT': {
        'label': 'Alanine Aminotransferase / ALT (IU/L)',
        'min': 1.0, 'max': 2914.0, 'median': 22.0,
        'help': 'Liver enzyme levels (elevated levels may suggest liver strain).'
    },
    'Gtp': {
        'label': 'Gamma-glutamyl Transferase / GTP (IU/L)',
        'min': 2.0, 'max': 999.0, 'median': 27.0,
        'help': 'Liver enzyme indicator heavily associated with alcohol/smoking impact.'
    },
    'dental caries': {
        'label': 'Dental Caries Presence',
        'min': 0, 'max': 1, 'median': 0,
        'help': 'Presence of tooth decay/cavities (0 = No, 1 = Yes).'
    }
}


def make_comparison_plot(feature_name, user_value):
    """Generates a matplotlib figure comparing the user value to the general population distribution."""
    # We use a preset normal-like distribution based on the stats to avoid loading a large CSV
    meta = FEATURE_METADATA[feature_name]
    
    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(6, 3))
    
    # We simulate a population curve around the median and min/max ranges for visualization
    import numpy as np
    median = meta['median']
    std_approx = (meta['max'] - meta['min']) / 4.0
    
    # Avoid zero std
    if std_approx <= 0:
        std_approx = 1.0
        
    samples = np.random.normal(loc=median, scale=std_approx, size=1000)
    samples = np.clip(samples, meta['min'], meta['max'])
    
    sns.kdeplot(samples, ax=ax, fill=True, color="#1f77b4", alpha=0.4, label="General Population")
    ax.axvline(user_value, color="#d62728", linestyle="--", linewidth=2.5, label=f"Your Input: {user_value}")
    
    ax.set_title(f"Distribution Comparison for {meta['label']}")
    ax.set_xlabel(meta['label'])
    ax.set_ylabel("Density")
    ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    return fig
