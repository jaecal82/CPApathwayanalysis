
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

def main():
    # Load the dataset
    try:
        # Read with header=0 to get variable names
        df = pd.read_csv("Alternative CPA Pathways Survey_December 31, 2025_09.45.csv", header=0)
    except FileNotFoundError:
        print("Error: CSV file not found.")
        sys.exit(1)

    # The first two rows are metadata (Question Text and Import Id).
    # Row 0 is the header (variable names).
    # Row 1 is Question Text.
    # Row 2 is Import Id.
    # We drop Row 1 and Row 2.
    df_clean = df.iloc[2:].reset_index(drop=True)

    # --- Task 1: Analysis of Employment Status vs. Program Priorities ---
    print("Starting Task 1...")

    # Identify columns based on inspection
    # Prompt said Q25 ("Do you currently work in a CPA firm?"), but inspection showed Q47 matches that text.
    # Prompt said Q24_1...Q24_6 for rankings.

    col_employment = 'Q47'
    col_rankings = {
        'Q24_1': 'CPA Exam Preparation',
        'Q24_2': 'Networking Opportunities',
        'Q24_3': 'Interaction with Experienced Faculty',
        'Q24_4': 'Technical Accounting Skills',
        'Q24_5': 'Soft Skill Development',
        'Q24_6': 'Internship and Recruitment Opportunities'
    }

    # Verify columns exist
    if col_employment not in df_clean.columns:
        print(f"Warning: {col_employment} not found. Searching for 'Do you currently work in a CPA firm?'")
        # Fallback search if needed, but we confirmed Q47.
        pass

    # Create a subset for Task 1
    task1_df = df_clean[[col_employment] + list(col_rankings.keys())].copy()

    # Rename columns
    task1_df.rename(columns=col_rankings, inplace=True)
    task1_df.rename(columns={col_employment: 'Work in CPA Firm'}, inplace=True)

    # Convert ranking columns to numeric
    ranking_cols = list(col_rankings.values())
    for col in ranking_cols:
        task1_df[col] = pd.to_numeric(task1_df[col], errors='coerce')

    # Drop rows with missing values
    task1_df.dropna(subset=['Work in CPA Firm'] + ranking_cols, inplace=True)

    # Calculate mean rank grouped by Employment Status
    task1_means = task1_df.groupby('Work in CPA Firm')[ranking_cols].mean()

    # Visualization for Task 1
    plt.figure(figsize=(12, 8))
    # Melt for seaborn
    task1_melted = task1_df.melt(id_vars='Work in CPA Firm', value_vars=ranking_cols, var_name='Benefit', value_name='Rank')

    # Clustered bar chart
    sns.barplot(data=task1_melted, x='Benefit', y='Rank', hue='Work in CPA Firm', errorbar=None)
    plt.title('Average Rank of Graduate Program Benefits by Employment Status (Lower is Higher Priority)')
    plt.ylabel('Average Rank')
    plt.xlabel('Benefit Category')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Work in CPA Firm')
    plt.tight_layout()
    plt.savefig('task1_priorities.png')
    plt.close()

    print("Task 1 completed. Saved task1_priorities.png")

    # --- Task 2: Analysis of Program Type vs. Lifetime Earnings Belief ---
    print("Starting Task 2...")

    # Prompt said Q57 ("Are you enrolled in an MAcc, MBA, or other graduate program?"), but inspection showed Q58 matches that text.
    # Inspection shows Q55 is for non-MAcc/MBA students (or non-enrolled). Q44 is for enrolled MAcc/MBA students.
    # We use Q44 for enrolled students.

    col_program = 'Q58'
    col_earnings = 'Q44'

    # Create subset
    task2_df = df_clean[[col_program, col_earnings]].copy()
    task2_df.rename(columns={col_program: 'Program Type', col_earnings: 'Lifetime Earnings Belief'}, inplace=True)

    # Filter for MAcc and MBA
    task2_df = task2_df[task2_df['Program Type'].isin(['MAcc', 'MBA'])]

    # Drop missing values
    task2_df.dropna(inplace=True)

    # Define order for Likert scale
    likert_order = [
        'Definitely yes',
        'Probably yes',
        'Might or might not',
        'Probably not',
        'Definitely not'
    ]
    # Filter to ensure only these values (or present values) are used and ordered correctly
    # Check present values
    present_values = task2_df['Lifetime Earnings Belief'].unique()
    # Create ordered categorical type if possible, or just reindex the crosstab

    # Crosstab
    task2_crosstab = pd.crosstab(task2_df['Program Type'], task2_df['Lifetime Earnings Belief'], normalize='index') * 100

    # Reorder columns if they exist in the order list
    existing_order = [val for val in likert_order if val in task2_crosstab.columns]
    # Append any others that might be there but not in list (unlikely based on previous check)
    others = [c for c in task2_crosstab.columns if c not in existing_order]
    final_order = existing_order + others
    task2_crosstab = task2_crosstab[final_order]

    # Visualization for Task 2
    # Stacked horizontal bar chart
    ax = task2_crosstab.plot(kind='barh', stacked=True, figsize=(12, 6), colormap='RdYlGn_r')
    # RdYlGn_r: Red (Bad) to Green (Good). Wait, Definitely Yes should be Green.
    # Likert order: Definitely Yes (Green) -> Definitely Not (Red).
    # If list is [Def Yes, Prob Yes, Might, Prob Not, Def Not]
    # RdYlGn_r goes Red -> Green.
    # We want Green -> Red. So 'RdYlGn' (reversed?) No, RdYlGn is Red to Green.
    # We want 'RdYlGn_r' (Green to Red).
    # Let's check colormap direction. 'RdYlGn': Red (low) to Green (high).
    # If index 0 is 'Definitely Yes', we want that to be Green.
    # So we want the reversed colormap if it maps index 0 to Green.
    # Usually colormaps map low values (start) to one color and high (end) to another.
    # I'll stick to default or a qualitative map if unsure, but diverging is nice.
    # Let's try 'viridis' or just default to avoid confusion, or 'RdYlBu'.

    plt.title('Belief in Higher Lifetime Earnings by Program Type')
    plt.xlabel('Percentage')
    plt.ylabel('Program Type')
    plt.legend(title='Belief Level', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('task2_earnings.png')
    plt.close()

    print("Task 2 completed. Saved task2_earnings.png")

    # --- Summary Output ---
    print("Generating summary...")
    with open('results_summary.csv', 'w') as f:
        f.write("Task 1: Mean Ranks by Employment Status (Lower = Higher Priority)\n")
        task1_means.to_csv(f)
        f.write("\n")
        f.write("Task 2: Percentage Belief in Higher Lifetime Earnings by Program Type\n")
        task2_crosstab.to_csv(f)

    print("Summary saved to results_summary.csv")

if __name__ == "__main__":
    main()
