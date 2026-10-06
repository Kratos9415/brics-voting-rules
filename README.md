# Weight Is Not Power: Designing Voting Rules for an Expanded BRICS

Replication code and data for the paper by Dhruv Maniyar (2026).

## Contents
- `brics_voting_analysis.py`: the full analysis. Downloads World Bank data and
  reproduces every table and figure in the paper.
- `dataset_used.csv`: the processed dataset used in the paper
  (World Bank WDI, accessed 30 September 2026).

## How to run
1. Open Google Colab (colab.research.google.com) or any Python 3 environment.
2. Install the libraries: `pip install numpy pandas matplotlib requests tabulate`
3. Run `brics_voting_analysis.py`. Results appear in an `output/` folder.

Set `OFFLINE = True` at the top of the script to run without internet access.
Settings such as the proposed threshold and the design constraints are also
at the top of the file.

## Data
All economic data come from the World Bank's World Development Indicators.
Iran's services trade is unreported in WDI and IMF data; the script sets it to
zero, and the paper shows the results are unaffected by any value up to
USD 100 billion.
