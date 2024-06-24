# Archetypal Analysis for Identification of New Substages in Sleep

## Project Overview
**Researchers**: Morten Mørup, Alexander Neergaard Zahid, Laura Rose

**Motivation**: 
Preclinical mice models are often used to study sleep disorders, as it is possible to mimic specific sleep phenotypes in mice through gene manipulation. This controlled environment allows for detailed study of sleep processes. During sleep studies, electroencephalogram (EEG), electromyogram (EMG), and video data are recorded over 24 hours to measure brain activity, muscle activity, and behavior. A trained sleep expert manually segments the sleep into Wake, NREM, and REM stages using 4-second windows.

Given the heterogeneity of sleep in mice, it has been hypothesized that the existing three sleep stages may not fully encompass the variability of sleep. Recent studies ([1] and [2]) have explored the identification of potential substages of sleep using various unsupervised clustering techniques. Identifying these substages could not only refine sleep study evaluations but might also facilitate the detection of disease-specific stages, potentially enhancing diagnosis significantly. The question remains whether it is possible to delineate additional substages of sleep in mice.

## Data
Our dataset comprises EEG and EMG recordings from approximately 150 mice, provided both as raw signals and as preprocessed features. The features include various EEG power bands—Delta (0.25–5 Hz), Theta (5–9 Hz), Alpha (9–12 Hz), Beta (12–20 Hz), and Gamma (20–50 Hz)—extracted from the frequency domain in 4-second windows.

## Setup Instructions

### Prerequisites
Ensure Python and all required packages are installed by running:

```bash
pip install -r requirements.txt
```

### Configuration
Before executing main.py, configure your config.json with the correct data path. Start by creating a config.json file based on the config.sample.json templat
```json
{
    "data_path": "path/to/your/data.csv",
    "narcolepsy_path": "path/to/your/narcolepsy.csv"
}
```
Replace "path/to/your/data.csv" with the actual path to your dataset's CSV file.

To run the main.py script from the src directory, navigate to the home directory, sleepy_mice, and execute:
```bash
python src/main.py
```

## Acknowledgments
1. Vasiliki-Maria Katsageorgiou, Diego Sona, Matteo Zanotto, Glenda Lassi, Celina Garcia-Garcia, Valter Tucci, and Vittorio Murino, "A novel unsupervised analysis of electrophysiological signals reveals new sleep substages in mice," PLOS Biology, vol. 16, no. 5, pp. 1–23, May 2018.
2. Farid Yaghouby and Sridhar Sunderam, "Segway: A simple framework for unsupervised sleep segmentation in experimental EEG recordings.," MethodsX, 2016.
3. Adele Cutler and Leo Breiman, “Archetypal analysis,” Technometrics, vol. 36, no. 4, pp. 338–347, 1994.
4. Morten Mørup and Lars Kai Hansen, “Archetypal analysis for machine learning and data mining,” Neurocomputing, vol. 80, pp. 54–63, 2012.


